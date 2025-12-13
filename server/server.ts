import dotenv from "dotenv";
import express from "express";
dotenv.config();
import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import fs from "fs";
import { Image, buildTextBoxesFromOptions, type MemeTextOption } from "./image_service.js";
import { MemeS3Client } from "./s3Client.js";
import { MemeSearch } from "./memeSearch.js";

const {
  PORT = "3000",
  PINECONE_API_KEY,
  PINECONE_INDEX,
  PINECONE_NAMESPACE,
  AWS_ACCESS_KEY,
  AWS_SECRET_KEY,
  S3_BUCKET,
  S3_REGION,
  S3_STATIC_PREFIX = "static",
  S3_GENERATED_PREFIX = "generated",
  S3_SIGNED_URL_TTL_SECONDS = "900",
} = process.env;

if (!PINECONE_API_KEY || !PINECONE_INDEX || !PINECONE_NAMESPACE) {
  console.error("Missing env: PINECONE_API_KEY, PINECONE_INDEX, PINECONE_NAMESPACE");
  process.exit(1);
}

if (!S3_BUCKET || !S3_REGION) {
  console.error("Missing env: S3_BUCKET, S3_REGION");
  process.exit(1);
}

if (!AWS_ACCESS_KEY || !AWS_SECRET_KEY) {
  console.error("Missing env: AWS_ACCESS_KEY, AWS_SECRET_KEY");
  process.exit(1);
}

const signedUrlTtlSeconds = Number(S3_SIGNED_URL_TTL_SECONDS) || 900;

const app = express();
app.use(express.json({ limit: "5mb" }));

const memesUrl = new URL("./memes.json", import.meta.url);

type MemeTemplate = {
  filename: string;
  width?: number;
  height?: number;
  image_url: string;
  text_options?: MemeTextOption[];
};

function loadMemes(memesFile: string | URL): Record<string, MemeTemplate> {
  const displayPath = typeof memesFile === "string" ? memesFile : memesFile.toString();
  if (!fs.existsSync(memesFile)) {
    console.error(`memes.json not found at ${displayPath}`);
    throw new Error(`memes.json not found at ${displayPath}`);
  }
  const raw = fs.readFileSync(memesFile, "utf8");
  return JSON.parse(raw) as Record<string, MemeTemplate>;
}

const memes = loadMemes(memesUrl);

const memeSearch = new MemeSearch({
  apiKey: PINECONE_API_KEY,
  indexName: PINECONE_INDEX,
  namespace: PINECONE_NAMESPACE,
});
const s3 = new MemeS3Client({
  bucket: S3_BUCKET,
  region: S3_REGION,
  staticPrefix: S3_STATIC_PREFIX,
  generatedPrefix: S3_GENERATED_PREFIX,
  defaultTtlSeconds: signedUrlTtlSeconds,
  credentials: {
    accessKeyId: AWS_ACCESS_KEY,
    secretAccessKey: AWS_SECRET_KEY,
  },
});

function build_server() {
  const server = new McpServer({ name: "meme-creator", version: "1.0.0" });

  /** find_meme: query Pinecone by text and return top match with text-region descriptions */
  server.registerTool(
    "find_meme",
    {
      title: "Find a meme template",
      description:
        "Searches for the most appropriate meme template for the user based on the text query.",
      inputSchema: { query: z.string().min(1) },
      outputSchema: {
        meme_id: z.string(),
        title: z.string().optional(),
        image_url: z.string(),
        text_regions: z
          .array(
            z.object({
              id: z.string(),
              description: z.string(),
            })
          )
          .default([]),
      },
    },
    async ({ query }) => {
      const results = await memeSearch.search(query, 1);
      const match = results[0];
      if (!match) {
        const empty = { meme_id: "", title: "", image_url: "", text_regions: [] };
        return {
          content: [{ type: "text", text: JSON.stringify(empty) }],
          structuredContent: empty,
        } as any;
      }

      const template = memes[match.meme_id];
      const payload = {
        meme_id: match.meme_id,
        title: match.title ?? "",
        image_url: template?.image_url ?? "",
        text_regions: match.text_regions,
      };
      return {
        content: [{ type: "text", text: JSON.stringify(payload) }],
        structuredContent: payload,
      } as any;
    }
  );

  /** create_meme: render provided texts into selected regions; not all descriptions are required */
  server.registerTool(
    "create_meme",
    {
      title: "Create meme image",
      description: "Creates a meme image from a meme template and user texts.",
      inputSchema: {
        meme_id: z.string(),
        texts: z.array(z.object({ id: z.string(), text: z.string().min(1) })),
      },
      outputSchema: {
        meme_id: z.string(),
        image_url: z.string(),        // from memes.json
        generated_key: z.string(),    // saved PNG key in S3
        generated_url: z.string(),    // signed URL to fetch PNG
      },
    },
    async ({ meme_id, texts }) => {
      // 1) Resolve template from cache
      const meme = memes[meme_id];
      console.log(`Found meme template: ${meme_id}`);
      if (!meme) throw new Error(`meme_id not found in memes.json: ${meme_id}`);

      // 2) Load base image from S3
      const staticKey = s3.staticKey(meme.filename);
      console.log(`Loading image from S3 key: ${staticKey}`);
      const baseBuffer = await s3.fetchObjectBuffer(staticKey);
      const image = await Image.fromBuffer(baseBuffer);

      // 3) Prepare canvas
      const w = meme.width ?? image.width;
      const h = meme.height ?? image.height;
      image.resizeTo(w, h);

      // 4) Render requested text into template regions
      const options = Array.isArray(meme.text_options) ? meme.text_options : [];
      console.log(`Processing ${options.length} text regions`);
      const textBoxes = buildTextBoxesFromOptions(texts, options);
      console.log(`Rendering ${textBoxes.length} text entries`);
      if (textBoxes.length) {
        await image.addTexts(textBoxes);
      }

      // 5) Upload PNG to S3 and return signed URL
      const pngBuffer = await image.toPngBuffer();
      const { key: generatedKey, url: generatedUrl } = await s3.storeGeneratedImage(meme_id, pngBuffer);
      console.log(`Uploaded generated meme to S3: ${generatedKey}`);

      const out = {
        meme_id,
        meme_image_url: generatedUrl,
      };

      return {
        content: [{ type: "text", text: JSON.stringify(out) }],
        structuredContent: out,
      };
    }
  );

  return server;
}

const server = build_server();
app.post('/mcp', async (req, res) => {
  // Create a new transport for each request to prevent request ID collisions
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
    enableJsonResponse: true
  });

  res.on('close', () => {
    transport.close();
  });

  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

const port = parseInt(PORT || "3000");
app.listen(port, () => {
  console.log(`Demo MCP Server running on http://localhost:${port}/mcp`);
}).on('error', error => {
  console.error('Server error:', error);
  process.exit(1);
});

export default app;