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
  image_description: string;
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
  const server = new McpServer({ name: "meme-creator", version: "1.0.1" });

  /** find_meme: query Pinecone by text and return top match with text-region descriptions */
  server.registerTool(
    "find_meme",
    {
      title: "Find a meme template",
      description:
        "Use this when the user describes a meme they want and you need the best matching template and text regions. If user does not have exact meme they are referring to then describe the intention of the meme. Do not use if the user already provided meme_id or only needs text rendered—use create_meme instead.",
      inputSchema: { query: z.string().min(1) },
      outputSchema: {
        results: z
          .array(
            z.object({
              meme_id: z.string(),
              title: z.string().optional(),
              image_url: z.string(),
              image_description: z.string(),
              text_regions: z
                .array(
                  z.object({
                    id: z.string(),
                    description: z.string(),
                  })
                )
                .default([]),
            })
          )
          .default([]),
      },
      _meta: {
        // "openai/outputTemplate": "ui://widget/kanban-board.html",
        "openai/toolInvocation/invoking": "Searching for meme templates",
        "openai/toolInvocation/invoked": "Meme templates found.",
      },
      annotations: {
        title: "Search meme templates",
        readOnlyHint: true,
        destructiveHint: false,
        openWorldHint: false,
        idempotentHint: true,
      },
    },
    async ({ query }) => {
      const results = await memeSearch.search(query, 5);
      if (!results.length) {
        const empty = { results: [] as any[] };
        return {
          _meta: { "openai/toolInvocation/invoked": "No meme templates found" },
          content: [{ type: "text", text: JSON.stringify(empty) }],
          structuredContent: empty,
        } as any;
      }

      const payload = results.map((match) => {
        const template = memes[match.meme_id];
        return {
          meme_id: match.meme_id,
          title: match.title ?? "",
          image_url: template?.image_url ?? "",
          image_description: template?.image_description ?? "",
          text_regions: match.text_regions,
        };
      });
      const structuredContent = { results: payload };
      return {
        _meta: { "openai/toolInvocation/invoked": `Found ${payload.length} matching meme templates` },
        content: [
          { type: "text", text: JSON.stringify(structuredContent), annotations: { audience: ["assistant"] } },
        ],
        structuredContent,
      } as any;
    }
  );

  /** create_meme: render provided texts into selected regions; not all descriptions are required */
  server.registerTool(
    "create_meme",
    {
      title: "Create meme image",
      description:
        "Use this when you already have a meme_id from the find_meme tool. User may have provided text regions, or the assistant may provide the text regions. You may use none or all of the text regions provided by the find_meme tool. Do not use for discovery; call find_meme first. Display the meme image at generated_url in the response.",
      inputSchema: {
        meme_id: z.string(),
        texts: z.array(z.object({ id: z.string(), text: z.string().min(1) })),
      },
      outputSchema: {
        meme_id: z.string(),
        generated_key: z.string(),    // saved PNG key in S3
        generated_url: z.string(),    // signed URL to fetch PNG
      },
      _meta: {
        // "openai/outputTemplate": "ui://widget/kanban-board.html",
        "openai/toolInvocation/invoking": "Making the meme",
        "openai/toolInvocation/invoked": "Meme ready.",
      },
      annotations: {
        title: "Render meme with text",
        readOnlyHint: true,
        destructiveHint: false,
        openWorldHint: false,
        idempotentHint: true,
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
      const { url: generatedUrl, key: generatedKey } = await s3.storeGeneratedImage(meme_id, pngBuffer);
      console.log(`Uploaded generated meme to S3: ${generatedUrl}`);

      const out = {
        meme_id,
        generated_key: generatedKey,
        generated_url: generatedUrl,
      };

      return {
        content: [
          {
            type: "text",
            text: "Meme ready!",
            annotations: { audience: ["user", "assistant"] },
          },
          {
            type: "resource_link",
            uri: generatedUrl,
            mimeType: "image/png",
            name: `generated-meme-${meme_id}`,
            title: `Generated meme ${meme_id}`,
            annotations: { audience: ["user"] },
          },
        ],
        structuredContent: out,
      };
    }
  );

  return server;
}

const server = build_server();
// app.post('/mcp', async (req, res) => {
//   // Create a new transport for each request to prevent request ID collisions
//   const transport = new StreamableHTTPServerTransport({
//     sessionIdGenerator: undefined,
//     enableJsonResponse: true
//   });

//   res.on('close', () => {
//     transport.close();
//   });

//   await server.connect(transport);
//   await transport.handleRequest(req, res, req.body);
// });
app.all("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
  res.on("close", () => transport.close());
  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

app.get('/', (req, res) => {
  res.send('Im alive');
});

const port = parseInt(PORT || "3000");
app.listen(port, () => {
  console.log(`Demo MCP Server running on port ${port}`);
}).on('error', error => {
  console.error('Server error:', error);
  process.exit(1);
});

export default app;