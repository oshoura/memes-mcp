import { randomUUID } from "crypto";
import { Readable } from "stream";
import {
  GetObjectCommand,
  PutObjectCommand,
  type GetObjectCommandOutput,
  type PutObjectCommandInput,
  S3Client,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

type S3Prefixes = {
  staticPrefix?: string;
  generatedPrefix?: string;
};

type MemeS3ClientConfig = S3Prefixes & {
  bucket: string;
  region: string;
  defaultTtlSeconds?: number;
};

function normalizePrefix(prefix?: string): string {
  if (!prefix) return "";
  const trimmed = prefix.replace(/^\/+/, "").replace(/\/+$/, "");
  return trimmed.length ? `${trimmed}/` : "";
}

async function streamToBuffer(body: GetObjectCommandOutput["Body"]): Promise<Buffer> {
  if (!body) throw new Error("S3 object body is empty");

  if (body instanceof Readable) {
    return new Promise<Buffer>((resolve, reject) => {
      const chunks: Buffer[] = [];
      body.on("data", (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
      body.on("error", reject);
      body.on("end", () => resolve(Buffer.concat(chunks)));
    });
  }

  if (typeof (body as any).arrayBuffer === "function") {
    const ab = await (body as any).arrayBuffer();
    return Buffer.from(ab);
  }

  // Fallback for Web streams in newer AWS SDK runtimes
  if (typeof (body as any).getReader === "function") {
    const response = new Response(body as any);
    const ab = await response.arrayBuffer();
    return Buffer.from(ab);
  }

  throw new Error("Unsupported S3 body type");
}

export class MemeS3Client {
  private readonly client: S3Client;
  private readonly bucket: string;
  private readonly staticPrefix: string;
  private readonly generatedPrefix: string;
  private readonly defaultTtlSeconds: number;

  constructor(config: MemeS3ClientConfig) {
    if (!config.bucket) throw new Error("S3 bucket is required");
    if (!config.region) throw new Error("S3 region is required");

    this.bucket = config.bucket;
    this.client = new S3Client({ region: config.region });
    this.staticPrefix = normalizePrefix(config.staticPrefix ?? "static");
    this.generatedPrefix = normalizePrefix(config.generatedPrefix ?? "generated");
    this.defaultTtlSeconds = config.defaultTtlSeconds ?? 900;
  }

  staticKey(filename: string): string {
    return `${this.staticPrefix}${filename}`;
  }

  generatedKey(filename: string): string {
    return `${this.generatedPrefix}${filename}`;
  }

  createGeneratedFilename(memeId: string): string {
    return `${memeId}-${Date.now()}-${randomUUID()}.png`;
  }

  async storeGeneratedImage(
    memeId: string,
    buffer: Buffer,
    options?: { cacheControl?: string; contentType?: string }
  ): Promise<{ key: string; url: string; filename: string }> {
    const filename = this.createGeneratedFilename(memeId);
    const key = this.generatedKey(filename);
    await this.putObject({
      key,
      body: buffer,
      contentType: options?.contentType ?? "image/png",
      cacheControl: options?.cacheControl ?? "public, max-age=31536000, immutable",
    });
    const url = await this.getSignedGetUrl(key);
    return { key, url, filename };
  }

  async fetchObjectBuffer(key: string): Promise<Buffer> {
    const res = await this.client.send(
      new GetObjectCommand({
        Bucket: this.bucket,
        Key: key,
      })
    );
    return streamToBuffer(res.Body);
  }

  async putObject(params: { key: string; body: Buffer; contentType: string; cacheControl?: string }): Promise<void> {
    const payload: PutObjectCommandInput = {
      Bucket: this.bucket,
      Key: params.key,
      Body: params.body,
      ContentType: params.contentType,
      CacheControl: params.cacheControl,
    };
    await this.client.send(new PutObjectCommand(payload));
  }

  async getSignedGetUrl(key: string, expiresInSeconds?: number): Promise<string> {
    return getSignedUrl(
      this.client,
      new GetObjectCommand({ Bucket: this.bucket, Key: key }),
      { expiresIn: expiresInSeconds ?? this.defaultTtlSeconds }
    );
  }
}

