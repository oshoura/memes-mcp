import { Pinecone, type Index } from "@pinecone-database/pinecone";

export type MemeSearchResult = {
  meme_id: string;
  title: string;
  text_regions: Array<{ id: string; description: string }>;
};

type MemeSearchConfig = {
  apiKey: string;
  indexName: string;
  namespace: string;
};

export class MemeSearch {
  private readonly index: Index;
  private readonly namespace: ReturnType<Index["namespace"]>;

  constructor(config: MemeSearchConfig) {
    const client = new Pinecone({ apiKey: config.apiKey });
    this.index = client.index(config.indexName);
    this.namespace = this.index.namespace(config.namespace);
  }

  async search(query: string, topK = 1): Promise<MemeSearchResult[]> {
    const response = await this.namespace.searchRecords({
      query: { topK, inputs: { text: query } },
      fields: ["display_name", "text_options"],
    });

    const hits = response.result?.hits ?? [];
    return hits.map((hit: any) => {
      const fields = (hit as any).fields as { display_name?: string; text_options?: any[] };
      return {
        meme_id: (hit as any)._id as string,
        title: fields?.display_name ?? "",
        text_regions: (fields?.text_options ?? []).map((option: any, index: number) => ({
          id: String(index),
          description: option,
        })),
      };
    });
  }
}

