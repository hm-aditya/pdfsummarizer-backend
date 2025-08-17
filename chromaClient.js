import { CloudClient } from "chromadb";
import dotenv from "dotenv";

dotenv.config();

const client = new CloudClient({
  apiKey: process.env.CHROMA_API_KEY, // required
  tenant: process.env.CHROMA_TENANT,  // required
});

const chromaCollectionPromise = client.getOrCreateCollection({
  name: "my_collection",
});

export default chromaCollectionPromise;
