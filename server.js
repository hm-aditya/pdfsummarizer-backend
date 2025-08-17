// server.js
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const pdfParse = require("pdf-parse");
const dotenv = require("dotenv");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { CloudClient } = require("chromadb");

dotenv.config();
const app = express();

const uploadDir = "uploads";
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
const upload = multer({ dest: uploadDir + "/" });

app.use(cors());
app.use(express.json());

// Gemini model
const googleGenerativeAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = googleGenerativeAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// Embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});

// Chroma Cloud Client
const chromaClient = new CloudClient({
  apiKey: process.env.CHROMA_API_KEY, // 🔑 get from Chroma Cloud
  path: "https://api.trychroma.com", // default Chroma Cloud endpoint
});

let collectionPromise;

// ----------------------
// Upload + Index PDF
// ----------------------
app.post("/summarize", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  const filePath = req.file.path;
  try {
    const pdfBuffer = fs.readFileSync(filePath);
    const { text } = await pdfParse(pdfBuffer);

    // Split into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const docs = await splitter.createDocuments([text]);

    // Create / get collection
    collectionPromise = await chromaClient.getOrCreateCollection({
      name: "pdf_chunks",
    });

    // Embed and insert chunks into Chroma
    const vectors = await Promise.all(
      docs.map(async (doc, i) => ({
        id: `chunk_${Date.now()}_${i}`,
        values: await embeddings.embedQuery(doc.pageContent),
        metadata: { text: doc.pageContent },
      }))
    );

    await collectionPromise.add({
      ids: vectors.map(v => v.id),
      embeddings: vectors.map(v => v.values),
      metadatas: vectors.map(v => v.metadata),
    });

    // Generate a quick summary
    const summaryPrompt = `Summarize the following PDF text in 5-6 bullet points:\n\n${text.slice(0, 5000)}`;
    const summaryResult = await model.generateContent(summaryPrompt);
    const summary = summaryResult.response.text();

    res.json({ summary, message: "PDF indexed successfully!" });
  } catch (error) {
    console.error("Error processing PDF:", error);
    res.status(500).json({ error: "Failed to process PDF", details: error.message });
  } finally {
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  }
});

// ----------------------
// Chat with PDF (RAG)
// ----------------------
app.post("/chat", async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "Missing question" });
  if (!collectionPromise) return res.status(400).json({ error: "No PDF indexed yet" });

  try {
    const collection = await collectionPromise;

    // Embed the question
    const qEmbedding = await embeddings.embedQuery(question);

    // Query top 3 relevant chunks
    const results = await collection.query({
      queryEmbeddings: [qEmbedding],
      nResults: 3,
    });

    const context = results.metadatas[0]
      .map(meta => meta.text)
      .join("\n\n");

    const prompt = `
You are an AI assistant. Use the context below to answer the question.

Context:
${context}

Question: ${question}
Answer:
    `;

    const result = await model.generateContent(prompt);
    res.json({ answer: result.response.text() });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "Failed to generate response", details: error.message });
  }
});

app.listen(5000, () => console.log("🚀 Backend running at http://localhost:5000"));
