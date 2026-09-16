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

if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

const upload = multer({
  dest: uploadDir + "/",
});

app.use(cors());
app.use(express.json());

// =====================================================
// Gemini
// =====================================================

const googleGenerativeAI = new GoogleGenerativeAI(
  process.env.GEMINI_API_KEY
);

const model = googleGenerativeAI.getGenerativeModel({
  model: "gemini-3.8-flash",
});

// =====================================================
// Gemini Embeddings
// =====================================================

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-embedding-2",
});

// =====================================================
// Chroma Cloud
// =====================================================

const chromaClient = new CloudClient({
  apiKey: process.env.CHROMA_API_KEY,
  path: "https://api.trychroma.com",
});

let collection = null;

// =====================================================
// Upload + Index PDF
// =====================================================

app.post("/summarize", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      error: "No file uploaded",
    });
  }

  const filePath = req.file.path;

  try {
    // -------------------------------------------------
    // 1. Read PDF
    // -------------------------------------------------

    const pdfBuffer = fs.readFileSync(filePath);

    const pdfData = await pdfParse(pdfBuffer);

    const text = pdfData.text || "";

    console.log("=================================");
    console.log("PDF PROCESSING");
    console.log("=================================");

    console.log("Extracted text length:", text.length);
    console.log("Number of PDF pages:", pdfData.numpages);

    // -------------------------------------------------
    // 2. Check extracted text
    // -------------------------------------------------

    if (!text.trim()) {
      return res.status(400).json({
        error:
          "No readable text was extracted from this PDF. The PDF may be scanned/image-based.",
      });
    }

    // -------------------------------------------------
    // 3. Split text into chunks
    // -------------------------------------------------

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await splitter.createDocuments([text]);

    console.log("Number of chunks:", docs.length);

    if (!docs.length) {
      return res.status(400).json({
        error: "No text chunks were created from the PDF.",
      });
    }

    // -------------------------------------------------
    // 4. Create / Get Chroma Collection
    // -------------------------------------------------

    collection = await chromaClient.getOrCreateCollection({
      name: "pdf_chunks",
    });

    console.log("Chroma collection ready.");

    // -------------------------------------------------
    // 5. Generate embeddings
    // -------------------------------------------------

    const vectors = [];

    for (let i = 0; i < docs.length; i++) {
      const content = docs[i].pageContent?.trim();

      if (!content) {
        console.log(`Skipping empty chunk ${i}`);
        continue;
      }

      console.log(
        `Generating embedding ${i + 1}/${docs.length}...`
      );

      const vector = await embeddings.embedQuery(content);

      console.log(
        `Embedding ${i + 1} size:`,
        vector?.length || 0
      );

      if (!vector || vector.length === 0) {
        console.log(`Skipping empty embedding for chunk ${i}`);
        continue;
      }

      vectors.push({
        id: `chunk_${Date.now()}_${i}`,
        values: vector,
        metadata: {
          text: content,
        },
      });
    }

    // -------------------------------------------------
    // 6. Check vectors
    // -------------------------------------------------

    console.log("=================================");
    console.log("VECTOR DATA");
    console.log("=================================");

    console.log("Vectors:", vectors.length);

    if (vectors.length === 0) {
      return res.status(400).json({
        error: "No embeddings were generated from the PDF.",
      });
    }

    // -------------------------------------------------
    // 7. Prepare Chroma arrays
    // -------------------------------------------------

    const ids = vectors.map((vector) => vector.id);

    const vectorEmbeddings = vectors.map(
      (vector) => vector.values
    );

    const metadatas = vectors.map(
      (vector) => vector.metadata
    );

    console.log("IDs:", ids.length);
    console.log("Embeddings:", vectorEmbeddings.length);
    console.log("Metadatas:", metadatas.length);

    // -------------------------------------------------
    // 8. Validate Chroma data
    // -------------------------------------------------

    if (
      ids.length === 0 ||
      vectorEmbeddings.length === 0 ||
      metadatas.length === 0
    ) {
      throw new Error(
        "Cannot add empty data to Chroma."
      );
    }

    if (
      ids.length !== vectorEmbeddings.length ||
      ids.length !== metadatas.length
    ) {
      throw new Error(
        `Chroma data mismatch: ids=${ids.length}, embeddings=${vectorEmbeddings.length}, metadatas=${metadatas.length}`
      );
    }

    // -------------------------------------------------
    // 9. Add vectors to Chroma
    // -------------------------------------------------

    console.log("Adding vectors to Chroma...");

    await collection.add({
      ids: ids,
      embeddings: vectorEmbeddings,
      metadatas: metadatas,
    });

    console.log("Successfully indexed PDF.");

    // -------------------------------------------------
    // 10. Generate summary
    // -------------------------------------------------

    const summaryPrompt = `
Summarize the following PDF text in 5-6 bullet points.

PDF CONTENT:

${text.slice(0, 5000)}
`;

    const summaryResult =
      await model.generateContent(summaryPrompt);

    const summary =
      summaryResult.response.text();

    // -------------------------------------------------
    // 11. Response
    // -------------------------------------------------

    res.json({
      summary,
      message: "PDF indexed successfully!",
      chunks: vectors.length,
    });

  } catch (error) {
    console.error(
      "Error processing PDF:",
      error
    );

    res.status(500).json({
      error: "Failed to process PDF",
      details: error.message,
    });

  } finally {
    // -------------------------------------------------
    // Delete uploaded file
    // -------------------------------------------------

    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  }
});

// =====================================================
// Chat with PDF
// =====================================================

app.post("/chat", async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({
      error: "Missing question",
    });
  }

  if (!collection) {
    return res.status(400).json({
      error: "No PDF indexed yet",
    });
  }

  try {
    // -------------------------------------------------
    // 1. Create question embedding
    // -------------------------------------------------

    console.log("Creating question embedding...");

    const qEmbedding =
      await embeddings.embedQuery(question);

    if (!qEmbedding || qEmbedding.length === 0) {
      throw new Error(
        "Question embedding is empty."
      );
    }

    console.log(
      "Question embedding size:",
      qEmbedding.length
    );

    // -------------------------------------------------
    // 2. Search Chroma
    // -------------------------------------------------

    const results = await collection.query({
      queryEmbeddings: [qEmbedding],
      nResults: 3,
    });

    console.log("Chroma query completed.");

    // -------------------------------------------------
    // 3. Extract context
    // -------------------------------------------------

    const metadataResults =
      results.metadatas?.[0] || [];

    const context = metadataResults
      .filter(Boolean)
      .map((metadata) => metadata.text)
      .filter(Boolean)
      .join("\n\n");

    if (!context) {
      return res.status(404).json({
        error:
          "No relevant content found in the PDF.",
      });
    }

    // -------------------------------------------------
    // 4. Generate answer
    // -------------------------------------------------

    const prompt = `
You are an AI assistant answering questions about a PDF.

Use ONLY the context provided below.

If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded PDF."

Context:
${context}

Question:
${question}

Answer:
`;

    const result =
      await model.generateContent(prompt);

    const answer =
      result.response.text();

    // -------------------------------------------------
    // 5. Return answer
    // -------------------------------------------------

    res.json({
      answer,
    });

  } catch (error) {
    console.error(
      "Chat error:",
      error
    );

    res.status(500).json({
      error: "Failed to generate response",
      details: error.message,
    });
  }
});

// =====================================================
// Server
// =====================================================

app.listen(5000, () => {
  console.log(
    "🚀 Backend running at http://localhost:5000"
  );
});
