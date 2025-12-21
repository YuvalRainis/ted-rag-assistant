import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import csvParser from "csv-parser";
import { Pinecone } from "@pinecone-database/pinecone";

import { fileURLToPath } from "url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");

// loading env variables from .env file in project root
dotenv.config({ path: path.join(projectRoot, ".env") });

const EMBED_URL = "https://api.llmod.ai/v1/embeddings";
const EMBED_MODEL = "RPRTHPB-text-embedding-3-small";
const EMBEDDING_DIM = 1536;
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200; // 20% overlap
const TARGET_NAMESPACE = "__default__";

const csvPath = path.join(projectRoot, "data", "ted_talks_en.csv");
const progressFile = path.join(projectRoot, "progress.log");

const apiKey = process.env.LLM_API_KEY;
const pineconeKey = process.env.PINECONE_API_KEY;
const indexName = process.env.PINECONE_INDEX_NAME;

if (!apiKey || !pineconeKey || !indexName) {
  console.error("FATAL ERROR: Missing environment variables (LLM_API_KEY, PINECONE_API_KEY, or PINECONE_INDEX_NAME). Please check your .env file in the root directory and ensure the format is 'KEY=VALUE' without spaces.");
  process.exit(1);
}

const pc = new Pinecone({ apiKey: pineconeKey });
const index = pc.index(indexName).namespace(TARGET_NAMESPACE);

console.log("--- Starting RAG Indexing Process ---");
console.log("Reading file from:", csvPath);

// Helper: Sleep
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ------------------------------------
// 1. Safe Embedding (LLMod)
// ------------------------------------
async function getEmbedding(text, retries = 5) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await fetch(EMBED_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: EMBED_MODEL,
          input: text
        })
      });

      if (!res.ok) {
        console.warn(`Embedding failed (attempt ${attempt}) [Status ${res.status}]:`, await res.text());
        throw new Error(`Embedding API failed with status ${res.status}`);
      }

      const data = await res.json();
      return data.data[0].embedding;
    } catch (err) {
      console.log(`Retrying embedding attempt ${attempt}/${retries}...`);
      await sleep(1000 * attempt);
    }
  }
  throw new Error("Failed to embed after all retries.");
}

// ------------------------------------
// 2. Safe Pinecone Upsert (The critical part)
// ------------------------------------
async function safeUpsert(vectors, retries = 5) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      // ניסיון Upsert רגיל
      const response = await index.upsert(vectors);
      
      // בדיקה מפורשת של תגובה מוצלחת (במקרה של כשל שקט)
      // אם התגובה ריקה או שגויה, עדיין נרצה לדעת.
      if (!response) {
        console.warn(`Pinecone upsert returned empty response on attempt ${attempt}. Continuing.`);
      }
      return;
    } catch (err) {
      // הדפסה אגרסיבית של השגיאה המלאה ללכידת הכשל השקט
      console.error(`FATAL UPSERT ERROR on attempt ${attempt}:`);
      console.error("Vector ID example:", vectors[0].id);
      console.error("Full Error Object:", err);
      console.warn(`Pinecone upsert failed (attempt ${attempt}): ${err.message}`);
      await sleep(2000 * attempt);
    }
  }
  throw new Error("Failed to upsert after multiple retries.");
}

// ------------------------------------
// 3. Chunking
// ------------------------------------
function chunkText(text) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = start + CHUNK_SIZE;
    const chunk = text.slice(start, end);
    chunks.push(chunk);
    start += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  return chunks;
}

// ------------------------------------
// MAIN Function
// ------------------------------------
async function run() {
  // Read progress if exists
  let startFrom = 1;
  if (fs.existsSync(progressFile)) {
    const saved = parseInt(fs.readFileSync(progressFile, "utf-8"));
    if (!isNaN(saved)) {
      startFrom = saved + 1;
      console.log(`Resuming from row #${startFrom} based on progress.log`);
    }
  }

  console.log(`Starting indexing from row #${startFrom} onward...`);

  const talks = [];
  await new Promise((resolve) => {
    fs.createReadStream(csvPath)
      .pipe(csvParser())
      .on("data", (row) => talks.push(row))
      .on("error", (err) => {
        console.error("FATAL ERROR: Error reading CSV file:", err);
        process.exit(1);
      })
      .on("end", resolve);
  });

  console.log(`Total talks in dataset: ${talks.length}`);

  const remainingTalks = talks.slice(startFrom);
  console.log(`Will process ${remainingTalks.length} talks.`);

  let vectorCounter = 0;
  let currentRow = startFrom;

  for (const row of remainingTalks) {
    const { talk_id, title, speaker_1, topics, event, description, transcript } = row;
    if (!transcript) {
      currentRow++;
      continue;
    }

    const chunks = chunkText(transcript);
    for (let i = 0; i < chunks.length; i++) {
      const chunkTextValue = chunks[i];
      const embedding = await getEmbedding(chunkTextValue);
      const vectorId = `${talk_id}-${i}`; 

      await safeUpsert([
        {
          id: vectorId,
          values: embedding,
          metadata: { talk_id, title, speaker: speaker_1, topics, event, description, text: chunkTextValue }
        }
      ]);

      vectorCounter++;
    }

    // Print and save progress every 5 talks
    if (currentRow % 5 === 0) {
      console.log(`Processed talk row #${currentRow} (${vectorCounter} total vectors added this run)`);
      fs.writeFileSync(progressFile, currentRow.toString());
    }

    currentRow++;
  }

  console.log("Indexing complete!");
  console.log("Total vectors inserted in this run:", vectorCounter);
  fs.writeFileSync(progressFile, currentRow.toString());
}

run().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});