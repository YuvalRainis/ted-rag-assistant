// check_connectivity.mjs
import dotenv from "dotenv";
dotenv.config();
import { Pinecone } from "@pinecone-database/pinecone";

// הגדרות משתנות (ודאי שהן תואמות לערכים ב-.env)
const EMBED_URL = "https://api.llmod.ai/v1/embeddings";
const EMBED_MODEL = "RPRTHPB-text-embedding-3-small";
const EMBEDDING_DIM = 1536;

// טעינת משתני סביבה
const apiKey = process.env.LLM_API_KEY;
const pineconeKey = process.env.PINECONE_API_KEY;
const indexName = process.env.PINECONE_INDEX_NAME;

if (!apiKey || !pineconeKey || !indexName) {
  console.error("❌ ERROR: Missing one or more environment variables (LLM_API_KEY, PINECONE_API_KEY, or PINECONE_INDEX_NAME).");
  process.exit(1);
}

// אתחול לקוח Pinecone
const pc = new Pinecone({ apiKey: pineconeKey });
const index = pc.index(indexName);

console.log("--- Starting Connectivity Check ---");

// --- בדיקה 1: Pinecone Connectivity ---
async function checkPinecone() {
  console.log("\n1. PINGING PINECONE INDEX...");
  try {
    const indexDescription = await index.describeIndexStats();
    console.log("✅ Pinecone Connection SUCCESS.");
    console.log(`   > Index Name: ${indexName}`);
    console.log(`   > Reported Dimensions: ${indexDescription.dimension}`);
    const vectorCount = indexDescription.namespaces['__default__']?.vectorCount || 0;
    console.log(`   > Reported Vector Count: ${vectorCount}`);
    
    if (indexDescription.totalVectorCount !== 8919) {
        console.warn(`   ⚠️ WARNING: Vector count (${indexDescription.totalVectorCount}) does not match the expected 8919. This confirms a mismatch between your script and the index state.`);
    }

  } catch (err) {
    console.error("❌ FATAL ERROR: Pinecone connectivity FAILED.");
    console.error("   > Possible cause: PINECONE_API_KEY or PINECONE_INDEX_NAME is incorrect/invalid.");
    console.error("   > Details:", err.message);
    process.exit(1);
  }
}

// --- בדיקה 2: LLMod Embedding Service ---
async function checkLLModEmbedding() {
  console.log("\n2. PINGING LLMOD EMBEDDING SERVICE...");
  const sampleText = "This is a test sentence for embedding connectivity.";
  try {
    const res = await fetch(EMBED_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: EMBED_MODEL,
        input: sampleText
      })
    });

    if (!res.ok) {
      // אם הקריאה נכשלה, הדפס את הסטטוס ואת תגובת השגיאה המלאה
      console.error(`❌ FATAL ERROR: LLMod Embedding FAILED with status ${res.status}.`);
      console.error("   > Possible cause: LLM_API_KEY is incorrect or service is down.");
      console.error("   > Full Error Response:", await res.text());
      process.exit(1);
    }

    const data = await res.json();
    const embedding = data?.data?.[0]?.embedding;
    
    if (!Array.isArray(embedding) || embedding.length !== EMBEDDING_DIM) {
        throw new Error(`Unexpected embedding dimension or format. Expected ${EMBEDDING_DIM}, got ${embedding?.length}`);
    }

    console.log("✅ LLMod Embedding SUCCESS.");
    console.log(`   > Received embedding of dimension: ${embedding.length}`);
    console.log("   > Note: This call consumed budget.");

  } catch (err) {
    console.error("❌ FATAL ERROR: LLMod Embedding check FAILED (Network/Parsing Error).");
    console.error("   > Details:", err.message);
    process.exit(1);
  }
}

// הפעלת כל הבדיקות ברצף
async function runChecks() {
    await checkPinecone();
    await checkLLModEmbedding();
    console.log("\n--- All Checks Complete ---");
    console.log("If both checks passed, the issue lies in the safeUpsert logic or Pinecone rate limiting/throttling.");
}

runChecks();