import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";

export const runtime = "nodejs";

// --- Required System Prompt ---
const SYSTEM_PROMPT = `
You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided TED data.” Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.If the user asks for multiple results, list them clearly with numbered points.
`.trim();

const EMBEDDING_DIM = 1536;
const TOP_K = 15;

// --- ENV ---
const pineconeApiKey = process.env.PINECONE_API_KEY;
const pineconeIndexName = process.env.PINECONE_INDEX_NAME;

const llmApiKey = process.env.LLM_API_KEY;              // FIXED
const llmodBaseUrl = process.env.LLMOD_BASE_URL || "https://api.llmod.ai";

if (!pineconeApiKey || !pineconeIndexName) {
  console.error("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in env");
}

const pineconeClient = new Pinecone({ apiKey: pineconeApiKey! });
const index = pineconeClient.index(pineconeIndexName!);

// --- Get Embedding from LLMod ---
async function getEmbedding(text: string): Promise<number[]> {
  if (!llmApiKey) {
    throw new Error("Missing LLM_API_KEY in env");
  }

  const res = await fetch(`${llmodBaseUrl}/v1/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${llmApiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "RPRTHPB-text-embedding-3-small",
      input: text,
    }),
  });

  if (!res.ok) {
    console.error("Embedding error:", await res.text());
    throw new Error(`Embedding request failed with status ${res.status}`);
  }

  const data = await res.json();
  const embedding = data?.data?.[0]?.embedding;

  if (!Array.isArray(embedding) || embedding.length !== EMBEDDING_DIM) {
    throw new Error("Unexpected embedding dimension from LLMod");
  }

  return embedding;
}

// --- Call RPRTHPB-gpt-5-mini ---
async function callTedModel(systemPrompt: string, userPrompt: string): Promise<string> {
  if (!llmApiKey) {
    throw new Error("Missing LLM_API_KEY in env");
  }

  const res = await fetch(`${llmodBaseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${llmApiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "RPRTHPB-gpt-5-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      // temperature intentionally omitted (model does not support)
    }),
  });

  if (!res.ok) {
    console.error("Chat error:", await res.text());
    throw new Error(`Chat request failed with status ${res.status}`);
  }

  const data = await res.json();
  return String(data?.choices?.[0]?.message?.content ?? "").trim();
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    if (!body?.question || typeof body.question !== "string") {
      return NextResponse.json(
        { error: "Missing 'question' field in JSON body" },
        { status: 400 }
      );
    }

    const question = body.question.trim();

    if (!question) {
      return NextResponse.json(
        { error: "Question cannot be empty" },
        { status: 400 }
      );
    }

    // 1) Convert question to embedding
    const queryEmbedding = await getEmbedding(question);

    // 2) Query Pinecone
    const queryResult = await index.query({
      topK: TOP_K,
      vector: queryEmbedding,
      includeMetadata: true,
    });

    const matches = queryResult.matches ?? [];

    // 3) Build context (NOW WITH FULL METADATA)
    const context = matches.map((m) => {
      const meta: any = m.metadata || {};

      return {
        talk_id: meta.talk_id ?? "",
        title: meta.title ?? "",
        speaker: meta.speaker ?? "",
        topics: meta.topics ?? "",
        event: meta.event ?? "",
        description: meta.description ?? "",
        chunk: meta.text ?? "",
        score: m.score ?? 0,
      };
    });

    // 4) User Prompt sent to GPT model
    const userPrompt = `
Question: ${question}

Context:
${context
  .map(
    (c, idx) => `#${idx + 1}
Talk ID: ${c.talk_id}
Title: ${c.title}
Speaker: ${c.speaker}
Topics: ${c.topics}
Event: ${c.event}
Description: ${c.description}
Score: ${c.score}
Chunk: ${c.chunk}`
  )
  .join("\n\n")}

Remember: Answer strictly from the context above. If you cannot answer, say: "I don't know based on the provided TED data."
    `.trim();

    // 5) Ask the language model
    const modelAnswer = await callTedModel(SYSTEM_PROMPT, userPrompt);

    return NextResponse.json({
      response: modelAnswer,
      context,
      Augmented_prompt: {
        System: SYSTEM_PROMPT,
        User: userPrompt,
      },
    });
  } catch (err) {
    console.error("Error in /api/prompt:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
