import { NextResponse } from "next/server";

const CHUNK_SIZE = 1024;
const OVERLAP_RATIO = 0.2;
const TOP_K = 15;

export async function GET() {
  return NextResponse.json({
    chunk_size: CHUNK_SIZE,
    overlap_ratio: OVERLAP_RATIO,
    top_k: TOP_K,
  });
}