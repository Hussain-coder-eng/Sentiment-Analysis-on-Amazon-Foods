import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import type { ReviewScore, DemoApiResponse } from '@/lib/types';

// Cached at module scope -- read once per cold start, not per request
let cachedReviews: ReviewScore[] | null = null;

function loadReviews(): ReviewScore[] {
  if (cachedReviews) return cachedReviews;
  const filePath = path.join(process.cwd(), 'public', 'demo-data.json');
  const raw = fs.readFileSync(filePath, 'utf8');
  const parsed: unknown = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error('demo-data.json is not an array');
  }
  cachedReviews = parsed as ReviewScore[];
  return cachedReviews;
}

export async function GET(): Promise<NextResponse> {
  let reviews: ReviewScore[];
  try {
    reviews = loadReviews();
  } catch {
    return NextResponse.json({ error: 'Demo data not found' }, { status: 500 });
  }

  const response: DemoApiResponse = {
    reviews,
    count: reviews.length,
    asin: null,
  };
  return NextResponse.json(response);
}
