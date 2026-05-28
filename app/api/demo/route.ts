import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import type { ReviewScore, DemoApiResponse } from '@/lib/types';

export async function GET(): Promise<NextResponse> {
  let reviews: ReviewScore[];
  try {
    const filePath = path.join(process.cwd(), 'public', 'demo-data.json');
    const raw = fs.readFileSync(filePath, 'utf8');
    reviews = JSON.parse(raw) as ReviewScore[];
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
