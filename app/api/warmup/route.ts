import { NextRequest, NextResponse } from 'next/server';
import { Redis } from '@upstash/redis';

export async function GET(request: NextRequest): Promise<NextResponse> {
  const hfApiKey = process.env.HF_API_KEY;
  if (!hfApiKey) {
    return NextResponse.json(
      { ok: false, error: 'Service misconfigured' },
      { status: 503 },
    );
  }

  let kv: Redis;
  try {
    kv = new Redis({
      url: process.env.KV_REST_API_URL!,
      token: process.env.KV_REST_API_TOKEN!,
    });
  } catch {
    return NextResponse.json({ ok: false }, { status: 503 });
  }

  const realIp = request.headers.get('x-real-ip');
  const forwardedFor = request.headers.get('x-forwarded-for');
  const ip = realIp ?? forwardedFor?.split(',')[0]?.trim() ?? 'unknown';
  const minBucket = Math.floor(Date.now() / 60_000);

  let count: number;
  try {
    count = await kv.incr(`warmup:${ip}:${minBucket}`);
    if (count === 1) {
      await kv.expire(`warmup:${ip}:${minBucket}`, 120);
    }
  } catch {
    return NextResponse.json({ ok: false }, { status: 503 });
  }

  if (count > 1) {
    return NextResponse.json({ ok: false }, { status: 429 });
  }

  let hfRes: Response;
  try {
    hfRes = await fetch(
      'https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest',
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${hfApiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ inputs: 'warmup' }),
        signal: AbortSignal.timeout(15000),
      },
    );
  } catch {
    return NextResponse.json(
      { ok: false, error: 'hf_timeout' },
      { status: 503 },
    );
  }

  if (!hfRes.ok) {
    return NextResponse.json(
      { ok: false, error: `hf_${hfRes.status}` },
      { status: 503 },
    );
  }

  return NextResponse.json({ ok: true });
}
