import { NextRequest, NextResponse } from 'next/server';
import { Redis } from '@upstash/redis';
import { createHash } from 'node:crypto';
import sanitizeHtml from 'sanitize-html';
import type { ReviewScore, VaderScore, RobertaScore } from '@/lib/types';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const vader = require('vader-sentiment');

export const maxDuration = 60;

const HF_URL =
  'https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest';

interface Review {
  text: string;
  rating: number;
  dedupKey: string;
}

function getFailureTtl(errorMessage: string): number {
  if (errorMessage.includes('canopy_429')) return 300;
  if (errorMessage.includes('canopy_401') || errorMessage.includes('canopy_403')) return 1800;
  if (errorMessage.includes('canopy_5') || errorMessage.includes('timeout')) return 120;
  if (errorMessage.includes('hf_')) return 120;
  if (errorMessage.includes('scoring_length_mismatch')) return 60;
  return 120;
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  // Env checks at startup (all four required)
  if (!process.env.HF_API_KEY) {
    return NextResponse.json({ error: 'Service misconfigured' }, { status: 503 });
  }
  if (!process.env.CANOPY_API_KEY) {
    return NextResponse.json({ error: 'Service misconfigured' }, { status: 503 });
  }
  if (!process.env.KV_REST_API_URL || !process.env.KV_REST_API_TOKEN) {
    return NextResponse.json({ error: 'Service misconfigured' }, { status: 503 });
  }

  // Step 2: Init KV, fail-closed
  let kv: Redis;
  try {
    kv = new Redis({
      url: process.env.KV_REST_API_URL!,
      token: process.env.KV_REST_API_TOKEN!,
    });
  } catch {
    return NextResponse.json({ error: 'Service misconfigured' }, { status: 503 });
  }

  // Step 1: Validate ASIN
  const { searchParams } = new URL(request.url);
  const normalizedAsin = (searchParams.get('asin') ?? '').trim().toUpperCase();
  if (!/^[A-Z0-9]{10}$/.test(normalizedAsin)) {
    return NextResponse.json({ error: 'Invalid ASIN' }, { status: 400 });
  }

  const scoredKey = `asin:v1:${normalizedAsin}:scored`;
  const failKey = `asin:v1:${normalizedAsin}:failed`;
  const lockKey = `asin:v1:${normalizedAsin}:inflight`;

  // Step 3: Cache check (success) — no rate limit applied to cache hits
  const cachedScored = await kv.get<ReviewScore[]>(scoredKey);
  if (cachedScored) {
    return NextResponse.json({
      reviews: cachedScored,
      count: cachedScored.length,
      asin: normalizedAsin,
    });
  }

  // Step 4: Failure cache check
  const cachedFailure = await kv.get<{ status: number; message: string }>(failKey);
  if (cachedFailure) {
    return NextResponse.json(
      { error: cachedFailure.message },
      { status: cachedFailure.status },
    );
  }

  // Step 5: Per-IP rate limit (cache-miss path only)
  const realIp = request.headers.get('x-real-ip');
  const forwardedFor = request.headers.get('x-forwarded-for');
  const ip = realIp ?? forwardedFor?.split(',')[0]?.trim() ?? 'unknown';
  const hourBucket = Math.floor(Date.now() / 3_600_000);
  const rateKey = `rate:${ip}:${hourBucket}`;
  const rateCount = await kv.incr(rateKey);
  if (rateCount === 1) {
    await kv.expire(rateKey, 3600);
  }
  if (rateCount > 5) {
    return NextResponse.json({ error: 'Rate limit exceeded' }, { status: 429 });
  }

  // Step 6: Inflight lock (atomic)
  const locked = await kv.set(lockKey, '1', { nx: true, ex: 90 });
  if (!locked) {
    return NextResponse.json(
      { error: 'Analysis already in progress for this product. Retry in 30s.' },
      { status: 202 },
    );
  }

  try {
    // Step 7: Cache re-check inside lock
    const recheckScored = await kv.get<ReviewScore[]>(scoredKey);
    if (recheckScored) {
      return NextResponse.json({
        reviews: recheckScored,
        count: recheckScored.length,
        asin: normalizedAsin,
      });
    }

    // Step 8: Monthly circuit breaker
    const monthBucket = new Date().toISOString().slice(0, 7);
    const quotaKey = `quota:canopy:${monthBucket}`;
    const used = await kv.incr(quotaKey);
    if (used === 1) {
      await kv.expire(quotaKey, 60 * 60 * 24 * 32);
    }
    if (used > 90) {
      return NextResponse.json(
        { error: 'Monthly lookup quota reached. Demo mode available.' },
        { status: 503 },
      );
    }

    let reviews: Review[];

    try {
      // Step 9: Canopy GraphQL fetch
      const canopyRes = await fetch('https://graphql.canopyapi.co/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'API-KEY': process.env.CANOPY_API_KEY!,
        },
        body: JSON.stringify({
          query: `{ amazonProduct(input:{asin:"${normalizedAsin}",domain:US}){ topReviews { id body rating verifiedPurchase } } }`,
        }),
        signal: AbortSignal.timeout(20_000),
      });

      if (!canopyRes.ok) {
        throw new Error(`canopy_${canopyRes.status}`);
      }

      const responseJson = await canopyRes.json();
      const raw: { id?: string; body?: string; rating?: number | string }[] =
        responseJson?.data?.amazonProduct?.topReviews ?? [];

      const seen = new Set<string>();
      const normalized: Review[] = [];

      for (const r of raw) {
        const text = sanitizeHtml(r.body ?? '', {
          allowedTags: [],
          allowedAttributes: {},
        })
          .trim()
          .slice(0, 1000);
        const rating = Math.round(Number(r.rating));
        const id = String(r.id ?? '');
        const dedupKey =
          id !== ''
            ? id
            : createHash('sha1')
                .update(`${text}_${rating}`)
                .digest('hex')
                .slice(0, 16);

        if (text.length === 0 || rating < 1 || rating > 5) continue;
        if (seen.has(dedupKey)) continue;
        seen.add(dedupKey);
        normalized.push({ text, rating, dedupKey });
        if (normalized.length >= 8) break;
      }

      reviews = normalized;

      // Step 9.6: Count gate
      if (reviews.length < 5) {
        await kv.set(
          failKey,
          JSON.stringify({
            status: 422,
            message: 'Not enough reviews to analyze. Try a more popular product.',
          }),
          { ex: 3600 },
        );
        return NextResponse.json(
          { error: 'Not enough reviews to analyze. Try a more popular product.' },
          { status: 422 },
        );
      }

      // Step 10: VADER (sync, no await)
      const vaderScores: VaderScore[] = reviews.map((r) => {
        const raw = vader.SentimentIntensityAnalyzer.polarity_scores(r.text);
        if (
          !Number.isFinite(raw.compound) ||
          !Number.isFinite(raw.pos) ||
          !Number.isFinite(raw.neg) ||
          !Number.isFinite(raw.neu)
        ) {
          throw new Error('vader_invalid');
        }
        return {
          compound: raw.compound,
          pos: raw.pos,
          neg: raw.neg,
          neu: raw.neu,
        };
      });

      // Step 10: HF (sequential, one per review, 100ms gap)
      const robertaScores: RobertaScore[] = [];
      for (let i = 0; i < reviews.length; i++) {
        const hfRes = await fetch(HF_URL, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${process.env.HF_API_KEY!}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ inputs: reviews[i].text }),
          signal: AbortSignal.timeout(8_000),
        });
        if (!hfRes.ok) throw new Error(`hf_${hfRes.status}`);
        const labels: { label: string; score: number }[] = await hfRes.json();
        for (const l of ['negative', 'neutral', 'positive']) {
          const entry = labels.find((x) => x.label === l);
          if (!entry || !Number.isFinite(entry.score)) {
            throw new Error(`hf_label_missing_${l}`);
          }
        }
        const get = (l: string) => labels.find((x) => x.label === l)!.score;
        robertaScores.push({
          positive: get('positive'),
          neutral: get('neutral'),
          negative: get('negative'),
        });
        if (i < reviews.length - 1) {
          await new Promise<void>((resolve) => setTimeout(resolve, 100));
        }
      }

      // Step 10.5: Build ReviewScore[]
      const scored: ReviewScore[] = reviews.map((r, i) => {
        const v = vaderScores[i];
        const rob = robertaScores[i];
        const disagreement = Math.abs(v.compound - (rob.positive - rob.negative));
        return { text: r.text, rating: r.rating, vader: v, roberta: rob, disagreement };
      });

      // Step 11: Post-scoring validation
      if (scored.length < 5) throw new Error('scoring_length_mismatch');
      for (const s of scored) {
        if (
          !Number.isFinite(s.vader.compound) ||
          !Number.isFinite(s.roberta.positive) ||
          !Number.isFinite(s.disagreement)
        ) {
          throw new Error('scoring_length_mismatch');
        }
      }

      // Step 12: Cache success
      await kv.set(scoredKey, JSON.stringify(scored), { ex: 86400 });

      // Step 14: Return
      return NextResponse.json({
        reviews: scored,
        count: scored.length,
        asin: normalizedAsin,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const ttl = getFailureTtl(message);
      await kv.set(
        failKey,
        JSON.stringify({ status: 500, message: `Analysis failed: ${message}` }),
        { ex: ttl },
      );
      return NextResponse.json(
        { error: `Analysis failed: ${message}` },
        { status: 500 },
      );
    }
  } finally {
    // Step 13: Release inflight lock
    try {
      await kv.del(lockKey);
    } catch (e) {
      console.error('lock_release_failed', e);
    }
  }
}
