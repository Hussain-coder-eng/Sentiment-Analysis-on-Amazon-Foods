# Session Handoff — Day 3 Complete

**Date:** 2026-06-09  
**Branch:** `main` (all work merged)  
**Live:** https://sentiment-amazon-analyzer.vercel.app  
**Status:** Feature complete. Pre-seeding in progress (rate-limited — resume next session).

---

## Production State

| Env var | Status |
|---------|--------|
| HF_API_KEY | ✅ Production |
| CANOPY_API_KEY | ✅ Production (re-added 2026-06-09 — was missing) |
| KV_REST_API_URL | ✅ Production + Preview |
| KV_REST_API_TOKEN | ✅ Production + Preview |
| KV_REST_API_READ_ONLY_TOKEN | ✅ Production + Preview |

Local env file note: `.env.development.local` has spaces before `=` signs. Refresh with `vercel env pull .env.development.local`.

---

## Confirmed Working ASINs (Canopy-indexed, 5+ reviews)

| ASIN | Reviews | Status |
|------|---------|--------|
| B000E7L2R4 | 8 | ✅ Cached in KV (scored 2026-06-09) |

Other tested ASINs returned PRODUCT_NOT_FOUND or < 5 reviews from Canopy. Need to find more valid ASINs for seeding (see Day 4 below).

---

## What Was Built

### `app/api/analyze/route.ts`
GET endpoint — 15-step ML pipeline:

1. Env checks → 503
2. KV init fail-closed → 503
3. ASIN validation `/^[A-Z0-9]{10}$/` → 400
4. Success cache hit (`asin:v1:<ASIN>:scored`) → return (bypasses rate limit)
5. Failure cache hit (`asin:v1:<ASIN>:failed`) → return cached error
6. Per-IP rate limit: 5/hour via `rate:<IP>:<hourBucket>` → 429
7. Inflight lock: atomic NX SET (90s TTL) → 202 if locked
8. Cache re-check inside lock
9. Monthly circuit breaker: `quota:canopy:YYYY-MM` > 90 → 503
10. Canopy GraphQL fetch (20s timeout) + `responseJson?.errors` guard
11. Normalize + dedup + count gate < 5 → 422
12. VADER sync scoring
13. HF sequential scoring (8s/call, 100ms gap, `-latest` model)
14. Disagreement = `|vader.compound - (roberta.positive - roberta.negative)|`
15. Best-effort success cache (24h) + return `{ reviews, count, asin }`

`maxDuration = 120` (budget: 84.7s worst-case).

### `app/page.tsx`
- ASIN input form with client-side validation + auto-uppercase
- `loadDemo()` extracted — shared by mount + reset
- `doAnalyze()` recursive with 202 auto-retry: up to 3 attempts, 30s countdown between
- Loading: "Waking up the ML model — this takes ~10-30s on first use."
- Countdown: "Another analysis in progress — retrying in Ns…"
- Reset to demo clears all state + countdown timers
- Charts hidden during analysis; shown for demo and live results

---

## API Patterns (copy-paste ready)

```typescript
// Canopy GraphQL
const canopyRes = await fetch('https://graphql.canopyapi.co/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'API-KEY': process.env.CANOPY_API_KEY! },
  body: JSON.stringify({ query: `{ amazonProduct(input:{asin:"${asin}",domain:US}){ topReviews { id body rating } } }` }),
  signal: AbortSignal.timeout(20_000),
});
if (responseJson?.errors) throw new Error('canopy_graphql_error');

// HF (sequential, -latest required, defensive unwrap)
const rawHfResponse: unknown = await hfRes.json();
const labels: { label: string; score: number }[] =
  Array.isArray(rawHfResponse) && Array.isArray((rawHfResponse as unknown[])[0])
    ? ((rawHfResponse as unknown[][])[0] as { label: string; score: number }[])
    : (rawHfResponse as { label: string; score: number }[]);

// KV client
new Redis({ url: process.env.KV_REST_API_URL!, token: process.env.KV_REST_API_TOKEN! })
```

---

## Day 4 (next session)

1. **Find more valid ASINs for seeding** — B000E7L2R4 is the only confirmed one. Try testing new ASINs via the live UI. Rate limit resets hourly; 5 Canopy calls/hour available.
2. **Seed 4+ more ASINs** — use the production UI or run curl against `/api/analyze` with valid ASINs
3. Minor: 202 error message after max retries could be friendlier ("Try again in a few minutes")
4. Minor: No loading indicator during demo reset (stale data shows briefly — acceptable)
