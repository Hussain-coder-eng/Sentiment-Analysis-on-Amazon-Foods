# Session Handoff — Day 2 Complete

**Date:** 2026-06-09  
**Branch:** `main` (all Day 2 work merged)  
**Status:** `/api/analyze` + frontend wired. Live at https://sentiment-amazon-analyzer.vercel.app

---

## What Was Accomplished (Day 1 — spike gate)

### Fix: Vercel framework detection
- Added `vercel.json` forcing Next.js — `requirements.txt` in root was causing Python mis-detection
- Deployed successfully: https://sentiment-amazon-analyzer.vercel.app
- `/api/demo` verified: returns 500 reviews ✅

### Spike Gate — API endpoint investigation (all findings on Vercel network)

Both original API endpoints are **dead/NXDOMAIN**. New endpoints confirmed working:

| API | Old (dead) | New (working) | Notes |
|-----|-----------|---------------|-------|
| HuggingFace | `api-inference.huggingface.co` | `router.huggingface.co/hf-inference/models/…` | Deprecated late 2025 |
| Canopy REST | `api.canopyapi.co/v1/…` | `rest.canopyapi.co/api/amazon/product/reviews` | REST blocked from Vercel IPs (Cloudflare WAF) |
| Canopy GraphQL | — | `graphql.canopyapi.co/` | **USE THIS** — works from Vercel ✅ |

**HF model finding (critical):**
- Base model `cardiffnlp/twitter-roberta-base-sentiment` → returns `LABEL_0/1/2` (unusable)
- `-latest` model `cardiffnlp/twitter-roberta-base-sentiment-latest` → returns `positive/neutral/negative` ✅
- **Must use `-latest` everywhere**

**Canopy GraphQL response shape:**
```
POST https://graphql.canopyapi.co/
Header: API-KEY: <key>
Body: { "query": "{ amazonProduct(input:{asin:\"ASIN\",domain:US}){ topReviews { id body rating verifiedPurchase } } }" }

Response: data.data.amazonProduct.topReviews[]  (8 reviews max)
Fields: id, title, body, imageUrls, videos, rating, helpfulVotes, verifiedPurchase, reviewer
```

Note: paginated endpoint gone — only `topReviews` (8 per ASIN) available.

### Env vars — current Vercel state
```
HF_API_KEY          = hf_q...   (Production)
CANOPY_API_KEY      = 2fc6...   (Production)
KV_REST_API_URL     = set       (Production, Preview)
KV_REST_API_TOKEN   = set       (Production, Preview)
KV_REST_API_READ_ONLY_TOKEN = set
KV_URL / REDIS_URL  = set
```

Local: `.env.development.local` has spaces before `=` signs — be careful when sourcing.  
Refresh local env: `vercel env pull .env.development.local`

---

## What Was Accomplished (Day 2)

### `app/api/analyze/route.ts` — merged to main

GET endpoint implementing the full 15-step flow:

1. Env checks → 503 (all 4 vars required before any logic)
2. KV init fail-closed → 503
3. ASIN validation regex `/^[A-Z0-9]{10}$/` → 400
4. Cache hit check (`asin:v1:<ASIN>:scored`) → return immediately, bypasses rate limit
5. Failure cache check (`asin:v1:<ASIN>:failed`) → return cached error
6. Per-IP rate limit: `rate:<IP>:<hourBucket>`, >5/hour → 429
7. Inflight lock: `kv.set(lockKey, '1', { nx:true, ex:90 })` → 202 if locked
8. Cache re-check inside lock
9. Monthly circuit breaker: `quota:canopy:YYYY-MM`, >90 → 503
10. Canopy GraphQL fetch (20s timeout) + GraphQL errors guard → `canopy_graphql_error`
11. Normalize + dedup (sanitize-html, sha1 dedupKey), count gate <5 → 422
12. VADER sync scoring (vader-sentiment npm)
13. HF sequential scoring (8s timeout per call, 100ms gap, `-latest` model)
14. Disagreement = `|vader.compound - (roberta.positive - roberta.negative)|`
15. Best-effort success cache (24h), return `{ reviews, count, asin }`

`maxDuration = 120` — worst-case budget: 84.7s (20s Canopy + 8×8s HF + 0.7s gaps).

Lock released in `finally` block. Success cache write is outside inner try/catch (best-effort).

**Key API patterns:**
```typescript
// Canopy GraphQL
const canopyRes = await fetch('https://graphql.canopyapi.co/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'API-KEY': process.env.CANOPY_API_KEY! },
  body: JSON.stringify({ query: `{ amazonProduct(input:{asin:"${normalizedAsin}",domain:US}){ topReviews { id body rating verifiedPurchase } } }` }),
  signal: AbortSignal.timeout(20_000),
});
if (responseJson?.errors) throw new Error('canopy_graphql_error');
const raw = responseJson?.data?.amazonProduct?.topReviews ?? [];

// HF (sequential, single string per call, -latest required)
const hfRes = await fetch(
  'https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest',
  { method: 'POST', headers: { Authorization: `Bearer ${process.env.HF_API_KEY!}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: reviewText }), signal: AbortSignal.timeout(8_000) }
);
// HF defensive unwrap (handles nested [[{label,score}]] and flat [{label,score}])
const rawHfResponse: unknown = await hfRes.json();
const labels = Array.isArray(rawHfResponse) && Array.isArray((rawHfResponse as unknown[])[0])
  ? ((rawHfResponse as unknown[][])[0] as { label: string; score: number }[])
  : (rawHfResponse as { label: string; score: number }[]);

// KV client
new Redis({ url: process.env.KV_REST_API_URL!, token: process.env.KV_REST_API_TOKEN! })
```

### `app/page.tsx` — merged to main

ASIN input form + live scoring:
- Controlled text input, client-side regex `/^[A-Z0-9]{10}$/` + auto-uppercase
- GET `/api/analyze?asin=<ASIN>` on submit, form disabled while analyzing
- Loading: "Waking up the ML model — this takes ~10-30s on first use."
- Inline error display, "Reset to demo" button on live/error state
- Charts hidden during analysis, shown for both demo and live results
- Warmup fire-and-forget preserved on mount

---

## What's Next (Day 3)

### Completed this session
- [x] `/api/analyze` 15-step ML pipeline — merged ✅
- [x] Frontend ASIN form wired — merged ✅
- [x] Deploy confirmed Ready (1m after push) ✅
- [x] Cache seed: fired 5 popular food ASINs (B001E4KFG0, B00813GRG4, B000LKTTTQ, B006K2ZZ7K, B001EO5Q64) against production

### Remaining Day 3
- [ ] Verify seed results (check /tmp/seed_*.json to see which ASINs scored successfully)
- [ ] Error state polish: 202 "in progress" UX (currently shows raw message — could poll/retry)
- [ ] Post-scoring validation hardening (minor deferred items from code review)
- [ ] Consider adding `<label>` to ASIN input for accessibility (WCAG 2.1)
- [ ] Deferred cleanup: remove `verifiedPurchase` from GraphQL query, remove dead `scoring_length_mismatch` TTL case

### Known deferred issues (non-blocking)
- GraphQL 200-with-errors now correctly throws `canopy_graphql_error` (120s TTL) — fixed ✅
- `scoring_length_mismatch` in `getFailureTtl` is now unreachable dead code
- `verifiedPurchase` queried from Canopy but unused
- Duplicate `/api/demo` fetch logic in `useEffect` and `handleResetDemo`
- Rate limit counter increments on 202 (lock-miss) responses; 202 message says "30s" but lock TTL is 90s
- No `<label>` on ASIN input (accessibility gap)

### Design spec reference
Full spec: `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md`

Key types: `lib/types.ts` — `ReviewScore`, `AnalyzeApiResponse`  
Reference implementations: `app/api/demo/route.ts`, `app/api/warmup/route.ts`
