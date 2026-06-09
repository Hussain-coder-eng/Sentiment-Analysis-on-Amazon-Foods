# Session Handoff — Day 2 In Progress

**Date:** 2026-06-08  
**Branch:** `feat/day2-analyze-api` (2 commits ahead of main, NOT merged)  
**Status:** `/api/analyze` implemented + spec-compliant. Code quality review open — 1 CRITICAL + 1 IMPORTANT unresolved.

---

## What Was Accomplished (previous session — spike gate)

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
- **Must use `-latest` everywhere in the codebase**

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
HF_API_KEY          = hf_q...   (Production) ← new valid token added this session
CANOPY_API_KEY      = 2fc6...   (Production) ← was already set
KV_REST_API_URL     = set       (Production, Preview)
KV_REST_API_TOKEN   = set       (Production, Preview)
KV_REST_API_READ_ONLY_TOKEN = set
KV_URL / REDIS_URL  = set
```

Local: `.env.development.local` has spaces before `=` signs — be careful when sourcing.  
Refresh local env: `vercel env pull .env.development.local`

---

## What Was Accomplished (this session — Day 2 start)

### Branch created: `feat/day2-analyze-api`

### `app/api/analyze/route.ts` — implemented and spec-compliant

GET endpoint implementing the full 14-step flow (spec-validated):

1. Env checks → 503 (all 4 vars required before any logic)
2. KV init fail-closed → 503
3. ASIN validation regex `/^[A-Z0-9]{10}$/` → 400
4. Cache hit check (`asin:v1:<ASIN>:scored`) → return immediately, bypasses rate limit
5. Failure cache check (`asin:v1:<ASIN>:failed`) → return cached error
6. Per-IP rate limit: `rate:<IP>:<hourBucket>`, >5/hour → 429
7. Inflight lock: `kv.set(lockKey, '1', { nx:true, ex:90 })` → 202 if locked
8. Cache re-check inside lock
9. Monthly circuit breaker: `quota:canopy:YYYY-MM`, >90 → 503
10. Canopy GraphQL fetch → `data.data.amazonProduct.topReviews` (8 max, no pagination)
11. Normalize + dedup (sanitize-html, sha1 dedupKey), count gate <5 → 422
12. VADER sync scoring (vader-sentiment npm)
13. HF sequential scoring (one string per POST, 100ms gap)
14. Disagreement = `|vader.compound - (roberta.positive - roberta.negative)|`
15. Post-scoring validation, cache success 24h, return `{ reviews, count, asin }`

Lock released in `finally` block on all paths. `maxDuration = 60` set.

**Key API patterns (use these exactly):**

```typescript
// Canopy GraphQL
const canopyRes = await fetch('https://graphql.canopyapi.co/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'API-KEY': process.env.CANOPY_API_KEY! },
  body: JSON.stringify({ query: `{ amazonProduct(input:{asin:"${normalizedAsin}",domain:US}){ topReviews { id body rating verifiedPurchase } } }` }),
  signal: AbortSignal.timeout(20_000),
});
const raw = responseJson?.data?.amazonProduct?.topReviews ?? [];

// HF (sequential, single string per call)
const hfRes = await fetch(
  'https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest',
  { method: 'POST', headers: { Authorization: `Bearer ${process.env.HF_API_KEY!}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: reviewText }), signal: AbortSignal.timeout(8_000) }
);

// KV client (always this pattern)
new Redis({ url: process.env.KV_REST_API_URL!, token: process.env.KV_REST_API_TOKEN! })
```

### Code quality review open issues (must fix before merge)

**CRITICAL — HF response shape defensive unwrap missing:**
- HF may return `[[{label,score},…]]` (nested) instead of `[{label,score},…]` (flat)
- Current code assumes flat → all label lookups fail if nested → poisons failure cache 120s
- Fix: add defensive unwrap after `hfRes.json()`:
  ```typescript
  const rawHfResponse: unknown = await hfRes.json();
  const labels: { label: string; score: number }[] =
    Array.isArray(rawHfResponse) && Array.isArray((rawHfResponse as unknown[])[0])
      ? ((rawHfResponse as unknown[][])[0] as { label: string; score: number }[])
      : (rawHfResponse as { label: string; score: number }[]);
  if (!Array.isArray(labels)) throw new Error('hf_unexpected_shape');
  ```

**IMPORTANT — Success cache write failure poisons failure cache:**
- Step 12 `kv.set(scoredKey, ...)` is inside the inner try/catch
- If Redis write fails after scoring succeeds → inner catch writes 500 to `failKey` → caller gets 500 + retries poisoned for 120s
- Fix: move step 12 + step 14 return outside the inner try/catch; wrap step 12 in its own best-effort try/catch that logs and continues

**MINOR (can defer to Day 3):**
- Rate limit counter increments on 202 (lock-miss) responses — low impact
- `scoring_length_mismatch` error string reused for value-invalid case — use `scoring_value_invalid` instead
- `'unknown'` IP fallback collapses headerless clients into shared bucket — document only

---

## What's Next

### Immediate: fix CRITICAL + IMPORTANT before merge

```
1. Fix HF defensive unwrap (nested array)
2. Fix success cache write → move outside inner try/catch (best-effort)
3. Fix scoring_value_invalid error string (minor, do with the above)
4. tsc --noEmit clean
5. Re-run code quality review → APPROVE
6. Task 2: Wire ASIN input on app/page.tsx
7. Code review → merge feat/day2-analyze-api to main
```

### Task 2: Wire frontend (app/page.tsx)

- Add ASIN input form (text input + submit button, disable on submit)
- GET `/api/analyze?asin=<ASIN>` on submit
- Replace scatter plot + disagreement panel with live results
- Loading state: "Waking up the ML model — this takes ~10-30s on first use."
- Error display (user-friendly messages from API error responses)
- "Reset to demo" button → reloads demo data from `/api/demo`
- Keep warmup fire-and-forget on page load (already present)

### Design spec reference
Full spec: `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md`

Key types: `lib/types.ts` — `ReviewScore`, `AnalyzeApiResponse`  
Reference implementations: `app/api/demo/route.ts`, `app/api/warmup/route.ts`

### Day 3 (after merge)
- All error states + "Reset to demo" button polish
- Post-scoring validation hardening
- Pre-populate cache with 5-10 notable ASINs
- Deploy + get live URL
