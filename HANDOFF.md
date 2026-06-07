# Session Handoff — Spike Gate Complete

**Date:** 2026-06-07  
**Branch:** `main` (all work merged)  
**Status:** Vercel deployed ✅. Spike gate passed ✅. Day 2 unblocked.

---

## What Was Accomplished (this session)

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

### Files updated
| File | Change |
|------|--------|
| `vercel.json` | Added (forces Next.js framework) |
| `app/api/warmup/route.ts` | HF URL → new router, model → `-latest` |
| `spikes/pipe1-hf.js` | Updated URL + model |
| `spikes/pipe2-canopy.js` | Updated to `rest.canopyapi.co` + `topReviews` shape |

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

## What's Next — Day 2

### Step 5: `/api/analyze` skeleton

Per design spec (`docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md` § "Build Order"):

```
POST /api/analyze { asin: string }

Flow:
1. Validate ASIN (regex: /^[A-Z0-9]{10}$/)
2. KV fail-closed check (if KV down → 503)
3. Cache check → return cached if hit
4. Rate limit per IP (1 req/min, KV bucket)
5. Inflight lock (atomic NX SET 90s TTL — prevent duplicate Canopy calls)
6. Cache re-check after acquiring lock
7. Monthly circuit breaker (KV counter, cap spend)
8. Canopy GraphQL fetch → extract topReviews (8 reviews)
9. Dedup by SHA-1(body) fallback if id missing
10. VADER scoring per review
11. HF batch scoring via router.huggingface.co (-latest model)
12. Compute disagreement = |vader_compound - (roberta_positive - roberta_negative)|
13. Cache result with TTL
14. Return { reviews: ReviewScore[], count, asin }
```

**Canopy call for analyze route:**
```typescript
const res = await fetch('https://graphql.canopyapi.co/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'API-KEY': process.env.CANOPY_API_KEY! },
  body: JSON.stringify({ query: `{ amazonProduct(input:{asin:"${asin}",domain:US}){ topReviews { id body rating verifiedPurchase } } }` }),
  signal: AbortSignal.timeout(20_000),
});
const data = await res.json();
const reviews = data?.data?.amazonProduct?.topReviews ?? [];
```

**HF call for analyze route:**
```typescript
const hfRes = await fetch(
  'https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest',
  {
    method: 'POST',
    headers: { Authorization: `Bearer ${process.env.HF_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: reviews.map(r => r.body) }),
    signal: AbortSignal.timeout(60_000),
  }
);
// Response: [[{label, score}, ...], ...]  — outer array = per review, inner = per label
```

**KV client pattern (always use this, NOT Redis.fromEnv()):**
```typescript
new Redis({ url: process.env.KV_REST_API_URL!, token: process.env.KV_REST_API_TOKEN! })
```

### Step 6: Wire frontend
- ASIN input form on `app/page.tsx`
- POST to `/api/analyze` on submit
- Render live scatter plot + disagreement panel with result

### Branch naming
- `feat/day2-analyze-api` for steps 5–6
- Review + merge when done

---

## Key Files Reference
- Design spec: `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md`
- Types: `lib/types.ts` (ReviewScore, VaderScore, RobertaScore, DemoApiResponse, AnalyzeApiResponse)
- Demo route (reference for cache pattern): `app/api/demo/route.ts`
- Warmup route (reference for KV + HF pattern): `app/api/warmup/route.ts`
