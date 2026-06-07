# Session Handoff — Day 1 Complete

**Date:** 2026-05-28  
**Branch:** `main` (Day 1 merged) + `fix/vercel-framework` (open, not merged)  
**Status:** Day 1 DONE. Vercel deploy attempted, failed — fix in progress.

---

## What Was Accomplished

### Day 1 — Full Next.js scaffold (merged to main, 11 commits)

| File | Purpose | Status |
|------|---------|--------|
| `next.config.mjs` | Build-time demo-data.json validation (≥100 rows, 10 fields) | ✅ |
| `lib/types.ts` | ReviewScore, VaderScore, RobertaScore, DemoApiResponse, AnalyzeApiResponse | ✅ |
| `app/api/demo/route.ts` | GET /api/demo → { reviews, count, asin: null }, cached at module scope | ✅ |
| `components/SentimentPlot.tsx` | Plotly scatter, dynamic+ssr:false, X=vader.compound, Y=roberta.positive, color=disagreement | ✅ |
| `components/DisagreementPanel.tsx` | Top 10 reviews sorted by disagreement desc, shadcn Card list | ✅ |
| `app/page.tsx` | Fetches /api/demo, renders plot + panel, sessionStorage warmup guard | ✅ |
| `app/api/warmup/route.ts` | HF_API_KEY check→503, KV rate limit 1/min per IP, POST HF 15s timeout | ✅ |
| `spikes/pipe1-hf.js` | HF API validator (labels + batch test) | ✅ |
| `spikes/pipe2-canopy.js` | Canopy API validator (reviews shape, field names) | ✅ |
| shadcn/ui | button, input, card components | ✅ |

### Env setup
- Vercel project created: `hussain-coder-engs-projects/sentiment-amazon-analyzer`
- Upstash KV connected, env vars pulled to `.env.development.local`
- `HF_API_KEY` and `CANOPY_API_KEY` added to `.env.development.local`
- `.env.development.local` has spaces before `=` signs (e.g. `HF_API_KEY =xxx`) — note when sourcing

---

## Issues Faced

### Issue 1: Network blocks both external APIs locally
- `api-inference.huggingface.co` → `ENOTFOUND`
- `api.canopyapi.co` → `ENOTFOUND`
- Same network block that hit Step 0 (fixed then by using local transformers)
- **Spikes cannot be run locally.** Must run on Vercel's network.

### Issue 2: Vercel deploy detected project as Python
- Error: `"No python entrypoint found"`
- Root cause: `requirements.txt` in repo root confuses Vercel auto-detection
- **Fix:** Add `vercel.json` to force Next.js framework
- Branch `fix/vercel-framework` created but `vercel.json` not yet written/committed
- **Next session must complete this fix before deploying**

---

## What's Next

### Immediate (fix/vercel-framework branch)

1. Create `vercel.json` in repo root:
   ```json
   {
     "framework": "nextjs",
     "buildCommand": "npm run build",
     "outputDirectory": ".next",
     "installCommand": "npm install"
   }
   ```
2. Commit + merge to main
3. Run `vercel --prod` → should deploy successfully
4. Verify live URL: hit `/api/demo`, confirm JSON returns 500 reviews

### Spike Gate (run on Vercel network — REQUIRED before Day 2)

After deploy, test from Vercel's network:
- HF spike: does `cardiffnlp/twitter-roberta-base-sentiment` return `negative/neutral/positive` labels (not `LABEL_0/1/2`)?
- Canopy spike: does `reviewsPaginated.reviews` return ≥10 reviews? Is `body` field non-empty?
- Both pass → Day 2 unblocked

Note: HANDOFF.md from Step 0 confirmed that `cardiffnlp/twitter-roberta-base-sentiment-latest` returns named labels. The design spec uses the base model (without `-latest`). Verify which model URL returns named labels on Vercel's network — may need to switch to `-latest` in `/api/warmup` and `/api/analyze`.

### Day 2 (after spike gate passes)

Per design spec build order:
5. `/api/analyze` skeleton: ASIN validation + KV fail-closed + cache check + rate limit + inflight lock + cache re-check + monthly circuit breaker
6. Wire Canopy fetch + dedup + VADER + HF sequential scoring + cache result
7. ASIN input form → live scatter plot + disagreement panel

**Environment vars still needed for Day 2:**
```
CANOPY_API_KEY=xxx    # already in .env.development.local (but API blocked locally)
HF_API_KEY=hf_xxx    # already in .env.development.local (but API blocked locally)
KV_REST_API_URL=xxx  # already set via Vercel KV
KV_REST_API_TOKEN=xxx # already set via Vercel KV
```

**Key files:**
- Design spec: `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md`
- Day 2 build order: design spec § "Build Order" → Day 2 steps 5-7
- KV client pattern: `new Redis({ url: process.env.KV_REST_API_URL!, token: process.env.KV_REST_API_TOKEN! })` (NOT `Redis.fromEnv()`)
