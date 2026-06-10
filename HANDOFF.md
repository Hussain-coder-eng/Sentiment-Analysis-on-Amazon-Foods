# Session Handoff — Day 4 Complete

**Date:** 2026-06-09  
**Branch:** `main` (all work merged)  
**Live:** https://sentiment-amazon-analyzer.vercel.app  
**Status:** Hero landing shipped. Demo mode fully removed. Pre-seeding ongoing (rate-limited).

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
| B00032G1S0 | 5 | ✅ Cached in KV (scored 2026-06-09) |

Tested and invalid (PRODUCT_NOT_FOUND in Canopy): B001E4KFG0, B07XTPQNLZ, B001FA1JWK, B00LMTMZQO, B01N9SPQHD, B00CLYA5EQ. Don't retry these.

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

### `app/page.tsx` (Day 4 rewrite — hero landing)
- Demo mode fully removed (`loadDemo`, `demoLoading`, `demoError`, `isLive` gone)
- Hero heading + subtitle; no chart on landing — results appear only after successful analyze
- Dark cinema theme: `#0F172A` bg, `#22C55E` accent, Exo heading + Roboto Mono body
- anime.js v3 entrance animations (hero stagger, form slide, results fade-in); `prefers-reduced-motion` CSS fallback ensures elements visible without JS
- `doAnalyze()` recursive with 202 auto-retry: up to 3 attempts, 30s countdown between
- After max 202 retries: sets inline error ("Analysis still in progress — try again") instead of crashing
- "Clear" button replaces "Reset to demo" — clears `reviews`, `resultAsin`, countdown
- Loading: "Waking up the ML model — this takes ~10–30s on first use."

### `app/layout.tsx`
- Google Fonts: Exo (headings) + Roboto Mono (body), `next/font/google`, `display: swap`
- `dark` class on `<html>` — activates shadcn dark CSS vars
- Metadata: "Amazon Review Sentiment Analyzer"

### `app/globals.css`
- Full dark palette: `--background: #0F172A`, `--primary: #22C55E`, `--card: #1E293B`, etc.
- `--chart-*` vars updated to colored palette (green/blue/amber/pink/violet) for chart legibility
- `.ambient-blob`, `.glow-green`, `.glow-green-text` utilities
- `@media (prefers-reduced-motion)` fallback: `.animate-in`, form, `.results-panel` → `opacity: 1`

### `components/DisagreementPanel.tsx`
- `text-gray-700` → `text-card-foreground`, `text-gray-500` → `text-muted-foreground` (dark theme fix)

### `components/SentimentPlot.tsx`
- Plotly layout: `paper_bgcolor: '#0F172A'`, `plot_bgcolor: '#1E293B'`, `font.color: '#F8FAFC'`, axis `gridcolor: '#334155'`
- Loading state: `text-muted-foreground` (dark theme fix)

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

## Day 5 (next session)

1. **Find + seed more valid ASINs** — only 2 confirmed so far (B000E7L2R4, B00032G1S0). Test new ASINs via live UI. Rate limit: 5 Canopy calls/hour; resets on the hour.
2. **Verify hero landing on production** — open https://sentiment-amazon-analyzer.vercel.app, confirm dark UI loads, enter B000E7L2R4, check results render. Vercel should have auto-deployed from the main push.
3. **Optional UX** — product name display (fetch title from Canopy alongside reviews); Amazon URL → ASIN extraction in the input field
