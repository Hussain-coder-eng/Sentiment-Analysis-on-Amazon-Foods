# Portfolio Upgrade — Design Spec

**Date:** 2026-06-10
**Status:** Approved pending user review
**Live app:** https://sentiment-amazon-analyzer.vercel.app

## Goal

Turn the working two-model sentiment demo into a standout ML-internship portfolio piece:

1. **More insightful** — add aspect-based sentiment (third ML technique) and a synthesized verdict.
2. **Useful to the average user** — verdict-first results a non-technical shopper understands; one-click example gallery.
3. **Memorable** — scroll-driven 3D Amazon-box hero (dstafin-style "one object travels the page").
4. **Recruiter-legible** — how-it-works narrative, shareable result URLs, README case study.

Already shipped, out of scope: URL→ASIN auto-extraction (Day 5, `app/page.tsx`).

## Decisions Made (with user)

| Decision | Choice |
|---|---|
| ML depth | Aspect-based sentiment via HF zero-shot NLI (`facebook/bart-large-mnli`) |
| Results voice | Verdict-first, technical detail below |
| Results layout | Editorial scroll (B) — full-width narrative sections |
| Packaging | Example gallery + how-it-works + README case study + `/p/[asin]` (all four) |
| Hero | Scroll-driven 3D CSS cardboard box; flaps open mid-scroll; **one** product card + sentiment chips emerge; box docks beside form |
| Gallery content | Popular Amazon buys (not food-only), pending Canopy seeding |

## Phases

Each phase = one feature branch, reviewed and shipped independently.

### Phase 1 — Verdict layer + editorial results redesign

**`lib/verdict.ts` (new, pure functions, unit-tested):**

- Per review polarity: `roberta.positive - roberta.negative` ∈ [−1, 1]
- Product score: `round((mean(polarity) + 1) * 50)` → 0–100
- Label bands: ≥75 "Loved it" · 60–74 "Mostly positive" · 45–59 "Mixed" · <45 "Mostly negative"
- Agreement %: share of reviews where VADER compound and RoBERTa polarity have the same sign
- One-liner: `"{positiveCount} of {total} reviews positive · models agree {agreement}%"`

**Results panel (`app/page.tsx` + new `components/VerdictCard.tsx`):**

Editorial scroll order:
1. Verdict hero card — score ring, label, one-liner, product title + ASIN
2. (Phase 2 slot: aspect bars)
3. Model deep-dive — existing `SentimentPlot` + `DisagreementPanel`, restyled section headers
4. How-it-works strip — 4 pipeline step cards (Canopy → VADER+RoBERTa → aspects → verdict)

**Example gallery (`lib/gallery.ts` + landing form section):**

- Hardcoded const: `{ asin, shortName, emoji, imageUrl? }[]` for known-cached ASINs
- Chips under the form ("— or try one —"); click → sets input + submits → instant KV hit
- Caption: "cached — instant results"
- Ships with current 5 food ASINs; swaps to popular products as seeding lands (see Seeding)

**Testing:** vitest (new dev dep, minimal config) for `lib/verdict.ts`. UI verified manually.

### Phase 2 — Aspect-based sentiment + cache v2

**Pipeline (in `app/api/analyze/route.ts`, after HF scoring):**

1. Per review, one zero-shot call: HF router, `facebook/bart-large-mnli`, `multi_label: true`, 5s timeout, sequential with the existing 100ms gap
2. Candidate labels (universal — threshold filters irrelevant ones per product):
   `["taste & flavor", "quality", "value for money", "packaging & shipping", "ease of use"]`
3. Aspect "mentioned" in a review if NLI score ≥ 0.7
4. Aspect polarity = mean RoBERTa polarity of reviews mentioning it; aspect included in response if ≥ 2 reviews mention it
5. **Fail-soft:** any zero-shot failure → `aspects` omitted entirely, response still 200. Aspects never block.

Known approximation (documented in README): review-level sentiment attributed to aspects, not sentence-level. Honest at 8-review scale.

**Budget:** worst case +8 calls × 5s = +40s. `maxDuration` 120 → 180.

**Cache envelope v2:**

- Key: `asin:v2:<ASIN>:scored` (failure key stays v1 — shape unchanged)
- Value: `{ reviews: ReviewScore[], productTitle?: string, aspects?: AspectScore[], analyzedAt: string }`
- Read v2 only. No migration: v1 keys expire via existing 24h TTL.
- Fixes Day 6 backlog item: cache hits now return `productTitle`.

**Types (`lib/types.ts`):** add `AspectScore = { label: string; polarity: number; mentions: number }`; extend `AnalyzeApiResponse` with `aspects?`.

**UI:** "What reviewers say" section between verdict and deep-dive — horizontal polarity bars (green positive / red negative), mention counts, model attribution footnote. Hidden when `aspects` absent.

**Testing:** vitest for aspect aggregation (pure function extracted to `lib/aspects.ts`); zero-shot call mocked.

### Phase 3 — Scroll-driven 3D box hero

**Object:** pure CSS 3D cuboid (6 faces, cardboard palette, tape stripe, green smile-arrow decal). No three.js.

**Choreography (scroll progress → transforms via rAF scroll handler; anime.js for idle loop + open burst):**

| Scroll | Box | Page |
|---|---|---|
| 0% (hero viewport) | Large, idle float + slow rotateY | Headline + tagline beside box |
| ~30% | Tumbles (rotateX/Y mapped to progress), shrinks, travels down-right | Headline parallaxes up faster |
| ~60% | Flaps open; **one product card** (first item in `lib/gallery.ts`, real image) rises out + 2–3 sentiment chips spill | How-it-works copy (3 pipeline steps) alongside |
| 100% | Docks small beside form as accent | Analyze form + gallery chips; results render below after analyze |

- Emerged product card is clickable → analyzes that ASIN
- `prefers-reduced-motion`: static box illustration, no scroll choreography, all content visible
- Mobile: reduced transform ranges, same content order
- How-it-works strip from Phase 1 results panel stays (results context); scroll-story version covers landing

### Phase 4 — Shareable routes + README case study

**`/p/[asin]`:** dynamic route rendering the same Home UI with ASIN prefilled + auto-analyze on mount. "Share" button on results copies URL. Uncached shared ASINs burn the visitor's rate limit — accepted, documented.

**README rewrite (case study):**

1. What/why (one paragraph) + live demo link + gallery GIF
2. Architecture diagram (mermaid): browser → Next.js API → Canopy / HF / KV
3. ML methodology: VADER, RoBERTa, zero-shot NLI aspects, verdict math
4. Engineering decisions: caching (v2 envelope), per-IP rate limit, inflight lock, circuit breaker, fail-soft aspects
5. Limitations: 8 reviews/product, review-level aspect attribution, Canopy index coverage
6. Local setup

## ASIN Seeding (ongoing, manual, rate-limited)

- Candidates (popular buys): Echo Dot, Fire TV Stick, Stanley 40oz tumbler, AirPods/earbuds, Kindle, Crocs, LEGO set, Squishmallow, phone charger, LED strip
- Constraint: 5 Canopy calls/hour, 90/month circuit breaker; many ASINs not Canopy-indexed (6 already blacklisted in HANDOFF)
- Process per session: try ≤5 candidates → cached successes get `imageUrl` (Canopy `mainImageUrl`) recorded into `lib/gallery.ts`
- Gallery always renders only confirmed-cached ASINs

## Error Handling Summary

- Aspects: fail-soft, omit on any error (UI hides section)
- Gallery click on expired cache: normal analyze flow handles (re-scores, 10–30s message)
- `/p/[asin]` invalid ASIN: existing validation error path
- Hero animations: reduced-motion fallback; JS failure leaves static content visible (CSS default states, same pattern as current `.animate-in` fallback)

## Out of Scope

- Sentence-level aspect attribution
- LLM-generated summaries
- three.js / photoreal 3D
- Auth, persistence beyond KV cache, comparison view

## Open Items

- `.superpowers/` added to `.gitignore` (Phase 1 chore)
- 5 npm vulnerabilities flagged 2026-06-09 — separate audit task, not this spec
