# Phase 2: Aspect-Based Sentiment + Cache Envelope v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-review zero-shot NLI aspect detection (5 universal labels) aggregated into product-level aspect polarity bars, plus a v2 cache envelope that finally returns `productTitle` (and now `aspects`) on cache hits.

**Architecture:** Pure aggregation math in `lib/aspects.ts` (TDD). Route gains a fail-soft zero-shot block after RoBERTa scoring — any aspect failure logs and omits `aspects`, never failing the request. Scored cache moves from bare `ReviewScore[]` under `asin:v1:` to an envelope under `asin:v2:` (old v1 keys expire naturally via 24h TTL — zero migration). New `AspectBars` component renders between VerdictCard and the model deep-dive.

**Tech Stack:** HF Inference router (`facebook/bart-large-mnli`, zero-shot, multi-label), Upstash Redis KV, vitest, Next.js 14 client page.

**Branch:** create `feat-aspects` from current `main`.

**Spec:** Phase 2 section of `docs/superpowers/specs/2026-06-10-portfolio-upgrade-design.md`

---

### Task 1: Types — `AspectScore`, response field, cache envelope

**Files:**
- Modify: `lib/types.ts`

- [ ] **Step 1: Create branch**

```bash
git checkout main && git pull && git checkout -b feat-aspects
```

- [ ] **Step 2: Append aspect types to `lib/types.ts` and extend the response**

Add after the `ReviewScore` interface:

```typescript
export interface AspectScore {
  label: string;     // one of the candidate aspect labels
  polarity: number;  // mean RoBERTa polarity of mentioning reviews, [-1, 1]
  mentions: number;  // count of reviews mentioning this aspect
}

/** Value stored at asin:v2:<ASIN>:scored (24h TTL). v1 keys held bare ReviewScore[]. */
export interface ScoredCacheV2 {
  reviews: ReviewScore[];
  productTitle?: string;
  aspects?: AspectScore[];
  analyzedAt: string; // ISO timestamp
}
```

And change `AnalyzeApiResponse` to:

```typescript
export interface AnalyzeApiResponse {
  reviews: ReviewScore[];
  count: number;
  asin: string;
  productTitle?: string;
  aspects?: AspectScore[];
}
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add lib/types.ts
git commit -m "feat: aspect score, cache envelope v2, and response types"
```

---

### Task 2: `lib/aspects.ts` — aggregation math (TDD)

**Files:**
- Create: `lib/aspects.test.ts`
- Create: `lib/aspects.ts`

Semantics (from spec):
- Candidate labels (universal; threshold filters irrelevant ones per product): `taste & flavor`, `quality`, `value for money`, `packaging & shipping`, `ease of use`
- A review "mentions" an aspect when its zero-shot score for that label ≥ 0.7
- Aspect included in output only when ≥ 2 reviews mention it
- Aspect polarity = mean RoBERTa polarity (`positive - negative`) of mentioning reviews
- Output sorted by mentions desc, ties by label asc
- Input arrays must align 1:1 with reviews; length mismatch throws (pipeline bug, fail loud)

- [ ] **Step 1: Write failing tests `lib/aspects.test.ts`**

```typescript
import { describe, it, expect } from 'vitest';
import { aggregateAspects, ASPECT_LABELS, type ZeroShotResult } from './aspects';
import type { ReviewScore } from './types';

function review(robertaPos: number, robertaNeg: number): ReviewScore {
  return {
    text: 'x',
    rating: 3,
    vader: { compound: 0, pos: 0, neg: 0, neu: 1 },
    roberta: { positive: robertaPos, negative: robertaNeg, neutral: 1 - robertaPos - robertaNeg },
    disagreement: 0,
  };
}

/** Zero-shot result scoring `hits` labels at 0.9 and everything else at 0.1. */
function zs(...hits: string[]): ZeroShotResult {
  return {
    labels: [...ASPECT_LABELS],
    scores: ASPECT_LABELS.map(l => (hits.includes(l) ? 0.9 : 0.1)),
  };
}

describe('aggregateAspects', () => {
  it('includes aspect mentioned by >= 2 reviews with mean polarity of mentioners', () => {
    const reviews = [review(0.9, 0.1), review(0.5, 0.3), review(0.1, 0.8)];
    const zsResults = [zs('taste & flavor'), zs('taste & flavor'), zs('quality')];
    const out = aggregateAspects(zsResults, reviews);
    expect(out).toHaveLength(1);
    expect(out[0].label).toBe('taste & flavor');
    expect(out[0].mentions).toBe(2);
    expect(out[0].polarity).toBeCloseTo(((0.9 - 0.1) + (0.5 - 0.3)) / 2, 10);
  });

  it('excludes aspects with fewer than 2 mentions', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1)];
    const zsResults = [zs('quality'), zs('ease of use')];
    expect(aggregateAspects(zsResults, reviews)).toHaveLength(0);
  });

  it('scores below the 0.7 threshold do not count as mentions', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1)];
    // zs() scores non-hit labels at 0.1 — all below threshold
    const zsResults = [zs(), zs()];
    expect(aggregateAspects(zsResults, reviews)).toHaveLength(0);
  });

  it('sorts by mentions desc, ties by label asc', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1), review(0.7, 0.1)];
    const zsResults = [
      zs('quality', 'value for money'),
      zs('quality', 'value for money'),
      zs('quality'),
    ];
    const out = aggregateAspects(zsResults, reviews);
    expect(out.map(a => a.label)).toEqual(['quality', 'value for money']);
    expect(out[0].mentions).toBe(3);
    expect(out[1].mentions).toBe(2);
  });

  it('handles unknown labels in zero-shot output by ignoring them', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1)];
    const weird: ZeroShotResult = { labels: ['something else'], scores: [0.99] };
    expect(aggregateAspects([weird, weird], reviews)).toHaveLength(0);
  });

  it('throws on length mismatch between results and reviews', () => {
    const reviews = [review(0.9, 0.1)];
    expect(() => aggregateAspects([], reviews)).toThrow();
  });

  it('returns empty array for empty inputs', () => {
    expect(aggregateAspects([], [])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `npm test`
Expected: FAIL — cannot find module './aspects'. (The 7 existing verdict tests still pass.)

- [ ] **Step 3: Implement `lib/aspects.ts`**

```typescript
import type { ReviewScore, AspectScore } from './types';

export const ASPECT_LABELS = [
  'taste & flavor',
  'quality',
  'value for money',
  'packaging & shipping',
  'ease of use',
] as const;

/** Zero-shot score >= this counts the review as mentioning the aspect. */
export const MENTION_THRESHOLD = 0.7;
/** Aspects mentioned by fewer reviews than this are dropped (too thin to chart). */
export const MIN_MENTIONS = 2;

/** Raw multi-label zero-shot output for one review (labels/scores are parallel arrays). */
export interface ZeroShotResult {
  labels: string[];
  scores: number[];
}

/**
 * Aggregate per-review zero-shot aspect detections into product-level aspect scores.
 * Aspect polarity reuses each mentioning review's RoBERTa polarity (review-level
 * attribution — documented approximation, honest at 8-review scale).
 * Throws on length mismatch (pipeline bug); zeroShot[i] must describe reviews[i].
 */
export function aggregateAspects(
  zeroShot: ZeroShotResult[],
  reviews: ReviewScore[],
): AspectScore[] {
  if (zeroShot.length !== reviews.length) {
    throw new Error('aggregateAspects: zeroShot/reviews length mismatch');
  }

  const out: AspectScore[] = [];

  for (const label of ASPECT_LABELS) {
    const polarities: number[] = [];
    for (let i = 0; i < zeroShot.length; i++) {
      const idx = zeroShot[i].labels.indexOf(label);
      if (idx === -1) continue;
      const score = zeroShot[i].scores[idx];
      if (typeof score === 'number' && score >= MENTION_THRESHOLD) {
        polarities.push(reviews[i].roberta.positive - reviews[i].roberta.negative);
      }
    }
    if (polarities.length >= MIN_MENTIONS) {
      out.push({
        label,
        polarity: polarities.reduce((a, b) => a + b, 0) / polarities.length,
        mentions: polarities.length,
      });
    }
  }

  return out.sort((a, b) => b.mentions - a.mentions || a.label.localeCompare(b.label));
}
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `npm test`
Expected: PASS — 14 tests total (7 verdict + 7 aspects).

- [ ] **Step 5: Commit**

```bash
git add lib/aspects.ts lib/aspects.test.ts
git commit -m "feat: aspect aggregation math (zero-shot mentions, mean polarity, thresholds)"
```

---

### Task 3: Route — zero-shot calls, fail-soft, cache envelope v2, maxDuration

**Files:**
- Modify: `app/api/analyze/route.ts`

Seven edits. The route's failure-cache semantics for the main pipeline must NOT change — aspects are strictly additive and fail-soft.

- [ ] **Step 1: Imports + constants**

Add to the existing type import (line 5):

```typescript
import type { ReviewScore, VaderScore, RobertaScore, AspectScore, ScoredCacheV2 } from '@/lib/types';
```

Add below it:

```typescript
import { aggregateAspects, ASPECT_LABELS, type ZeroShotResult } from '@/lib/aspects';
```

Change `export const maxDuration = 120;` to:

```typescript
export const maxDuration = 180; // worst case: 84.7s pipeline + 8 zero-shot calls x 5s
```

Add below `HF_URL`:

```typescript
const ZERO_SHOT_URL =
  'https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli';
const ZERO_SHOT_TIMEOUT_MS = 5_000;
```

- [ ] **Step 2: Bump scored key to v2 (fail/lock keys stay v1 — shapes unchanged)**

Replace:

```typescript
const scoredKey = `asin:v1:${normalizedAsin}:scored`;
```

with:

```typescript
// v2 = envelope { reviews, productTitle?, aspects?, analyzedAt }. v1 keys (bare
// ReviewScore[]) are never read again and expire via their 24h TTL — no migration.
const scoredKey = `asin:v2:${normalizedAsin}:scored`;
```

- [ ] **Step 3: Cache-hit path (Step 3 in route) returns envelope fields**

Replace the first cache-hit block:

```typescript
const cachedScored = await kv.get<ScoredCacheV2>(scoredKey);
if (cachedScored) {
  return NextResponse.json({
    reviews: cachedScored.reviews,
    count: cachedScored.reviews.length,
    asin: normalizedAsin,
    productTitle: cachedScored.productTitle,
    aspects: cachedScored.aspects,
  });
}
```

- [ ] **Step 4: In-lock cache re-check (Step 7 in route) — same envelope shape**

Replace the re-check block:

```typescript
const recheckScored = await kv.get<ScoredCacheV2>(scoredKey);
if (recheckScored) {
  return NextResponse.json({
    reviews: recheckScored.reviews,
    count: recheckScored.reviews.length,
    asin: normalizedAsin,
    productTitle: recheckScored.productTitle,
    aspects: recheckScored.aspects,
  });
}
```

- [ ] **Step 5: Fail-soft zero-shot block — insert AFTER the main pipeline `catch` block closes (after the `return NextResponse.json({ error: \`Analysis failed: ...\` }...)` block, line ~295) and BEFORE "Step 12: Cache success"**

```typescript
// Step 11.5: Aspect detection (fail-soft — aspects are an enhancement, never a blocker).
// Errors here must NOT reach the failure cache; we log and omit aspects instead.
let aspects: AspectScore[] | undefined;
try {
  const zeroShotResults: ZeroShotResult[] = [];
  for (let i = 0; i < scored!.length; i++) {
    const zsRes = await fetch(ZERO_SHOT_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.HF_API_KEY!}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: scored![i].text,
        parameters: { candidate_labels: [...ASPECT_LABELS], multi_label: true },
      }),
      signal: AbortSignal.timeout(ZERO_SHOT_TIMEOUT_MS),
    });
    if (!zsRes.ok) throw new Error(`zeroshot_${zsRes.status}`);
    const raw: unknown = await zsRes.json();
    const candidate = raw as { labels?: unknown; scores?: unknown };
    if (!Array.isArray(candidate.labels) || !Array.isArray(candidate.scores)) {
      throw new Error('zeroshot_unexpected_shape');
    }
    zeroShotResults.push({
      labels: candidate.labels as string[],
      scores: candidate.scores as number[],
    });
    if (i < scored!.length - 1) {
      await new Promise<void>((resolve) => setTimeout(resolve, 100));
    }
  }
  aspects = aggregateAspects(zeroShotResults, scored!);
} catch (e) {
  console.error('aspects_failed', e);
  aspects = undefined;
}
```

- [ ] **Step 6: Cache write stores the envelope (Step 12 in route)**

Replace:

```typescript
await kv.set(scoredKey, JSON.stringify(scored!), { ex: 86400 });
```

with:

```typescript
const envelope: ScoredCacheV2 = {
  reviews: scored!,
  productTitle,
  aspects,
  analyzedAt: new Date().toISOString(),
};
await kv.set(scoredKey, JSON.stringify(envelope), { ex: 86400 });
```

- [ ] **Step 7: Fresh-path response includes aspects (Step 14 in route)**

Replace the final return:

```typescript
return NextResponse.json({
  reviews: scored!,
  count: scored!.length,
  asin: normalizedAsin,
  productTitle,
  aspects,
});
```

- [ ] **Step 8: Verify**

```bash
npm test && npx tsc --noEmit && npm run lint && npm run build
```

Expected: 14 tests pass, zero TS/lint/build errors.

- [ ] **Step 9: Commit**

```bash
git add app/api/analyze/route.ts
git commit -m "feat: zero-shot aspect detection (fail-soft) + scored cache envelope v2"
```

---

### Task 4: `components/AspectBars.tsx`

**Files:**
- Create: `components/AspectBars.tsx`

Presentational only. Polarity bar: width `|polarity| * 100`%, green when polarity ≥ 0, red otherwise. Parent decides whether to render (only when aspects non-empty).

- [ ] **Step 1: Create `components/AspectBars.tsx`**

```tsx
import type { AspectScore } from '@/lib/types';

interface Props {
  aspects: AspectScore[];
}

export default function AspectBars({ aspects }: Props) {
  return (
    <section aria-label="What reviewers say" className="mt-8">
      <h2 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4">
        What reviewers say
      </h2>
      <div className="rounded-2xl border border-slate-700/60 bg-slate-800/60 p-5 backdrop-blur-sm">
        <ul className="space-y-3">
          {aspects.map(aspect => {
            const positive = aspect.polarity >= 0;
            const width = Math.min(100, Math.round(Math.abs(aspect.polarity) * 100));
            return (
              <li key={aspect.label} className="flex items-center gap-3">
                <span className="w-36 shrink-0 truncate text-xs font-mono text-slate-300">
                  {aspect.label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-700">
                  <div
                    className={positive ? 'h-full bg-green-500' : 'h-full bg-red-400'}
                    style={{ width: `${width}%` }}
                  />
                </div>
                <span
                  className={`w-12 shrink-0 text-right text-xs font-mono ${
                    positive ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {aspect.polarity >= 0 ? '+' : '−'}
                  {Math.abs(aspect.polarity).toFixed(2)}
                </span>
                <span className="w-8 shrink-0 text-right text-[10px] font-mono text-slate-500">
                  {aspect.mentions}×
                </span>
              </li>
            );
          })}
        </ul>
        <p className="mt-4 text-[10px] font-mono text-slate-600">
          zero-shot NLI · facebook/bart-large-mnli · review-level attribution
        </p>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add components/AspectBars.tsx
git commit -m "feat: aspect polarity bars component"
```

---

### Task 5: Wire aspects into `app/page.tsx`

**Files:**
- Modify: `app/page.tsx`

Read the file first; it now has `beginAnalysis` (post-Phase-1). Four edits:

- [ ] **Step 1: Imports**

Add with the other component imports:

```tsx
import AspectBars from '@/components/AspectBars';
```

Extend the existing type import to include `AspectScore`:

```tsx
import type { ReviewScore, AnalyzeApiResponse, AspectScore } from '@/lib/types';
```

- [ ] **Step 2: State**

After the `productTitle` state line, add:

```tsx
const [aspects, setAspects] = useState<AspectScore[] | undefined>(undefined);
```

- [ ] **Step 3: Set + reset**

In `doAnalyze` success branch, after `setProductTitle(response.productTitle);` add:

```tsx
setAspects(response.aspects);
```

In `beginAnalysis`, after `setProductTitle(undefined);` add:

```tsx
setAspects(undefined);
```

In `handleClear`, after `setProductTitle(undefined);` add:

```tsx
setAspects(undefined);
```

- [ ] **Step 4: Render between VerdictCard and the "Model deep-dive" div**

```tsx
{aspects && aspects.length > 0 && <AspectBars aspects={aspects} />}
```

- [ ] **Step 5: Verify**

```bash
npm test && npx tsc --noEmit && npm run lint && npm run build
```

Expected: all green.

- [ ] **Step 6: Manual smoke (cached ASINs have no aspects yet — section correctly hidden)**

```bash
npm run dev
```

- Click a gallery chip → results render WITHOUT "What reviewers say" (old v1 cache is gone; this triggers a fresh analyze — see note below)
- Note: with the key bump, previously cached ASINs are cache-misses now; a fresh analyze (10–60s) re-scores and returns aspects. Verify one ASIN end-to-end locally ONLY if `.env.development.local` is fresh (`vercel env pull`); otherwise defer end-to-end to production verification. Each fresh analyze costs 1 Canopy call (5/hour budget).

- [ ] **Step 7: Commit**

```bash
git add app/page.tsx
git commit -m "feat: render aspect bars in results panel"
```

---

### Task 6: Code review + merge (workflow gate)

- [ ] **Step 1: SHAs**

```bash
BASE_SHA=$(git merge-base main HEAD)
HEAD_SHA=$(git rev-parse HEAD)
```

- [ ] **Step 2: Dispatch reviewer (spec + quality), fix all Critical/Important, re-review until clean**

- [ ] **Step 3: Merge + push**

```bash
git checkout main && git merge --no-ff feat-aspects && git push origin main && git branch -d feat-aspects
```

- [ ] **Step 4: Production verification (burns 1-2 Canopy calls + KV writes)**

- `curl -s "https://sentiment-amazon-analyzer.vercel.app/api/analyze?asin=B01B57DVNE"` → first call after deploy is a v2 cache miss → full pipeline (~30-60s, may 202) → response includes `productTitle` and (if zero-shot succeeded) `aspects`
- Repeat same curl → instant cache hit → MUST now include `productTitle` + `aspects` (the Day-6 backlog item closes here)
- Browser: analyze same ASIN → "What reviewers say" bars render between verdict and deep-dive

---

## Self-Review Notes

- Spec coverage (Phase 2 section): zero-shot per review w/ 5s timeout + sequential 100ms gap ✅ Task 3 Step 5 · labels + 0.7 threshold + ≥2 mentions ✅ Task 2 · fail-soft ✅ Task 3 Step 5 · maxDuration 180 ✅ Task 3 Step 1 · v2 envelope key + shape + no migration ✅ Task 3 Steps 2/3/4/6 · productTitle on cache hits ✅ Task 3 Steps 3/4 · UI section hidden when absent ✅ Task 5 Step 4 · types ✅ Task 1 · vitest for aggregation ✅ Task 2
- Type consistency: `ZeroShotResult`/`aggregateAspects`/`ASPECT_LABELS` (Task 2) match Task 3 usage; `AspectScore`/`ScoredCacheV2` (Task 1) match Tasks 3-5; `aspects` state type matches `AnalyzeApiResponse.aspects`
- Known cost: key bump invalidates 5 cached ASINs → each re-analyze burns Canopy quota; acceptable, seeding planned anyway
- No placeholders
