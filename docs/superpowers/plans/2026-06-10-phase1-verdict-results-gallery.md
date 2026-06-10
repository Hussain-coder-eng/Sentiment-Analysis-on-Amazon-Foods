# Phase 1: Verdict Layer + Editorial Results + Example Gallery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verdict-first editorial results page (score card → charts → how-it-works) plus a one-click example gallery of cached ASINs on the landing form.

**Architecture:** Pure verdict math in `lib/verdict.ts` (unit-tested with vitest), presentational `VerdictCard` + `HowItWorksStrip` components, gallery data const in `lib/gallery.ts`, wired into `app/page.tsx`. No API changes in this phase.

**Tech Stack:** Next.js 14 (App Router, client page), TypeScript, Tailwind, vitest (new dev dep). Existing: anime.js animations, Plotly chart, shadcn cards.

**Branch:** work on `feat-portfolio-upgrade` (already created, spec committed).

**Spec:** `docs/superpowers/specs/2026-06-10-portfolio-upgrade-design.md`

---

### Task 1: Vitest setup + repo chore

**Files:**
- Modify: `package.json` (scripts + devDependencies)
- Create: `vitest.config.ts`
- Modify: `.gitignore`

- [ ] **Step 1: Install vitest**

```bash
npm install -D vitest
```

Expected: `vitest` appears in `devDependencies` (v3.x).

- [ ] **Step 2: Create `vitest.config.ts`**

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['lib/**/*.test.ts'],
  },
});
```

- [ ] **Step 3: Add test script to `package.json`**

In `"scripts"`, after `"lint"`:

```json
"test": "vitest run"
```

- [ ] **Step 4: Add `.superpowers/` to `.gitignore`**

Append line to `.gitignore`:

```
.superpowers/
```

- [ ] **Step 5: Verify vitest runs (no tests yet — expect "no test files found" exit)**

Run: `npm test`
Expected: vitest runs, reports no test files found (non-zero exit is fine at this step).

- [ ] **Step 6: Commit**

```bash
git add package.json package-lock.json vitest.config.ts .gitignore
git commit -m "chore: add vitest + ignore .superpowers brainstorm artifacts"
```

---

### Task 2: `lib/verdict.ts` — verdict math (TDD)

**Files:**
- Create: `lib/verdict.test.ts`
- Create: `lib/verdict.ts`

Verdict semantics (from spec):
- Review polarity = `roberta.positive - roberta.negative` ∈ [−1, 1]
- Score = `Math.round((meanPolarity + 1) * 50)` → 0–100
- Bands: ≥75 "Loved it" · 60–74 "Mostly positive" · 45–59 "Mixed" · <45 "Mostly negative"
- Positive review: polarity > 0
- Models agree on a review when `(vader.compound >= 0) === (polarity >= 0)`
- One-liner: `"{positiveCount} of {total} reviews positive · models agree {agreementPct}%"`
- Empty input throws (upstream API guarantees ≥5 reviews; throwing surfaces pipeline bugs loudly)

- [ ] **Step 1: Write failing tests `lib/verdict.test.ts`**

```typescript
import { describe, it, expect } from 'vitest';
import { computeVerdict } from './verdict';
import type { ReviewScore } from './types';

/** Build a minimal ReviewScore; only fields verdict math reads are meaningful. */
function review(robertaPos: number, robertaNeg: number, vaderCompound: number): ReviewScore {
  return {
    text: 'x',
    rating: 3,
    vader: { compound: vaderCompound, pos: 0, neg: 0, neu: 1 },
    roberta: { positive: robertaPos, negative: robertaNeg, neutral: 1 - robertaPos - robertaNeg },
    disagreement: 0,
  };
}

describe('computeVerdict', () => {
  it('throws on empty reviews', () => {
    expect(() => computeVerdict([])).toThrow();
  });

  it('all strongly positive → high score, "Loved it"', () => {
    const v = computeVerdict([review(0.9, 0.05, 0.8), review(0.95, 0.02, 0.9)]);
    expect(v.score).toBeGreaterThanOrEqual(75);
    expect(v.label).toBe('Loved it');
    expect(v.positiveCount).toBe(2);
    expect(v.total).toBe(2);
  });

  it('all strongly negative → low score, "Mostly negative"', () => {
    const v = computeVerdict([review(0.05, 0.9, -0.7), review(0.02, 0.95, -0.8)]);
    expect(v.score).toBeLessThan(45);
    expect(v.label).toBe('Mostly negative');
    expect(v.positiveCount).toBe(0);
  });

  it('neutral polarity → score 50, "Mixed"', () => {
    const v = computeVerdict([review(0.3, 0.3, 0.0)]);
    expect(v.score).toBe(50);
    expect(v.label).toBe('Mixed');
  });

  it('band boundaries: 75 → Loved it, 60 → Mostly positive, 45 → Mixed', () => {
    // polarity 0.5 → score 75
    expect(computeVerdict([review(0.6, 0.1, 0.5)]).score).toBe(75);
    expect(computeVerdict([review(0.6, 0.1, 0.5)]).label).toBe('Loved it');
    // polarity 0.2 → score 60
    expect(computeVerdict([review(0.4, 0.2, 0.3)]).label).toBe('Mostly positive');
    // polarity -0.1 → score 45
    expect(computeVerdict([review(0.2, 0.3, -0.1)]).label).toBe('Mixed');
  });

  it('agreement: same-sign reviews agree, opposite-sign disagree', () => {
    const v = computeVerdict([
      review(0.8, 0.1, 0.7),   // polarity + , vader + → agree
      review(0.1, 0.8, 0.5),   // polarity − , vader + → disagree
    ]);
    expect(v.agreementPct).toBe(50);
  });

  it('one-liner format', () => {
    const v = computeVerdict([review(0.8, 0.1, 0.7), review(0.1, 0.8, -0.5)]);
    expect(v.oneLiner).toBe('1 of 2 reviews positive · models agree 100%');
  });
});
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `npm test`
Expected: FAIL — `Cannot find module './verdict'` (or equivalent).

- [ ] **Step 3: Implement `lib/verdict.ts`**

```typescript
import type { ReviewScore } from './types';

const LOVED_THRESHOLD = 75;
const POSITIVE_THRESHOLD = 60;
const MIXED_THRESHOLD = 45;

export type VerdictLabel = 'Loved it' | 'Mostly positive' | 'Mixed' | 'Mostly negative';

export interface Verdict {
  score: number;        // 0-100
  label: VerdictLabel;
  positiveCount: number;
  total: number;
  agreementPct: number; // 0-100, % of reviews where VADER and RoBERTa share sign
  oneLiner: string;
}

/** RoBERTa polarity for one review: positive minus negative probability, in [-1, 1]. */
function polarity(r: ReviewScore): number {
  return r.roberta.positive - r.roberta.negative;
}

function labelFor(score: number): VerdictLabel {
  if (score >= LOVED_THRESHOLD) return 'Loved it';
  if (score >= POSITIVE_THRESHOLD) return 'Mostly positive';
  if (score >= MIXED_THRESHOLD) return 'Mixed';
  return 'Mostly negative';
}

/** Synthesize a product-level verdict from per-review model scores. Throws on empty input. */
export function computeVerdict(reviews: ReviewScore[]): Verdict {
  if (reviews.length === 0) {
    throw new Error('computeVerdict requires at least one review');
  }

  const polarities = reviews.map(polarity);
  const mean = polarities.reduce((a, b) => a + b, 0) / polarities.length;
  const score = Math.min(100, Math.max(0, Math.round((mean + 1) * 50)));

  const positiveCount = polarities.filter(p => p > 0).length;
  // Models "agree" when both place the review on the same side of neutral.
  const agreeCount = reviews.filter(
    (r, i) => (r.vader.compound >= 0) === (polarities[i] >= 0)
  ).length;
  const agreementPct = Math.round((agreeCount / reviews.length) * 100);

  return {
    score,
    label: labelFor(score),
    positiveCount,
    total: reviews.length,
    agreementPct,
    oneLiner: `${positiveCount} of ${reviews.length} reviews positive · models agree ${agreementPct}%`,
  };
}
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `npm test`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add lib/verdict.ts lib/verdict.test.ts
git commit -m "feat: verdict synthesis math (score, label bands, model agreement)"
```

---

### Task 3: `lib/gallery.ts` — example gallery data

**Files:**
- Create: `lib/gallery.ts`

Data-only module. The 5 ASINs are confirmed Canopy-indexed and KV-cached (see HANDOFF.md table). First two predate title capture — generic labels until reseeded; swap for popular products as future seeding lands (spec "ASIN Seeding").

- [ ] **Step 1: Create `lib/gallery.ts`**

```typescript
export interface GalleryItem {
  asin: string;
  shortName: string;
  emoji: string;
}

/**
 * Known-cached ASINs (instant results, no Canopy call, no rate-limit cost).
 * Keep in sync with HANDOFF.md "Confirmed Working ASINs" table.
 * First item doubles as the product that emerges from the 3D box hero (Phase 3).
 */
export const GALLERY_ITEMS: GalleryItem[] = [
  { asin: 'B01B57DVNE', shortName: "Jack Link's Snack Sticks", emoji: '🥩' },
  { asin: 'B017835JPC', shortName: 'Teriyaki Snack Packs', emoji: '🍖' },
  { asin: 'B0C2FV4W2S', shortName: 'Snack Mix Variety', emoji: '🥨' },
  { asin: 'B000E7L2R4', shortName: 'Fine Foods Classic', emoji: '🛒' },
  { asin: 'B00032G1S0', shortName: 'Pantry Pick', emoji: '🍪' },
];
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add lib/gallery.ts
git commit -m "feat: example gallery data for cached ASINs"
```

---

### Task 4: `components/VerdictCard.tsx`

**Files:**
- Create: `components/VerdictCard.tsx`

Presentational only — receives a computed `Verdict`. Score ring is an SVG circle with `strokeDasharray` proportional to score. Matches dark editorial style (gradient card, centered, green accent at positive labels, red at "Mostly negative").

- [ ] **Step 1: Create `components/VerdictCard.tsx`**

```tsx
import type { Verdict } from '@/lib/verdict';

interface Props {
  verdict: Verdict;
  asin: string;
  productTitle?: string;
}

const RING_RADIUS = 34;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

export default function VerdictCard({ verdict, asin, productTitle }: Props) {
  const isNegative = verdict.label === 'Mostly negative';
  const accent = isNegative ? '#F87171' : '#22C55E';
  const dash = (verdict.score / 100) * RING_CIRCUMFERENCE;

  return (
    <section
      aria-label="Verdict"
      className="rounded-2xl border border-slate-700/60 bg-gradient-to-br from-slate-800 to-slate-800/40 px-6 py-8 text-center backdrop-blur-sm"
    >
      <p className="text-slate-500 text-xs font-mono tracking-wide uppercase truncate">
        {productTitle ?? 'Product'} · {asin}
      </p>

      <div className="relative mx-auto mt-5 h-24 w-24">
        <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90" aria-hidden="true">
          <circle cx="40" cy="40" r={RING_RADIUS} fill="none" stroke="#334155" strokeWidth="6" />
          <circle
            cx="40"
            cy="40"
            r={RING_RADIUS}
            fill="none"
            stroke={accent}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${RING_CIRCUMFERENCE - dash}`}
          />
        </svg>
        <span
          className="absolute inset-0 flex items-center justify-center font-heading text-3xl font-bold"
          style={{ color: accent }}
        >
          {verdict.score}
        </span>
      </div>

      <h2 className="mt-4 font-heading text-2xl font-bold text-white">{verdict.label}</h2>
      <p className="mt-2 text-sm font-mono text-slate-400">{verdict.oneLiner}</p>
    </section>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add components/VerdictCard.tsx
git commit -m "feat: verdict hero card with score ring"
```

---

### Task 5: `components/HowItWorksStrip.tsx`

**Files:**
- Create: `components/HowItWorksStrip.tsx`

Static 4-step pipeline explainer shown at the bottom of results. (Phase 2 adds real aspects; the step copy already names the full pipeline per spec.)

- [ ] **Step 1: Create `components/HowItWorksStrip.tsx`**

```tsx
const STEPS: { title: string; body: string }[] = [
  { title: 'Fetch reviews', body: 'Top customer reviews pulled live from Amazon via the Canopy GraphQL API.' },
  { title: 'Score twice', body: 'Each review scored by VADER (lexicon) and RoBERTa (transformer) independently.' },
  { title: 'Find aspects', body: 'Zero-shot NLI tags what reviewers discuss — taste, value, packaging, more.' },
  { title: 'Synthesize', body: 'Scores fuse into a 0–100 verdict with model-agreement transparency.' },
];

export default function HowItWorksStrip() {
  return (
    <section aria-label="How it works" className="mt-10">
      <h2 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4">
        How it works
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {STEPS.map((step, i) => (
          <div
            key={step.title}
            className="rounded-xl border border-slate-700/60 bg-slate-800/60 p-4"
          >
            <span className="font-mono text-xs text-green-400">0{i + 1}</span>
            <h3 className="mt-1 text-sm font-semibold text-slate-100">{step.title}</h3>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">{step.body}</p>
          </div>
        ))}
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
git add components/HowItWorksStrip.tsx
git commit -m "feat: how-it-works pipeline strip"
```

---

### Task 6: Wire into `app/page.tsx` — gallery chips + editorial results order

**Files:**
- Modify: `app/page.tsx`

Three edits. Existing animations, retry logic, error handling untouched.

- [ ] **Step 1: Add imports (after existing imports, `app/page.tsx:6`)**

```tsx
import VerdictCard from '@/components/VerdictCard';
import HowItWorksStrip from '@/components/HowItWorksStrip';
import { GALLERY_ITEMS } from '@/lib/gallery';
import { computeVerdict } from '@/lib/verdict';
```

- [ ] **Step 2: Add gallery click handler inside `Home` (after `handleClear`, ~line 179)**

Submits via the same path as the form so all validation/retry behavior is shared:

```tsx
function handleGalleryClick(asin: string) {
  if (analyzing) return;
  setAsinInput(asin);
  if (countdownRef.current) clearInterval(countdownRef.current);
  setRetryCountdown(null);
  setAnalyzeError(null);
  setReviews(null);
  setResultAsin(null);
  setProductTitle(undefined);
  setAnalyzing(true);
  doAnalyze(asin, 1)
    .catch(() => setAnalyzeError('Network error — please try again.'))
    .finally(() => {
      setAnalyzing(false);
      setRetryCountdown(null);
      if (countdownRef.current) clearInterval(countdownRef.current);
    });
}
```

Also fix a latent bug while here: `handleAnalyze` never resets `productTitle` (only `handleClear` does, line 178). Add `setProductTitle(undefined);` after `setResultAsin(null);` in `handleAnalyze` (~line 159) so a stale title can't leak into the next run's results.

- [ ] **Step 3: Add gallery chips UI — insert directly after the `</form>` closing tag (~line 297)**

```tsx
{/* Example gallery — cached ASINs, instant results */}
{reviews === null && !analyzing && (
  <div className="mb-10">
    <p className="text-slate-500 text-xs font-mono tracking-widest uppercase mb-3">
      — or try one —
    </p>
    <div className="flex flex-wrap gap-2">
      {GALLERY_ITEMS.map(item => (
        <button
          key={item.asin}
          type="button"
          onClick={() => handleGalleryClick(item.asin)}
          className="rounded-full border border-slate-700 bg-slate-800/70 px-4 py-2 text-xs font-mono text-slate-300 transition-all duration-150 hover:border-green-500/50 hover:text-green-400 cursor-pointer focus:outline-none focus:ring-2 focus:ring-green-500/60"
        >
          {item.emoji} {item.shortName}
        </button>
      ))}
    </div>
    <p className="mt-2 text-[11px] font-mono text-slate-600">
      cached — instant results, no rate limit
    </p>
  </div>
)}
```

- [ ] **Step 4: Replace results block (lines 299–315) with editorial order**

Replace the entire `{reviews !== null && !analyzing && (...)}` block with:

```tsx
{reviews !== null && !analyzing && (
  <div ref={resultsRef} className="results-panel opacity-0">
    <VerdictCard
      verdict={computeVerdict(reviews)}
      asin={resultAsin ?? ''}
      productTitle={productTitle}
    />
    <div className="mt-8">
      <h2 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4">
        Model deep-dive
      </h2>
      <SentimentPlot reviews={reviews} />
      <DisagreementPanel reviews={reviews} />
    </div>
    <HowItWorksStrip />
  </div>
)}
```

(The old "N reviews scored for ASIN" context box is superseded by VerdictCard, which shows title, ASIN, and counts.)

- [ ] **Step 5: Verify — tests, types, lint, build**

```bash
npm test && npx tsc --noEmit && npm run lint && npm run build
```

Expected: all pass, zero errors.

- [ ] **Step 6: Manual smoke test**

```bash
npm run dev
```

- Landing shows gallery chips; click one → instant cached result
- Results: VerdictCard (score ring, label, one-liner) → deep-dive heading → plot → disagreements → how-it-works strip
- Clear button resets; manual ASIN entry still works; URL paste still extracts

- [ ] **Step 7: Commit**

```bash
git add app/page.tsx
git commit -m "feat: verdict-first editorial results + example gallery chips"
```

---

### Task 7: Code review + merge (workflow gate)

Per repo rules (CLAUDE.md): review before merge, no exceptions.

- [ ] **Step 1: Compute review SHAs**

```bash
BASE_SHA=$(git merge-base main HEAD)
HEAD_SHA=$(git rev-parse HEAD)
```

- [ ] **Step 2: Dispatch `superpowers:code-reviewer` subagent with SHAs + spec + plan paths**

- [ ] **Step 3: Fix all Critical/Important findings on branch, re-review until clean**

- [ ] **Step 4: Merge**

```bash
git checkout main && git merge --no-ff feat-portfolio-upgrade && git push origin main
```

(Keep branch for Phases 2–4? No — one branch per phase. Delete after merge: `git branch -d feat-portfolio-upgrade`. Phase 2 starts fresh from main.)

---

## Self-Review Notes

- Spec coverage (Phase 1 section): verdict math ✅ Task 2 · VerdictCard ✅ Task 4 · editorial order ✅ Task 6 · how-it-works strip ✅ Task 5 · gallery ✅ Tasks 3+6 · vitest ✅ Task 1 · `.gitignore` chore ✅ Task 1
- Types consistent: `Verdict`/`computeVerdict` (Task 2) match usage (Tasks 4, 6); `GalleryItem.emoji/shortName/asin` (Task 3) match Task 6 usage
- No placeholders; all code complete
- Phases 2–4 get their own plans after Phase 1 ships
