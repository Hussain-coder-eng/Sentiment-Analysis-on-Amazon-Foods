# Design: Remove Demo Mode — Hero Landing + ASIN-Only UX

**Date:** 2026-06-09  
**Status:** APPROVED  
**Branch target:** `feat/hero-landing`

---

## Problem Statement

The current landing page loads 500 pre-scored Amazon food reviews automatically and renders
them as a scatter plot before the user does anything. This framing confuses the product:
it looks like a food-review dashboard, not a general-purpose Amazon product analyzer.
The original spec intended any-product analysis from day one. The demo mode is an artifact
of the initial build order, not a feature worth keeping.

---

## Goal

Replace the food-review demo landing with a clean hero + ASIN form. No results until the
user submits. Any Amazon product — not just food.

---

## What Gets Deleted

| File | Action |
|------|--------|
| `app/api/demo/route.ts` | Delete entirely |
| `public/demo-data.json` | Delete entirely |
| `lib/types.ts` → `DemoApiResponse` interface | Remove, keep rest of file |

`/api/warmup` is unchanged — still fires on mount to warm the HF model.

---

## `app/page.tsx` — New Structure

### State

```typescript
const [asinInput, setAsinInput] = useState('');
const [analyzing, setAnalyzing] = useState(false);
const [analyzeError, setAnalyzeError] = useState<string | null>(null);
const [retryCountdown, setRetryCountdown] = useState<number | null>(null);
const [reviews, setReviews] = useState<ReviewScore[] | null>(null);
const [resultAsin, setResultAsin] = useState<string | null>(null);
const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
```

Removed from current state: `demoLoading`, `demoError`, `isLive`.  
`reviews` is now `ReviewScore[] | null` — `null` means no results yet.

### Imports removed

`DemoApiResponse` import removed from `@/lib/types`.

### Functions removed

`loadDemo()` — deleted entirely.

### Functions kept / unchanged

- `startCountdown()` — unchanged
- `doAnalyze()` — add `setResultAsin(asin)` on success path
- `handleAnalyze()` — at start, clear `reviews`, `resultAsin`, `analyzeError` before setting `analyzing = true`; this prevents old results re-appearing if the new submit fails

### "Clear" button replaces "Reset to demo"

Show when `reviews !== null && !analyzing`. Clicking sets `reviews = null`, `resultAsin = null`, `analyzeError = null`, clears countdown.

### Layout

```
<main className="min-h-screen p-8">

  [Hero]
  <h1>Amazon Review Sentiment Analyzer</h1>
  <p>Enter any Amazon ASIN to analyze real customer reviews
     with VADER and RoBERTa sentiment models.</p>

  [Form]
  <form>
    <label htmlFor="asin-input" className="sr-only">Amazon ASIN</label>
    <input id="asin-input" ... placeholder="Enter ASIN (e.g. B000E7L2R4)" />
    <button type="submit">{analyzing ? 'Analyzing…' : 'Analyze'}</button>
    {reviews !== null && !analyzing && (
      <button type="button" onClick={handleClear}>Clear</button>
    )}
  </form>

  [Status]
  {analyzing && <p>{retryCountdown ? `Retrying in ${retryCountdown}s…` : 'Waking up the ML model…'}</p>}
  {analyzeError && <p className="text-red-500">{analyzeError}</p>}

  [Results — only when reviews !== null && !analyzing]
  {reviews !== null && !analyzing && (
    <>
      <p>Showing {reviews.length} reviews for ASIN {resultAsin}</p>
      <SentimentPlot reviews={reviews} />
      <DisagreementPanel reviews={reviews} />
    </>
  )}

</main>
```

### `useEffect` (mount)

```typescript
useEffect(() => {
  if (typeof window !== 'undefined' && !sessionStorage.getItem('warmup-done')) {
    sessionStorage.setItem('warmup-done', '1');
    fetch('/api/warmup').catch(() => {});
  }
}, []);
```

No demo fetch. Warmup only.

---

## `lib/types.ts` — Change

Remove `DemoApiResponse` interface. Keep `VaderScore`, `RobertaScore`, `ReviewScore`, `AnalyzeApiResponse`.

---

## Scope Boundaries

**In scope:**
- Delete demo files and route
- Rewrite `page.tsx` as described
- Remove `DemoApiResponse` from types

**Out of scope (not this change):**
- Any changes to `/api/analyze`
- Any changes to `SentimentPlot` or `DisagreementPanel` components
- Product name lookup (fetch product title from Canopy alongside reviews)
- Amazon URL → ASIN extraction

---

## Error Handling (unchanged from current)

All existing error paths in `doAnalyze` stay as-is. The 202 auto-retry with countdown is preserved. Errors display inline below the form.

---

## Confirmed Working ASINs (for placeholder text reference)

- `B000E7L2R4` — 8 reviews, cached
- `B00032G1S0` — 5 reviews, cached
