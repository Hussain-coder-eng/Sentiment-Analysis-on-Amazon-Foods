# Plan: Day 5 UX Features

**Branch:** `feat/day5-ux`  
**Base:** `main`

## Context

Next.js sentiment analysis app. Dark cinema theme. ASIN-only UX (no demo mode).

Key files:
- `app/page.tsx` — hero landing, ASIN input, results display, anime.js animations
- `app/api/analyze/route.ts` — 15-step ML pipeline, Canopy GraphQL, KV caching
- `lib/types.ts` — `AnalyzeApiResponse`, `ReviewScore`, `VaderScore`, `RobertaScore`

Existing Canopy query:
```
{ amazonProduct(input:{asin:"${normalizedAsin}",domain:US}){ topReviews { id body rating } } }
```

Existing `AnalyzeApiResponse`:
```ts
{ reviews: ReviewScore[]; count: number; asin: string; }
```

---

## Task 1: URL → ASIN Auto-Extraction

**File:** `app/page.tsx` only  
**Goal:** When user pastes an Amazon product URL, auto-extract the ASIN.

### Spec

The `<Input>` onChange handler currently does `setAsin(e.target.value)`. Change it to:

1. Check if the input value matches an Amazon URL pattern
2. If yes: extract the ASIN and set that (not the URL)
3. If no: set the value as-is (existing behavior)

Extraction regex: `/(?:\/dp\/|\/product\/)([A-Z0-9]{10})/i`

Example: `https://www.amazon.com/dp/B000E7L2R4/ref=abc123` → sets `B000E7L2R4`

**No other changes.** Existing ASIN validation, submit logic, and animations untouched.

### Verify
- Paste `https://www.amazon.com/dp/B000E7L2R4` → input shows `B000E7L2R4`
- Paste `https://www.amazon.com/Some-Product/dp/B000E7L2R4/ref=sr_1_1` → input shows `B000E7L2R4`
- Type `B000E7L2R4` directly → unchanged behavior

---

## Task 2: Product Title Display

**Files:** `lib/types.ts`, `app/api/analyze/route.ts`, `app/page.tsx`  
**Goal:** Show the Amazon product title above results when available.

### Spec

#### `lib/types.ts`
Add optional field to `AnalyzeApiResponse`:
```ts
export interface AnalyzeApiResponse {
  reviews: ReviewScore[];
  count: number;
  asin: string;
  productTitle?: string;  // add this
}
```

#### `app/api/analyze/route.ts`

1. Change Canopy GraphQL query to also fetch `title`:
   ```
   { amazonProduct(input:{asin:"${normalizedAsin}",domain:US}){ title topReviews { id body rating } } }
   ```

2. After parsing `raw` from `responseJson?.data?.amazonProduct?.topReviews`, also capture:
   ```ts
   const productTitle: string | undefined =
     typeof responseJson?.data?.amazonProduct?.title === 'string'
       ? responseJson.data.amazonProduct.title
       : undefined;
   ```

3. Include `productTitle` in the final return (fresh fetch path only — cache hit path stays unchanged, title will be absent there which is acceptable):
   ```ts
   return NextResponse.json({
     reviews: scored!,
     count: scored!.length,
     asin: normalizedAsin,
     productTitle,
   });
   ```

   Also include it in the cache re-check inside lock (Step 7) return — no, leave that unchanged since we can't reconstruct title from cached `ReviewScore[]`.

#### `app/page.tsx`

In the results section, if `data.productTitle` exists, show it as a single line above the plot.

- Element: `<p>` or `<h3>` styled with existing dark-theme classes
- Style: subtle, e.g. `text-green-400 text-sm font-mono opacity-80` to match the accent
- Placement: between the "Analyzed N reviews for ASIN" line and the `<SentimentPlot>`
- Keep it one line; truncate with `truncate` class if too long

### Verify
- Fresh analyze of B000E7L2R4 shows product title above the chart
- If title is undefined (cache hit), results still render without error
- No TypeScript errors

---

## Done Criteria

- [ ] Both tasks committed on `feat/day5-ux`
- [ ] `npm run build` passes (no TS errors)
- [ ] Spec reviewer approves each task
- [ ] Code quality reviewer approves each task
