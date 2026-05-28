# Session Handoff — Step 0 Complete

**Date:** 2026-05-27  
**Branch:** merged to `main`  
**Status:** Step 0 DONE. 500 reviews scored. demo-data.json committed.

---

## What Was Accomplished

### Design Doc + Plan
- Architecture spec (867 lines, 8 adversarial review passes): `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-27-step0-preprocessing.md`

### Scripts (all merged to main)
| File | Purpose | Status |
|------|---------|--------|
| `requirements-scripts.txt` | Python deps (transformers, torch, vader, bs4, pandas, tqdm) | ✅ |
| `scripts/validate_demo.py` | Schema validator for demo-data.json | ✅ |
| `scripts/preprocess_demo.py` | Preprocessing script | ✅ |
| `public/demo-data.json` | 500 pre-scored reviews, 327KB | ✅ |

---

## Issues Faced + Solutions

### Bug 1: Wrong HF model name
- `cardiffnlp/twitter-roberta-base-sentiment` → returns `LABEL_0/1/2` (not named labels)
- **Fix:** Use `cardiffnlp/twitter-roberta-base-sentiment-latest` → returns `negative/neutral/positive`

### Bug 2: HF Inference API unreachable
- `MaxRetryError` on all calls — network blocks `api-inference.huggingface.co`
- **Fix:** Switched to local `transformers.pipeline()` (torch CPU). 500 reviews in ~17s.

### Bug 3: Python 3.9 incompatible type hint
- `dict | None` union syntax requires Python 3.10+
- **Fix:** `Optional[dict]` from `typing`

### Bug 4: NaN crash on missing Score/Text fields
- `int(float('nan'))` raises `ValueError`; `str(nan or "")` produces literal `"nan"`
- **Fix:** `try/except` around int(), `pd.isna()` guard on text

### Bug 5: Wrong Python env
- Script installed to micromamba (Python 3.9), but torch/transformers only in `~/Coding/venv` (Python 3.11)
- **Fix:** Run with `/Users/hussianaltufayli/Coding/venv/bin/python3`

---

## Run Results

```
Done: 500 scored, 0 skipped
Written: public/demo-data.json (327KB)
VALIDATION PASSED — 500 rows, schema OK
```

---

## What's Next: Day 1

Per design spec build order:

1. Scaffold Next.js 14 App Router + shadcn/ui + Vercel KV
2. `/api/demo` → reads `public/demo-data.json` → returns JSON
3. Frontend: scatter plot from demo data (Plotly.js, `dynamic` import with `ssr: false`)
4. `/api/warmup` → warms HF model on page load (rate-limited 1 req/min per IP via KV)

**Key files to read before Day 1:**
- Design spec: `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md`
- Day 1 build order: design spec § "Build Order" → Day 1 steps 1-4
- Shared types: design spec § "Shared Types" — `ReviewScore`, `VaderScore`, `RobertaScore`
- `/api/demo` response shape: `{ reviews: ReviewScore[], count: number, asin: null }`

**Environment vars needed for Day 1:**
```
CANOPY_API_KEY=xxx        # canopyapi.co dashboard (100 free/month)
KV_REST_API_URL=https://…  # Vercel dashboard → Storage → KV
KV_REST_API_TOKEN=xxx     # auto-injected by Vercel
HF_API_KEY=hf_xxx         # huggingface.co/settings/tokens (optional for warmup)
```

**Note on HF API:** The HF Inference API is blocked on this machine's network. Day 1's `/api/warmup` and Day 2's `/api/analyze` both call HF remotely — verify Vercel's network can reach `api-inference.huggingface.co` before building those routes.
