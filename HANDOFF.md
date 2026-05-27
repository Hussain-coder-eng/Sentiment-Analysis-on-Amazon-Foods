# Session Handoff — Step 0 Preprocessing

**Date:** 2026-05-27  
**Branch:** `feat/step0-preprocessing`  
**Status:** Tasks 1-3 done. Task 4 (run script) blocked — two bugs need fixing first.

---

## What Was Accomplished

### Design Doc
- Full architecture spec (867 lines, 8 adversarial review passes) recovered from `~/.gstack/projects/` and committed to repo.
- **Location:** `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md`

### Implementation Plan
- Step 0 plan written and committed.
- **Location:** `docs/superpowers/plans/2026-05-27-step0-preprocessing.md`

### Scripts Written (all on `feat/step0-preprocessing`)
| File | Purpose | Status |
|------|---------|--------|
| `requirements-scripts.txt` | Python deps for preprocessing | ✅ Done |
| `scripts/validate_demo.py` | Schema validator for demo-data.json | ✅ Done |
| `scripts/preprocess_demo.py` | Main preprocessing script | ✅ Done (2 bugs below) |

---

## Issues Faced + Solutions

### Bug 1 (Critical, FIXED in script): Wrong HF model name
- **Problem:** Design doc used `cardiffnlp/twitter-roberta-base-sentiment` and claimed labels were `negative/neutral/positive`. Confirmed false — that model's `id2label` returns `LABEL_0/LABEL_1/LABEL_2`.
- **Cause:** The design spike (obs 52) incorrectly "verified" label names.
- **Fix applied:** Changed `HF_MODEL` to `cardiffnlp/twitter-roberta-base-sentiment-latest` which correctly returns `negative/neutral/positive`.
- **File:** `scripts/preprocess_demo.py` line 31

### Bug 2 (Important, FIXED in script): NaN crash on Score field
- **Problem:** `int(row.get("Score", 0))` raises `ValueError` if Score cell is NaN (pandas fills missing numeric cells with NaN, not None).
- **Fix applied:** Wrapped in `try/except (ValueError, TypeError)` with skip-and-continue.
- **File:** `scripts/preprocess_demo.py` lines 127-131

### Bug 3 (FIXED in script): NaN text producing literal "nan" string
- **Problem:** `str(row.get("Text") or "")` — NaN is truthy, so `nan or ""` = `nan`, then `str(nan)` = `"nan"` (5-char string, passes empty check, scored as real review).
- **Fix applied:** `pd.isna()` guard before `str()` cast.
- **File:** `scripts/preprocess_demo.py` lines 120-121

---

## Current Blockers (MUST FIX Before Running Script)

### Blocker 1: Python 3.9 incompatible type hint syntax

**Error:**
```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
Line 58: def score_hf(text: str, client: InferenceClient) -> dict | None:
```

**Root cause:** `dict | None` union syntax requires Python 3.10+. User is on Python 3.9 (micromamba).

**Fix:** In `scripts/preprocess_demo.py`, add `from typing import Optional` import and change the type hint:

```python
# Add to imports at top of file:
from typing import Optional

# Line 58 change:
def score_hf(text: str, client: InferenceClient) -> Optional[dict]:
```

### Blocker 2: Packages installed to system Python, not venv

**Symptom:** `pip install` went to micromamba Python 3.9, not `.venv-scripts`. The `source .venv-scripts/bin/activate` command didn't stick across shell sessions in Claude Code terminal.

**Fix for next session:**
```bash
# Verify venv is active before running:
which python3  # must show .venv-scripts/bin/python3
# If not, reactivate:
source .venv-scripts/bin/activate
# Then reinstall to venv:
pip install -r requirements-scripts.txt
```

Alternatively, use the system micromamba Python directly (packages are already installed there):
```bash
/Users/hussianaltufayli/micromamba/bin/python3 scripts/preprocess_demo.py
```

---

## What's Next

### Step 1: Fix Python 3.9 type hint (2 min)

Edit `scripts/preprocess_demo.py`:
1. Add `from typing import Optional` to imports (after `import os`)
2. Change `def score_hf(...) -> dict | None:` to `def score_hf(...) -> Optional[dict]:`
3. Commit on `feat/step0-preprocessing`

### Step 2: Run the script (~5 min)

```bash
# Option A: use system Python (packages already there)
/Users/hussianaltufayli/micromamba/bin/python3 scripts/preprocess_demo.py

# Option B: activate venv properly first
source /Users/hussianaltufayli/Downloads/Sentiment-Analysis-on-Amazon-Foods-main/.venv-scripts/bin/activate
python3 scripts/preprocess_demo.py
```

Expected output: `Done: ~487 scored, ~13 skipped` → `Written: public/demo-data.json (~189KB)`

### Step 3: Validate output

```bash
python3 scripts/validate_demo.py
```

Expected: `VALIDATION PASSED — NNN rows, schema OK`

### Step 4: Commit demo data + merge

```bash
git add public/demo-data.json
git commit -m "feat: add pre-scored demo data (500 Amazon food reviews, VADER+RoBERTa)"
git checkout main
git merge --no-ff feat/step0-preprocessing
git push origin main
git branch -d feat/step0-preprocessing
```

Then start **Day 1**: Next.js scaffold.

---

## Key File Locations

| What | Where |
|------|-------|
| Design spec | `docs/superpowers/specs/2026-05-25-amazon-sentiment-web-app-design.md` |
| Implementation plan | `docs/superpowers/plans/2026-05-27-step0-preprocessing.md` |
| Preprocessing script | `scripts/preprocess_demo.py` |
| Schema validator | `scripts/validate_demo.py` |
| Python deps | `requirements-scripts.txt` |
| Source CSV | `/Users/hussianaltufayli/Coding/amazon-data/Reviews.csv` (NOT in repo) |
| Output (pending) | `public/demo-data.json` |

## HF Model Note

The correct model is `cardiffnlp/twitter-roberta-base-sentiment-latest` (NOT the non-`-latest` version). The original design doc had this wrong — the `-latest` suffix is required to get human-readable labels (`negative/neutral/positive`).
