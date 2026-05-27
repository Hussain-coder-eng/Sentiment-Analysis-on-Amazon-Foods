# Step 0 — Demo Data Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `public/demo-data.json` — a pre-scored array of 500 Amazon food reviews with VADER + RoBERTa sentiment scores and disagreement values, committed to git so the Next.js app can serve demo mode without any external calls.

**Architecture:** Python script reads `Reviews.csv` from Kaggle dataset, strips HTML from review text, scores each review sequentially with VADER (local, sync) and HF Inference API (remote, async-sequential with backoff), computes per-review disagreement, and writes a typed JSON array matching the `ReviewScore` schema defined in the design spec.

**Tech Stack:** Python 3.11+, `vaderSentiment`, `huggingface_hub.InferenceClient` (free tier — no API key required for `cardiffnlp/twitter-roberta-base-sentiment`), `beautifulsoup4`, `pandas`, `tqdm`

**Source CSV:** `/Users/hussianaltufayli/Coding/amazon-data/Reviews.csv` (do NOT copy into project — .gitignore already covers `amazon-data/`)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `scripts/preprocess_demo.py` | Create | Full preprocessing script |
| `scripts/validate_demo.py` | Create | Schema validator for output JSON |
| `public/demo-data.json` | Generated + committed | Static demo dataset |
| `requirements-scripts.txt` | Create | Python deps for scripts only (separate from Next.js) |
| `.gitignore` | Verify | `amazon-data/` already excluded |

---

## Task 1: Python environment setup

**Files:**
- Create: `requirements-scripts.txt`

- [ ] **Step 1: Create requirements file**

```
# requirements-scripts.txt
vaderSentiment==3.3.2
huggingface_hub==0.23.4
beautifulsoup4==4.12.3
pandas==2.2.2
tqdm==4.66.4
```

- [ ] **Step 2: Create and activate venv**

```bash
cd /Users/hussianaltufayli/Downloads/Sentiment-Analysis-on-Amazon-Foods-main
python3 -m venv .venv-scripts
source .venv-scripts/bin/activate
pip install -r requirements-scripts.txt
```

Expected: All packages install without error. `python -c "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; print('ok')"` prints `ok`.

- [ ] **Step 3: Verify CSV is readable**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('/Users/hussianaltufayli/Coding/amazon-data/Reviews.csv', nrows=5)
print(df.columns.tolist())
print(f'Shape: {df.shape}')
"
```

Expected output contains: `['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']`

- [ ] **Step 4: Verify CSV path in .gitignore**

```bash
grep -n "amazon-data" .gitignore
```

Expected: at least one line matching `amazon-data/` or `amazon-data`. If missing:
```bash
echo "amazon-data/" >> .gitignore
git add .gitignore && git commit -m "chore: exclude amazon-data CSV from git"
```

- [ ] **Step 5: Commit requirements file**

```bash
git checkout -b feat/step0-preprocessing
git add requirements-scripts.txt
git commit -m "chore: add Python script dependencies for demo preprocessing"
```

---

## Task 2: Write validation schema test first (TDD)

**Files:**
- Create: `scripts/validate_demo.py`

- [ ] **Step 1: Write the validator script**

```python
#!/usr/bin/env python3
"""
Validates public/demo-data.json against the ReviewScore schema.
Run after preprocess_demo.py to confirm output integrity.
Exits 0 on success, 1 on any failure.
"""
import json
import sys
from pathlib import Path

MIN_ROWS = 100
OUTPUT_PATH = Path(__file__).parent.parent / "public" / "demo-data.json"


def validate(path: Path) -> list[str]:
    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    if not isinstance(data, list):
        return ["Root must be a JSON array"]

    if len(data) < MIN_ROWS:
        errors.append(f"Too few rows: {len(data)} < {MIN_ROWS}")

    for i, row in enumerate(data):
        prefix = f"Row {i}"

        # text
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            errors.append(f"{prefix}: 'text' must be a non-empty string")

        # rating
        r = row.get("rating")
        if not isinstance(r, (int, float)) or not (1 <= r <= 5):
            errors.append(f"{prefix}: 'rating' must be a number in [1, 5], got {r!r}")

        # vader
        vader = row.get("vader", {})
        for field, lo, hi in [
            ("compound", -1, 1),
            ("pos", 0, 1),
            ("neg", 0, 1),
            ("neu", 0, 1),
        ]:
            v = vader.get(field)
            if not isinstance(v, float) or not (lo <= v <= hi):
                errors.append(f"{prefix}: vader.{field}={v!r} not in [{lo}, {hi}]")

        # roberta
        roberta = row.get("roberta", {})
        for field in ("positive", "neutral", "negative"):
            v = roberta.get(field)
            if not isinstance(v, float) or not (0 <= v <= 1):
                errors.append(f"{prefix}: roberta.{field}={v!r} not in [0, 1]")

        # disagreement
        d = row.get("disagreement")
        if not isinstance(d, float) or not (0 <= d <= 2):
            errors.append(f"{prefix}: disagreement={d!r} not in [0, 2]")

    return errors


if __name__ == "__main__":
    errs = validate(OUTPUT_PATH)
    if errs:
        print(f"VALIDATION FAILED — {len(errs)} error(s):")
        for e in errs[:20]:
            print(f"  {e}")
        sys.exit(1)
    data = json.loads(OUTPUT_PATH.read_text())
    print(f"VALIDATION PASSED — {len(data)} rows, schema OK")
    sys.exit(0)
```

- [ ] **Step 2: Run validator against non-existent file — confirm it fails**

```bash
python3 scripts/validate_demo.py
```

Expected: `VALIDATION FAILED — 1 error(s):  File not found: .../public/demo-data.json`

- [ ] **Step 3: Commit validator**

```bash
git add scripts/validate_demo.py
git commit -m "test: add demo-data.json schema validator"
```

---

## Task 3: Write preprocessing script

**Files:**
- Create: `scripts/preprocess_demo.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Preprocesses Amazon Reviews.csv into public/demo-data.json for demo mode.

Usage:
    source .venv-scripts/bin/activate
    python3 scripts/preprocess_demo.py

Environment:
    HF_TOKEN  (optional) — Hugging Face token for higher rate limits.
               Omit to use anonymous free-tier access.

Input:  /Users/hussianaltufayli/Coding/amazon-data/Reviews.csv
Output: public/demo-data.json
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Constants ──────────────────────────────────────────────────────────────────
CSV_PATH = Path("/Users/hussianaltufayli/Coding/amazon-data/Reviews.csv")
OUTPUT_PATH = Path(__file__).parent.parent / "public" / "demo-data.json"
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
MAX_REVIEWS = 500
TEXT_CHAR_CAP = 1000
GAP_SECONDS = 0.2          # 200ms between HF calls
BATCH_LOG_INTERVAL = 10    # log progress every N reviews
RETRY_DELAYS = [1, 2, 4]   # exponential backoff on 503/429 (seconds)


# ── HTML stripping ─────────────────────────────────────────────────────────────
def strip_html(raw: str) -> str:
    """Strip HTML tags and decode entities. Returns plain text, capped."""
    text = BeautifulSoup(raw or "", "html.parser").get_text(separator=" ")
    return text.strip()[:TEXT_CHAR_CAP]


# ── VADER ──────────────────────────────────────────────────────────────────────
def score_vader(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    scores = sia.polarity_scores(text)
    return {
        "compound": float(scores["compound"]),
        "pos":      float(scores["pos"]),
        "neg":      float(scores["neg"]),
        "neu":      float(scores["neu"]),
    }


# ── HF Inference ───────────────────────────────────────────────────────────────
def score_hf(text: str, client: InferenceClient) -> dict | None:
    """
    Score one review via HF Inference API. Returns None on unrecoverable error.
    Labels returned: 'negative', 'neutral', 'positive' (lowercase).
    """
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            results = client.text_classification(text, model=HF_MODEL)
            # results: list of ClassificationOutput with .label and .score
            label_map = {r.label.lower(): float(r.score) for r in results}
            required = {"negative", "neutral", "positive"}
            if not required.issubset(label_map):
                print(f"\n  [WARN] HF missing labels: {label_map.keys()}, skipping")
                return None
            total = sum(label_map[l] for l in required)
            if not (0.95 <= total <= 1.05):
                print(f"\n  [WARN] HF score sum={total:.3f} out of range, skipping")
                return None
            return {
                "positive": label_map["positive"],
                "neutral":  label_map["neutral"],
                "negative": label_map["negative"],
            }
        except Exception as e:
            msg = str(e)
            if attempt < len(RETRY_DELAYS):
                print(f"\n  [RETRY {attempt+1}] HF error: {msg[:80]}")
            else:
                print(f"\n  [SKIP] HF failed after retries: {msg[:80]}")
                return None
    return None


# ── Disagreement ───────────────────────────────────────────────────────────────
def compute_disagreement(vader: dict, roberta: dict) -> float:
    """
    |vader_compound - (roberta_positive - roberta_negative)|
    Range: [0, 2]. Not a calibrated distance — a useful heuristic.
    """
    return abs(vader["compound"] - (roberta["positive"] - roberta["negative"]))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")  # None = anonymous free tier
    client = InferenceClient(token=hf_token)
    sia = SentimentIntensityAnalyzer()

    print(f"Loading {MAX_REVIEWS} reviews from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, nrows=MAX_REVIEWS, usecols=["Id", "Text", "Score"])

    results: list[dict] = []
    skipped: list[tuple[int, str]] = []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Scoring")):
        raw_text = str(row.get("Text") or "")
        text = strip_html(raw_text)
        if not text:
            skipped.append((i, "empty text after HTML strip"))
            continue

        rating = int(row.get("Score", 0))
        if not (1 <= rating <= 5):
            skipped.append((i, f"invalid rating {rating}"))
            continue

        vader = score_vader(text, sia)

        roberta = score_hf(text, client)
        if roberta is None:
            skipped.append((i, "HF scoring failed"))
            continue

        results.append({
            "text":         text,
            "rating":       rating,
            "vader":        vader,
            "roberta":      roberta,
            "disagreement": float(compute_disagreement(vader, roberta)),
        })

        # 200ms gap between HF calls; 1s pause every 10 reviews
        if i < len(df) - 1:
            time.sleep(GAP_SECONDS)
            if (i + 1) % BATCH_LOG_INTERVAL == 0:
                print(f"\n  [{i+1}/{len(df)}] scored {len(results)} ok, {len(skipped)} skipped")
                time.sleep(1.0 - GAP_SECONDS)  # extra 800ms to reach 1s total

    print(f"\nDone: {len(results)} scored, {len(skipped)} skipped")
    if skipped:
        print("Skipped:")
        for idx, reason in skipped:
            print(f"  row {idx}: {reason}")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Written: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create public/ directory**

```bash
mkdir -p public
```

- [ ] **Step 3: Commit script (before running)**

```bash
git add scripts/preprocess_demo.py
git commit -m "feat: add demo data preprocessing script (VADER + HF RoBERTa)"
```

---

## Task 4: Run the preprocessing script

**Files:**
- Generated: `public/demo-data.json`

> **Note:** This task runs locally. Expect ~3-5 minutes for 500 reviews (200ms/review + backoff pauses). If HF returns 429 errors, the backoff retries automatically. If rate limits persist, set `HF_TOKEN` to a free token from huggingface.co/settings/tokens.

- [ ] **Step 1: Run the script**

```bash
source .venv-scripts/bin/activate
python3 scripts/preprocess_demo.py
```

Expected output (approx):
```
Loading 500 reviews from /Users/.../Reviews.csv ...
Scoring:  2%|...|  10/500 [00:02<01:45]
  [10/500] scored 10 ok, 0 skipped
Scoring: 100%|...|500/500 [03:45<00:00]
Done: 487 scored, 13 skipped
Written: public/demo-data.json (189KB)
```

If you see repeated `[RETRY]` lines: HF is rate-limiting. Wait 60s and re-run. Set `HF_TOKEN` env var if problem persists.

- [ ] **Step 2: Run the validator**

```bash
python3 scripts/validate_demo.py
```

Expected: `VALIDATION PASSED — 487 rows, schema OK`

If validation fails, check the error messages. Common issues:
- `roberta.positive not in [0,1]` → HF returned unexpected label names → check `score_hf()` label map
- `Too few rows` → too many skips → check printed skip reasons, likely HF rate limits

- [ ] **Step 3: Sanity-check first 2 rows manually**

```bash
python3 -c "
import json
data = json.load(open('public/demo-data.json'))
print(f'Total rows: {len(data)}')
for r in data[:2]:
    print(f'text[:80]: {r[\"text\"][:80]!r}')
    print(f'rating: {r[\"rating\"]}')
    print(f'vader compound: {r[\"vader\"][\"compound\"]}')
    print(f'roberta positive: {r[\"roberta\"][\"positive\"]}')
    print(f'disagreement: {r[\"disagreement\"]}')
    print()
"
```

Expected: two reviews with non-empty text, rating 1-5, all scores in expected ranges, disagreement ≥ 0.

---

## Task 5: Commit output and merge

**Files:**
- Commit: `public/demo-data.json`

- [ ] **Step 1: Stage and commit demo data**

```bash
git add public/demo-data.json
git commit -m "feat: add pre-scored demo data (500 Amazon food reviews, VADER+RoBERTa)"
```

- [ ] **Step 2: Push branch**

```bash
git push origin feat/step0-preprocessing
```

- [ ] **Step 3: Run superpowers:code-reviewer before merge**

Dispatch `superpowers:code-reviewer` subagent with:
- `BASE_SHA=$(git merge-base main HEAD)`
- `HEAD_SHA=$(git rev-parse HEAD)`
- Context: "Step 0 preprocessing — new Python script + generated JSON output. Check: HTML stripping correctness, HF label parsing, disagreement formula, retry logic, schema validity."

Fix any Critical/Important issues, then:

- [ ] **Step 4: Merge to main**

```bash
git checkout main
git merge --no-ff feat/step0-preprocessing
git push origin main
git branch -d feat/step0-preprocessing
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|---|---|
| Read Reviews.csv, first 500 rows | Task 4 step 1 |
| Strip HTML with html.parser (not regex) | Task 3 — `strip_html()` uses BeautifulSoup4 |
| VADER via `vaderSentiment` | Task 3 — `score_vader()` |
| HF sequential, one call per review | Task 3 — `score_hf()` loop |
| 200ms gap between HF calls | Task 3 — `time.sleep(GAP_SECONDS)` |
| 1s pause every 10 reviews | Task 3 — `BATCH_LOG_INTERVAL` check |
| Exponential backoff on 503: 1s, 2s, 4s | Task 3 — `RETRY_DELAYS = [1, 2, 4]` |
| `disagreement = │vader_compound - (roberta_positive - roberta_negative)│` | Task 3 — `compute_disagreement()` |
| Output schema: `text/rating/vader/roberta/disagreement` | Task 3 — `results.append({...})` |
| Field is `rating` not `stars` | Task 3 — `"rating": rating` |
| `text` capped at 1000 chars | Task 3 — `TEXT_CHAR_CAP = 1000` in `strip_html()` |
| Validate ≥100 rows, all fields finite | Task 2 — `validate_demo.py` |
| HF labels: lowercase `negative/neutral/positive` | Task 3 — `r.label.lower()` |
| Score sum ≈ 1.0 validation | Task 3 — `0.95 <= total <= 1.05` check |
| Commit `public/demo-data.json` (not CSV) | Task 5 |
| `amazon-data/` excluded from git | Task 1 step 4 |

**Gaps found:** None. All spec requirements mapped.

**Placeholder scan:** No TBD, TODO, or vague steps found.

**Type consistency:** `ReviewScore` fields used consistently: `text` (str), `rating` (int), `vader` (dict with compound/pos/neg/neu), `roberta` (dict with positive/neutral/negative), `disagreement` (float). Matches design spec exactly.
