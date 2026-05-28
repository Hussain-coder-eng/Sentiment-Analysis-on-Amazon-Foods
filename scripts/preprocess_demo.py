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
from typing import Optional
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
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
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
def score_hf(text: str, client: InferenceClient) -> Optional[dict]:
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
        raw_text = row.get("Text")
        raw_text = "" if pd.isna(raw_text) else str(raw_text)
        text = strip_html(raw_text)
        if not text:
            skipped.append((i, "empty text after HTML strip"))
            continue

        try:
            rating = int(row.get("Score", 0))
        except (ValueError, TypeError):
            skipped.append((i, "non-numeric rating"))
            continue
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
