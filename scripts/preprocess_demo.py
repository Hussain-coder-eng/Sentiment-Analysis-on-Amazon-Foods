#!/usr/bin/env python3
"""
Preprocesses Amazon Reviews.csv into public/demo-data.json for demo mode.

Usage:
    /Users/hussianaltufayli/Coding/venv/bin/python3 scripts/preprocess_demo.py

Input:  /Users/hussianaltufayli/Coding/amazon-data/Reviews.csv
Output: public/demo-data.json
"""

import json
from typing import Optional
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Constants ──────────────────────────────────────────────────────────────────
CSV_PATH = Path("/Users/hussianaltufayli/Coding/amazon-data/Reviews.csv")
OUTPUT_PATH = Path(__file__).parent.parent / "public" / "demo-data.json"
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_REVIEWS = 500
TEXT_CHAR_CAP = 1000
BATCH_LOG_INTERVAL = 10    # log progress every N reviews


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


# ── Local RoBERTa scoring ──────────────────────────────────────────────────────
def score_local(text: str, clf) -> Optional[dict]:
    """
    Score one review via local transformers pipeline. Returns None on error.
    Labels: 'negative', 'neutral', 'positive' (from -latest model id2label).
    """
    try:
        # pipeline returns [[{label, score}, ...]] — one inner list per input
        results = clf(text, truncation=True, max_length=512)[0]
        label_map = {r["label"].lower(): float(r["score"]) for r in results}
        required = {"negative", "neutral", "positive"}
        if not required.issubset(label_map):
            print(f"\n  [WARN] missing labels: {list(label_map.keys())}, skipping")
            return None
        total = sum(label_map[l] for l in required)
        if not (0.95 <= total <= 1.05):
            print(f"\n  [WARN] score sum={total:.3f} out of range, skipping")
            return None
        return {
            "positive": label_map["positive"],
            "neutral":  label_map["neutral"],
            "negative": label_map["negative"],
        }
    except Exception as e:
        print(f"\n  [SKIP] scoring failed: {str(e)[:80]}")
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

    print(f"Loading model {HF_MODEL} ...")
    clf = pipeline("text-classification", model=HF_MODEL, top_k=None, device=-1)
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

        roberta = score_local(text, clf)
        if roberta is None:
            skipped.append((i, "local scoring failed"))
            continue

        results.append({
            "text":         text,
            "rating":       rating,
            "vader":        vader,
            "roberta":      roberta,
            "disagreement": float(compute_disagreement(vader, roberta)),
        })

        if (i + 1) % BATCH_LOG_INTERVAL == 0:
            print(f"\n  [{i+1}/{len(df)}] scored {len(results)} ok, {len(skipped)} skipped")

    print(f"\nDone: {len(results)} scored, {len(skipped)} skipped")
    if skipped:
        print("Skipped:")
        for idx, reason in skipped:
            print(f"  row {idx}: {reason}")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Written: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
