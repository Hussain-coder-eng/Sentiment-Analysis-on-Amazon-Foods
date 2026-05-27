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
