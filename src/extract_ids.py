"""
src/extract_ids.py
─────────────────────────────────────────────────────
Helper: extracts candidate_ids from a submission CSV into a JSON list.
Used to feed the shortlist into src/rerank.py.

HOW TO RUN:
  # First, generate a top-500 CSV using rank.py:
  python rank.py --candidates data/candidates.jsonl --top 500 --out artifacts/top500.csv

  # Then extract the IDs:
  python src/extract_ids.py artifacts/top500.csv artifacts/top_500_ids.json
─────────────────────────────────────────────────────
"""

import csv
import json
import sys
import os


def extract_ids(csv_path: str, out_path: str):
    ids = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("candidate_id", "").strip()
            if cid:
                ids.append(cid)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ids, f, indent=2)

    print(f"Extracted {len(ids)} IDs from {csv_path} → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/extract_ids.py <input.csv> <output_ids.json>")
        sys.exit(1)
    extract_ids(sys.argv[1], sys.argv[2])
