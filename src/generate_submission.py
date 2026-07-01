"""
src/generate_submission.py
─────────────────────────────────────────────────────
WHAT THIS FILE DOES:

  Takes rank.py's output CSV and produces the final team_xxx.csv
  that exactly matches submission_spec.docx requirements.

  It's a safety layer — rank.py already writes a valid CSV, but this
  script double-checks every rule and fixes common issues before upload.

HOW TO RUN:
  python src/generate_submission.py \
      --input submission.csv \
      --out team_xxx.csv \
      --team-id team_xxx

  Then validate:
  python validate_submission.py team_xxx.csv
  # Must print: "Submission is valid."
─────────────────────────────────────────────────────
"""

import argparse
import csv
import json
import os
import re
import sys


CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


def load_csv(path: str) -> list:
    """Loads a CSV and returns list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def validate_and_fix(rows: list, candidates_file: str = None) -> list:
    """
    Checks all submission rules and fixes what can be auto-fixed.
    Raises ValueError for unfixable violations.

    Rules enforced:
      - Exactly 100 rows
      - candidate_id matches CAND_XXXXXXX pattern
      - All candidate_ids are unique
      - Ranks 1-100 each appear exactly once
      - Score is non-increasing with rank
      - Tie-break: same score → candidate_id ascending
      - reasoning is non-empty (fills a default if missing)
    """
    print(f"Validating {len(rows)} rows...")

    # ── Check count ────────────────────────────────────────────────────────────
    if len(rows) != 100:
        raise ValueError(f"Expected exactly 100 rows, got {len(rows)}")

    # ── Validate candidate_ids ────────────────────────────────────────────────
    seen_ids = set()
    for i, row in enumerate(rows, 1):
        cid = row.get("candidate_id", "").strip()
        if not CANDIDATE_ID_RE.match(cid):
            raise ValueError(f"Row {i}: invalid candidate_id '{cid}'")
        if cid in seen_ids:
            raise ValueError(f"Row {i}: duplicate candidate_id '{cid}'")
        seen_ids.add(cid)

    # ── Sort by score descending, then candidate_id ascending (tie-break) ────
    rows_sorted = sorted(rows,
                         key=lambda r: (-float(r["score"]), r["candidate_id"]))

    # ── Re-assign ranks 1-100 (clean sequential ranks) ───────────────────────
    for rank_idx, row in enumerate(rows_sorted, start=1):
        row["rank"] = rank_idx

    # ── Ensure score is non-increasing (fix float rounding edge cases) ───────
    prev_score = float("inf")
    for row in rows_sorted:
        s = float(row["score"])
        if s > prev_score:
            row["score"] = prev_score
        prev_score = float(row["score"])

    # ── Round scores to 4 decimal places ─────────────────────────────────────
    for row in rows_sorted:
        row["score"] = round(float(row["score"]), 4)

    # ── Fill missing reasoning ────────────────────────────────────────────────
    for row in rows_sorted:
        if not row.get("reasoning", "").strip():
            row["reasoning"] = (
                f"Included at rank {row['rank']} based on composite score; "
                f"profile merits further review."
            )

    # ── Sanitise reasoning (no line breaks, reasonable length) ───────────────
    for row in rows_sorted:
        r = row["reasoning"].replace("\n", " ").replace("\r", " ").strip()
        if len(r) > 400:
            r = r[:397] + "..."
        row["reasoning"] = r

    print(f"  ✓ 100 rows")
    print(f"  ✓ All candidate_ids valid and unique")
    print(f"  ✓ Ranks 1-100 assigned")
    print(f"  ✓ Scores non-increasing")
    print(f"  ✓ Reasoning non-empty")

    return rows_sorted


def write_submission_csv(rows: list, out_path: str):
    """Writes the final submission CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                row["score"],
                row["reasoning"],
            ])
    print(f"\n✓ Saved → {out_path}")


def spot_check(rows: list, n: int = 5):
    """Prints a sample of rows for manual verification."""
    print(f"\n── Spot-check (first {n} and last {n} rows) ─────────────────")
    for row in rows[:n]:
        print(f"  #{row['rank']}  {row['candidate_id']}  score={row['score']}")
        print(f"      {row['reasoning'][:100]}")
    print("  ...")
    for row in rows[-n:]:
        print(f"  #{row['rank']}  {row['candidate_id']}  score={row['score']}")
        print(f"      {row['reasoning'][:100]}")


def main():
    parser = argparse.ArgumentParser(description="Generate final submission CSV")
    parser.add_argument("--input",   default="submission.csv",
                        help="Input CSV from rank.py")
    parser.add_argument("--out",     default="team_xxx.csv",
                        help="Output CSV filename (use your team ID, e.g. team_001.csv)")
    parser.add_argument("--candidates", default="data/candidates.jsonl",
                        help="Path to candidates file (for optional ID verification)")
    args = parser.parse_args()

    print(f"\nGenerating submission CSV")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.out}\n")

    # Load
    rows = load_csv(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    # Validate and fix
    try:
        rows = validate_and_fix(rows)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Fix the issue in rank.py and re-run.")
        sys.exit(1)

    # Spot check
    spot_check(rows)

    # Write
    write_submission_csv(rows, args.out)

    print(f"\nFinal step: python validate_submission.py {args.out}")
    print("Must print: 'Submission is valid.'")


if __name__ == "__main__":
    main()
