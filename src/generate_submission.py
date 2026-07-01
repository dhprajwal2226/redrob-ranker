"""
generate_submission.py
──────────────────────
Reads top100.csv and produces the final team_redrob.csv
in the exact format required by the hackathon validator.

HOW TO RUN:
  python src/generate_submission.py --input artifacts/top100.csv --out team_redrob.csv
"""

import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to top100.csv")
    parser.add_argument("--out", required=True, help="Path to write final submission CSV")
    args = parser.parse_args()

    print("Generating submission CSV")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.out}")

    # ── Load input CSV ──
    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {args.input}")

    if len(rows) != 100:
        print(f"[ERROR] Expected exactly 100 rows, got {len(rows)}")
        sys.exit(1)

    # ── Find the score column (handles composite_score or score) ──
    score_col = None
    for col in ["composite_score", "score", "final_score"]:
        if col in rows[0]:
            score_col = col
            break

    if score_col is None:
        print(f"[ERROR] Cannot find score column. Columns are: {list(rows[0].keys())}")
        sys.exit(1)

    print(f"Using score column: '{score_col}'")

    # ── Sort by score descending, tie-break by candidate_id ascending ──
    rows_sorted = sorted(
        rows,
        key=lambda r: (-float(r[score_col]), r["candidate_id"])
    )

    # ── Build final submission rows ──
    output_rows = []
    prev_score = None
    for rank, row in enumerate(rows_sorted, start=1):
        score = round(float(row[score_col]), 4)

        # Ensure score is non-increasing (fix tiny floating point issues)
        if prev_score is not None and score > prev_score:
            score = prev_score
        prev_score = score

        # Generate reasoning from available fields
        title = row.get("current_title", "")
        company = row.get("current_company", "")
        location = row.get("location", "")
        yoe = row.get("years_of_experience", "")
        skill_s = float(row.get("skill_score", 0))
        struct_s = float(row.get("structural_score", 0))
        behav_s = float(row.get("behavioral_score", 0))

        reasoning = (
            f"{yoe} years experience as {title} at {company} ({location}). "
            f"Skill match: {skill_s:.0%}, structural fit: {struct_s:.0%}, "
            f"platform engagement: {behav_s:.0%}."
        )

        output_rows.append({
            "candidate_id": row["candidate_id"],
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
        })

    # ── Write output ──
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nWritten {len(output_rows)} rows → {args.out}")
    print("\nTOP 5 PREVIEW:")
    for r in output_rows[:5]:
        print(f"  #{r['rank']} {r['candidate_id']} score={r['score']} — {r['reasoning'][:80]}...")
    print(f"\nNext: python validate_submission.py {args.out}")


if __name__ == "__main__":
    main()