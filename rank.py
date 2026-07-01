import argparse
import csv
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=500)
    args = parser.parse_args()

    precomputed_path = "artifacts/precomputed_scores.csv"

    print("=" * 60)
    print("  Redrob Hackathon — Candidate Ranker")
    print(f"  Output:     {args.out}")
    print(f"  Top N:      {args.top}")
    print("=" * 60)

    if not os.path.exists(precomputed_path):
        print(f"\n[ERROR] Missing {precomputed_path}")
        print("Run: python precompute_embeddings.py first")
        sys.exit(1)

    print(f"\nLoading precomputed scores from {precomputed_path}...")
    rows = []
    with open(precomputed_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["composite_score"] = float(row["composite_score"])
            rows.append(row)

    rows.sort(key=lambda r: r["composite_score"], reverse=True)
    top_rows = rows[: args.top]

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None

    fieldnames = list(top_rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"\nWritten {len(top_rows)} candidates → {args.out}")
    print("\nTOP 10 PREVIEW:")
    for i, r in enumerate(top_rows[:10]):
        print(f"  {i+1}. {r['candidate_id']} — {r.get('current_title','')} at "
              f"{r.get('current_company','')} ({r.get('location','')}) "
              f"score={r['composite_score']}")

if __name__ == "__main__":
    main()