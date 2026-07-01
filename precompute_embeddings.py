"""
precompute_embeddings.py  (fast version)
─────────────────────────────────────────
Strategy:
  1. Embed only the JD (instant)
  2. Stream all 100K candidates one by one
  3. SKIP honeypots and disqualified immediately (no embedding)
  4. For remaining candidates: compute skill_score + structural_score
     using fast rule-based functions (no ML, microseconds each)
  5. Only embed the TOP 5000 by rule-based score
     (embedding 5K takes ~3 min vs 100K taking 8+ hours)
  6. Save final scores to artifacts/precomputed_scores.csv

This way rank.py just loads precomputed_scores.csv and sorts — runs in seconds.
"""

import json
import os
import sys
import time
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from jd_parser import load_requirements
from features import (
    get_candidate_text,
    compute_skill_score,
    compute_structural_score,
    is_honeypot,
    is_disqualified,
    compute_behavioral_score_fallback,
)

CANDIDATES_FILE = "data/candidates.jsonl"
OUTPUT_FILE = "artifacts/precomputed_scores.csv"
JD_EMBEDDING_FILE = "artifacts/jd_embedding.npy"

# How many top candidates to actually embed with the ML model
# Rule-based filter cuts 100K → ~5K, then we embed those 5K
EMBED_TOP_N = 5000


def main():
    start = time.time()

    # ── Load JD ──
    print("Loading JD requirements...")
    jd = load_requirements("artifacts/jd_requirements.json")

    # ── Embed JD once ──
    print("Embedding JD text...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    jd_text = (
        f"Senior AI Engineer role requiring: "
        f"{', '.join(jd['hard_skills'])}. "
        f"Product company, India location preferred. "
        f"5-9 years experience in embeddings, retrieval, vector databases, ranking evaluation."
    )
    jd_embedding = model.encode(jd_text)
    np.save(JD_EMBEDDING_FILE, jd_embedding)
    print(f"  JD embedding saved → {JD_EMBEDDING_FILE}")

    # ── Load behavioral scorer ──
    try:
        from behavioral import compute_behavioral_score
        print("Using behavioral.py (full scorer)")
    except Exception:
        compute_behavioral_score = compute_behavioral_score_fallback
        print("Using fallback behavioral scorer")

    # ── PASS 1: Stream all 100K, apply fast filters, compute rule-based scores ──
    print(f"\nPASS 1: Streaming all candidates, fast rule-based scoring...")
    print("(No ML embedding yet — this pass takes ~1-2 minutes)\n")

    rule_based = []   # candidates that pass filters
    total = 0
    skipped_honeypot = 0
    skipped_disqualified = 0

    with open(CANDIDATES_FILE) as f:
        for line in f:
            candidate = json.loads(line.strip())
            total += 1

            if total % 10000 == 0:
                elapsed = time.time() - start
                print(f"  ...{total} processed ({elapsed:.0f}s) — {len(rule_based)} viable so far")

            # Fast filters — no ML needed
            if is_honeypot(candidate):
                skipped_honeypot += 1
                continue

            if is_disqualified(candidate, jd):
                skipped_disqualified += 1
                continue

            # Fast rule-based scores (microseconds each)
            skill = compute_skill_score(candidate["skills"], jd["hard_skills"])
            structural = compute_structural_score(candidate, jd)
            behavioral = compute_behavioral_score(candidate.get("redrob_signals", {}))

            # Quick pre-score without semantic (used to pick top 5K for embedding)
            pre_score = 0.4 * skill + 0.35 * structural + 0.25 * behavioral

            rule_based.append({
                "candidate_id": candidate["candidate_id"],
                "pre_score": pre_score,
                "skill_score": round(skill, 4),
                "structural_score": round(structural, 4),
                "behavioral_score": round(behavioral, 4),
                "current_title": candidate["profile"].get("current_title", ""),
                "current_company": candidate["profile"].get("current_company", ""),
                "location": candidate["profile"].get("location", ""),
                "years_of_experience": candidate["profile"].get("years_of_experience", 0),
                "_text": get_candidate_text(candidate),  # store text for embedding
            })

    elapsed = time.time() - start
    print(f"\nPass 1 done in {elapsed:.0f}s")
    print(f"  Total processed:    {total}")
    print(f"  Honeypots skipped:  {skipped_honeypot}")
    print(f"  Disqualified:       {skipped_disqualified}")
    print(f"  Viable candidates:  {len(rule_based)}")

    # ── Sort by pre_score, take top EMBED_TOP_N ──
    rule_based.sort(key=lambda x: x["pre_score"], reverse=True)
    to_embed = rule_based[:EMBED_TOP_N]

    print(f"\nPASS 2: Embedding top {len(to_embed)} candidates with ML model...")
    print("(This is the only ML step — should take 3-5 minutes)\n")

    # Extract texts for batch encoding
    texts = [c["_text"] for c in to_embed]

    # Encode in batches of 64 — fast enough on CPU
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(texts))
            elapsed = time.time() - start
            print(f"  ...embedded {done}/{len(texts)} ({elapsed:.0f}s elapsed)")

    all_embeddings = np.vstack(all_embeddings)

    # ── Compute final composite scores ──
    print("\nComputing final composite scores...")
    results = []
    for i, candidate in enumerate(to_embed):
        cand_emb = all_embeddings[i]
        semantic = float(
            np.dot(cand_emb, jd_embedding) /
            (np.linalg.norm(cand_emb) * np.linalg.norm(jd_embedding) + 1e-8)
        )

        composite = (
            0.35 * semantic +
            0.25 * candidate["skill_score"] +
            0.20 * candidate["structural_score"] +
            0.20 * candidate["behavioral_score"]
        )

        results.append({
            "candidate_id": candidate["candidate_id"],
            "composite_score": round(composite, 4),
            "semantic_score": round(semantic, 4),
            "skill_score": candidate["skill_score"],
            "structural_score": candidate["structural_score"],
            "behavioral_score": candidate["behavioral_score"],
            "current_title": candidate["current_title"],
            "current_company": candidate["current_company"],
            "location": candidate["location"],
            "years_of_experience": candidate["years_of_experience"],
        })

    # Sort final results
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # ── Save ──
    os.makedirs("artifacts", exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DONE in {total_elapsed:.0f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"Saved {len(results)} scored candidates → {OUTPUT_FILE}")
    print(f"\nTOP 10 PREVIEW:")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. {r['candidate_id']} — {r['current_title']} at "
              f"{r['current_company']} ({r['location']}) score={r['composite_score']}")
    print(f"{'='*60}")
    print(f"\nNext step: python rank.py --candidates data/candidates.jsonl --out artifacts/top500.csv --top 500")


if __name__ == "__main__":
    main()