"""
rank.py
─────────────────────────────────────────────────────
THE MAIN GRADED COMMAND. Judges run this.

  python rank.py --candidates data/candidates.jsonl --out submission.csv

CONSTRAINTS (from submission_spec.docx — strictly enforced):
  ✓ ≤ 5 minutes wall-clock
  ✓ ≤ 16 GB RAM
  ✓ CPU only — no GPU
  ✗ NO network calls (no API calls during ranking)
  ✗ NO LLM inference during ranking

WHAT IT DOES:
  1. Loads precomputed embeddings from artifacts/  (fast — numpy load)
  2. Computes cosine similarity of all 100K candidates vs JD (vectorized)
  3. Applies rule-based filters: honeypots & disqualified candidates → score=0
  4. Computes composite score (semantic + skill + structural + behavioral)
  5. Applies title mismatch penalty (the keyword-stuffing trap check)
  6. Loads pre-generated reasoning from artifacts/reasoning_cache.json
     (generated offline by src/rerank.py — no API calls here)
  7. Outputs top 100 ranked candidates as a valid CSV

PRECOMPUTE STEP (run once before this, not counted in the 5-min limit):
  python precompute_embeddings.py   → artifacts/candidate_embeddings.npy
  python src/rerank.py              → artifacts/reasoning_cache.json

SCORING WEIGHTS:
  semantic_score    0.35  (meaning similarity — main signal, per JD)
  skill_score       0.20  (required skills coverage)
  structural_score  0.20  (experience, location, company type)
  behavioral_score  0.25  (platform engagement — availability proxy)
─────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
import csv
import numpy as np

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.jd_parser     import load_requirements
from src.features      import (
    compute_skill_score,
    compute_structural_score,
    is_honeypot,
    is_disqualified,
)
from src.behavioral    import compute_behavioral_score


# ── Scoring weights ────────────────────────────────────────────────────────────

W_SEMANTIC    = 0.35
W_SKILL       = 0.20
W_STRUCTURAL  = 0.20
W_BEHAVIORAL  = 0.25

assert abs(W_SEMANTIC + W_SKILL + W_STRUCTURAL + W_BEHAVIORAL - 1.0) < 1e-9, \
    "Weights must sum to 1.0"

# ── Paths ─────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR     = "artifacts"
JD_EMB_PATH       = os.path.join(ARTIFACTS_DIR, "jd_embedding.npy")
CAND_EMB_PATH     = os.path.join(ARTIFACTS_DIR, "candidate_embeddings.npy")
CAND_IDS_PATH     = os.path.join(ARTIFACTS_DIR, "candidate_ids.json")
JD_REQ_PATH       = os.path.join(ARTIFACTS_DIR, "jd_requirements.json")
REASONING_PATH    = os.path.join(ARTIFACTS_DIR, "reasoning_cache.json")

# ── Title trap detection ───────────────────────────────────────────────────────
# The JD explicitly warns: candidate with all AI skills but titled
# "Marketing Manager" is NOT a fit.  We penalise high-skill-score candidates
# whose title has nothing to do with AI/engineering.

NON_TECH_TITLES = {
    "hr manager", "human resources", "marketing manager", "content writer",
    "graphic designer", "accountant", "civil engineer", "mechanical engineer",
    "sales executive", "operations manager", "customer support",
    "business analyst", "project manager",
}

AI_ADJACENT_TITLES = {
    "ml engineer", "ai engineer", "machine learning", "data scientist",
    "data engineer", "nlp", "search engineer", "recommendation",
    "backend engineer", "software engineer", "full stack", "cloud engineer",
    "devops", "analytics engineer", "applied ml", "senior data",
    "senior software", "frontend", "mobile developer", "java developer",
    ".net developer", "qa engineer",
}


def compute_title_mismatch_penalty(candidate: dict) -> float:
    """
    Returns a multiplier 0.0-1.0 to apply to composite score.

    Non-tech title + high skill count = classic keyword-stuffing trap.
    We apply a penalty, not a hard rejection — the semantic score alone
    should already push these down, but this adds an extra guard.

    1.0 = no penalty
    0.3 = strong penalty (non-tech title, high semantic mismatch likely)
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()
    headline = candidate.get("profile", {}).get("headline", "").lower()
    summary = candidate.get("profile", {}).get("summary", "").lower()

    is_non_tech = any(nt in title for nt in NON_TECH_TITLES)
    if not is_non_tech:
        return 1.0

    # If their headline or summary strongly indicate AI work, give partial credit
    # (career changers mid-transition are a real pattern)
    text = f"{headline} {summary}"
    ai_terms = ["embeddings", "vector", "retrieval", "nlp", "llm",
                 "machine learning", "neural", "transformer", "ranking", "search"]
    has_ai_context = sum(1 for t in ai_terms if t in text) >= 3

    if has_ai_context:
        return 0.6   # partial penalty — might be a genuine career changer
    else:
        return 0.25  # strong penalty — this is the keyword-stuffing trap


def build_reasoning(candidate: dict, scores: dict, reasoning_cache: dict) -> str:
    """
    Returns the reasoning string for the submission CSV.

    Priority:
      1. Use Claude-generated reasoning from reasoning_cache if available
      2. Fall back to a rule-based template (specific, not generic)
    """
    cid = candidate["candidate_id"]

    # Use pre-generated LLM reasoning if available
    if cid in reasoning_cache:
        return reasoning_cache[cid]

    # Fallback: template-based but specific to this candidate
    profile  = candidate.get("profile", {})
    title    = profile.get("current_title", "N/A")
    company  = profile.get("current_company", "N/A")
    yoe      = profile.get("years_of_experience", 0)
    location = profile.get("location", "N/A")
    country  = profile.get("country", "")
    skills   = [s.get("name") for s in candidate.get("skills", [])[:4]]

    loc_str = f"{location}, {country}" if country else location
    skill_str = ", ".join(skills) if skills else "various"

    notice   = candidate.get("redrob_signals", {}).get("notice_period_days", "?")
    response = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0)

    sem  = scores.get("semantic", 0)
    sk   = scores.get("skill", 0)
    beh  = scores.get("behavioral", 0)

    if sem >= 0.6 and sk >= 0.3:
        fit = "Strong semantic and skill fit"
    elif sem >= 0.5:
        fit = "Good semantic alignment with JD"
    elif sk >= 0.4:
        fit = "Good skills coverage"
    else:
        fit = "Partial fit"

    return (
        f"{title} at {company}; {yoe:.1f} yrs; {loc_str}. "
        f"{fit} (semantic={sem:.2f}, skills={sk:.2f}). "
        f"Top skills: {skill_str}. "
        f"Notice {notice}d, response rate {response:.0%}."
    )


def load_precomputed_data():
    """
    Loads all precomputed artifacts. These must exist before rank.py is run.
    Fails fast with a clear error message if missing.
    """
    missing = [p for p in [JD_EMB_PATH, CAND_EMB_PATH, CAND_IDS_PATH, JD_REQ_PATH]
               if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] Missing precomputed artifacts:")
        for p in missing:
            print(f"  ✗ {p}")
        print("\nRun the precompute step first:")
        print("  python precompute_embeddings.py")
        print("  python src/rerank.py   (optional, for richer reasoning)")
        sys.exit(1)

    print("Loading precomputed artifacts...", end="", flush=True)
    t0 = time.time()

    jd_emb     = np.load(JD_EMB_PATH)
    cand_embs  = np.load(CAND_EMB_PATH)
    cand_ids   = json.load(open(CAND_IDS_PATH))
    jd_req     = load_requirements(JD_REQ_PATH)

    reasoning_cache = {}
    if os.path.exists(REASONING_PATH):
        reasoning_cache = json.load(open(REASONING_PATH))

    print(f" done in {time.time()-t0:.1f}s")
    print(f"  candidates: {len(cand_ids)}, embedding dim: {cand_embs.shape[1]}")
    print(f"  reasoning cache: {len(reasoning_cache)} entries\n")

    return jd_emb, cand_embs, cand_ids, jd_req, reasoning_cache


def compute_semantic_scores_vectorized(jd_emb: np.ndarray,
                                       cand_embs: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between JD and ALL candidates in one
    matrix operation — this is the fast path.

    Both jd_emb and cand_embs are L2-normalised (done in precompute step
    with normalize_embeddings=True), so dot product = cosine similarity.

    Shape: (N_candidates,) — one score per candidate.
    """
    return cand_embs @ jd_emb  # shape: (100000,)


def rank_candidates(candidates_file: str,
                    jd_emb: np.ndarray,
                    cand_embs: np.ndarray,
                    cand_ids: list,
                    jd_req: dict,
                    reasoning_cache: dict,
                    top_n: int = 100) -> list:
    """
    Core ranking function. Returns a list of dicts, sorted best → worst.

    Each dict has: candidate_id, rank, score, reasoning, _debug
    """

    # ── Step 1: vectorized semantic scores (fast) ────────────────────────────
    print("Computing semantic similarity for all candidates...", end="", flush=True)
    t0 = time.time()
    sem_scores = compute_semantic_scores_vectorized(jd_emb, cand_embs)
    id_to_sem = {cid: float(sem_scores[i]) for i, cid in enumerate(cand_ids)}
    print(f" done in {time.time()-t0:.1f}s")

    # ── Step 2: stream candidates file for rule-based features ───────────────
    print(f"Applying rule-based filters and computing composite scores...")
    t0 = time.time()

    results = []
    n_honeypot = 0
    n_disqualified = 0
    n_processed = 0

    with open(candidates_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c["candidate_id"]
            n_processed += 1

            if n_processed % 10000 == 0:
                print(f"  {n_processed}/100000 processed...", flush=True)

            # Hard filters — these candidates get score 0 and are excluded
            if is_honeypot(c):
                n_honeypot += 1
                continue

            if is_disqualified(c, jd_req):
                n_disqualified += 1
                continue

            # Compute sub-scores
            sem  = id_to_sem.get(cid, 0.0)
            sk   = compute_skill_score(c.get("skills", []), jd_req.get("hard_skills", []))
            st   = compute_structural_score(c, jd_req)
            beh  = compute_behavioral_score(c.get("redrob_signals", {}))

            # Title mismatch penalty (keyword-stuffing trap guard)
            penalty = compute_title_mismatch_penalty(c)

            # Composite score
            composite = (
                W_SEMANTIC   * sem
                + W_SKILL    * sk
                + W_STRUCTURAL * st
                + W_BEHAVIORAL * beh
            ) * penalty

            scores = {"semantic": sem, "skill": sk, "structural": st, "behavioral": beh}
            reasoning = build_reasoning(c, scores, reasoning_cache)

            results.append({
                "candidate_id": cid,
                "score": composite,
                "reasoning": reasoning,
                "_debug": {
                    "sem": round(sem, 3), "sk": round(sk, 3),
                    "st": round(st, 3), "beh": round(beh, 3),
                    "penalty": round(penalty, 2),
                    "title": c["profile"].get("current_title", ""),
                }
            })

    elapsed = time.time() - t0
    print(f"\nProcessed {n_processed} candidates in {elapsed:.1f}s")
    print(f"  Excluded: {n_honeypot} honeypots, {n_disqualified} disqualified")
    print(f"  Eligible: {len(results)} candidates\n")

    # ── Step 3: sort by composite score (desc), tie-break by candidate_id ────
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # ── Step 4: assign ranks and cap to top_n ─────────────────────────────────
    top = results[:top_n]
    for rank_idx, row in enumerate(top, start=1):
        row["rank"] = rank_idx

    return top


def write_csv(results: list, output_path: str):
    """
    Writes the top-100 results to a CSV file matching submission_spec.docx format.

    Required columns (in order): candidate_id, rank, score, reasoning
    Score: non-increasing, 4 decimal places
    Exactly 100 rows of data + 1 header row
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Normalise scores to [0, 1] range with 4dp, ensuring non-increasing
    # The scores are already in a sensible range but we normalise for presentation
    if results:
        max_score = results[0]["score"]
        min_score = results[-1]["score"]
        score_range = max_score - min_score if max_score != min_score else 1.0

        for i, row in enumerate(results):
            # Map to [0.1, 0.999] range, non-increasing guaranteed by sort
            normalised = 0.1 + 0.899 * (row["score"] - min_score) / score_range
            row["_normalised_score"] = round(normalised, 4)

        # Verify non-increasing (defensive check)
        for i in range(len(results) - 1):
            if results[i]["_normalised_score"] < results[i+1]["_normalised_score"]:
                results[i+1]["_normalised_score"] = results[i]["_normalised_score"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in results:
            score_val = row.get("_normalised_score", round(row["score"], 4))
            # Clean reasoning: no newlines, no double quotes that break CSV
            reasoning = row["reasoning"].replace("\n", " ").replace('"', "'").strip()
            writer.writerow([row["candidate_id"], row["rank"], score_val, reasoning])

    print(f"Wrote {len(results)} rows to {output_path}")


def print_top10_debug(results: list):
    """Prints a human-readable summary of the top 10 to help with spot-checking."""
    print("\n── TOP 10 (for spot-checking) ───────────────────────────────────────")
    for row in results[:10]:
        d = row["_debug"]
        print(f"\n  #{row['rank']}  {row['candidate_id']}")
        print(f"      Title:     {d['title']}")
        print(f"      Scores:    sem={d['sem']:.3f}  skill={d['sk']:.3f}  "
              f"struct={d['st']:.3f}  behav={d['beh']:.3f}  "
              f"penalty={d['penalty']:.2f}")
        print(f"      Composite: {row['score']:.4f}")
        print(f"      Reasoning: {row['reasoning'][:100]}...")
    print("\n──────────────────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Hackathon"
    )
    parser.add_argument(
        "--candidates",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl (default: data/candidates.jsonl)"
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)"
    )
    args = parser.parse_args()

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Redrob Hackathon — Candidate Ranker")
    print(f"  Candidates: {args.candidates}")
    print(f"  Output:     {args.out}")
    print(f"  Top N:      {args.top}")
    print(f"{'='*60}\n")

    # Load precomputed artifacts
    jd_emb, cand_embs, cand_ids, jd_req, reasoning_cache = load_precomputed_data()

    # Run ranking
    results = rank_candidates(
        candidates_file=args.candidates,
        jd_emb=jd_emb,
        cand_embs=cand_embs,
        cand_ids=cand_ids,
        jd_req=jd_req,
        reasoning_cache=reasoning_cache,
        top_n=args.top,
    )

    # Print top-10 for spot check
    print_top10_debug(results)

    # Write output CSV
    write_csv(results, args.out)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s")

    if total_time > 300:
        print("[WARNING] Runtime exceeded 5 minutes. Review submission_spec.docx constraints.")
    else:
        print(f"[OK] Within 5-minute limit ({total_time:.0f}s / 300s)")

    print(f"\nNext step: python validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
