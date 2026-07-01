"""
src/rerank.py
─────────────────────────────────────────────────────
OFFLINE STEP — Run this once, before your final submission.
Do NOT run this during the ranking step (violates compute constraints).

WHAT IT DOES:
  1. Reads top-500 shortlist from artifacts/top_500_shortlist.json
     (produced by rank.py --top 500 --shortlist-only mode, or
      you can use the top 500 rows of any intermediate CSV)
  2. For each of the 500 candidates, calls Claude API with their
     full profile + the JD
  3. Claude returns a rich, specific 1-2 sentence reasoning
  4. Saves all 500 reasonings to artifacts/reasoning_cache.json
     (rank.py reads this file at runtime — zero API calls during grading)

HOW TO RUN:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python src/rerank.py --candidates data/candidates.jsonl \
                       --shortlist artifacts/top_500_ids.json \
                       --out artifacts/reasoning_cache.json

COST ESTIMATE:
  500 candidates × ~800 tokens input + ~150 tokens output ≈ $0.50 USD
  Run time: ~20-30 minutes (rate-limit friendly pacing built in)

RATE LIMITING:
  Default: 3 requests/second with exponential backoff on errors.
  Adjust REQUESTS_PER_SECOND below if you hit rate limits.
─────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
import requests
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUESTS_PER_SECOND = 2  # conservative — claude.ai rate limits vary

# ── Full JD text for context ───────────────────────────────────────────────────
JD_SUMMARY = """
Role: Senior AI Engineer — Founding Team at Redrob AI (Series A startup).
Must-have: Production embeddings-based retrieval (BGE, E5, sentence-transformers),
vector databases (Milvus, Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch),
hybrid search, Python, ranking evaluation (NDCG, MRR, MAP).
Nice-to-have: LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, open-source contributions.
Hard disqualifiers: consulting-only career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini),
computer vision / speech / robotics primary expertise without NLP overlap,
pure research without production deployment.
Location: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR, Bengaluru OK.
Experience: 5-9 years, product companies strongly preferred.
Behavioral: Active on platform (recent login), quick responder, short notice period.
"""


def candidate_to_profile_text(candidate: dict) -> str:
    """Formats a candidate dict into a compact profile text for the API prompt."""
    p = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills_list = [f"{s['name']} ({s.get('proficiency','?')}, {s.get('duration_months',0)}mo)"
                   for s in candidate.get("skills", [])[:12]]

    career_lines = []
    for job in candidate.get("career_history", []):
        career_lines.append(
            f"  - {job.get('title')} at {job.get('company')} "
            f"({job.get('duration_months',0)} months): {job.get('description','')[:200]}"
        )

    return f"""
CANDIDATE {candidate['candidate_id']}
Title: {p.get('current_title')} at {p.get('current_company')} ({p.get('current_company_size')})
Experience: {p.get('years_of_experience')} years
Location: {p.get('location')}, {p.get('country')}
Headline: {p.get('headline')}
Summary: {p.get('summary','')[:400]}

Career History:
{''.join(career_lines[:5])}

Skills: {', '.join(skills_list)}

Behavioral Signals:
  last_active: {signals.get('last_active_date')}
  open_to_work: {signals.get('open_to_work_flag')}
  recruiter_response_rate: {signals.get('recruiter_response_rate')}
  notice_period_days: {signals.get('notice_period_days')}
  github_activity_score: {signals.get('github_activity_score')}
  interview_completion_rate: {signals.get('interview_completion_rate')}
""".strip()


RERANK_PROMPT_TEMPLATE = """
You are a senior AI/ML recruiter evaluating a candidate for this role:

{jd_summary}

Here is the candidate profile:
{profile_text}

Write a 1-2 sentence reasoning for why this candidate is or isn't a good fit.

REQUIREMENTS:
- Be specific: mention their actual title, company, years, or named skills from their profile
- Be honest: if they have gaps, mention them
- Connect to the JD: reference what the JD needs vs what they have
- Do NOT hallucinate: only mention skills/experience that appear in the profile above
- Length: 1-2 sentences maximum, ideally under 200 characters
- Do NOT start with "This candidate" or "The candidate"

Return ONLY the reasoning string, no JSON, no labels, no quotes.
"""


def call_claude_api(prompt: str, api_key: str, max_retries: int = 3) -> Optional[str]:
    """
    Calls Claude API and returns the text response.
    Handles rate limiting with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                return "".join(
                    block["text"] for block in data["content"]
                    if block["type"] == "text"
                ).strip()

            elif response.status_code == 429:
                wait = (2 ** attempt) * 5
                print(f"  Rate limited. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
                continue

            else:
                print(f"  API error {response.status_code}: {response.text[:200]}")
                return None

        except requests.Timeout:
            print(f"  Timeout on attempt {attempt+1}. Retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None

    return None


def load_shortlist_candidates(candidates_file: str, shortlist_ids: list) -> dict:
    """
    Loads full candidate records for a list of IDs from the large jsonl file.
    Returns {candidate_id: candidate_dict}
    """
    id_set = set(shortlist_ids)
    found = {}
    print(f"Loading {len(id_set)} shortlisted candidates from {candidates_file}...")
    with open(candidates_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in id_set:
                found[c["candidate_id"]] = c
                if len(found) == len(id_set):
                    break
    print(f"Found {len(found)}/{len(id_set)} candidates.\n")
    return found


def main():
    parser = argparse.ArgumentParser(description="Offline LLM re-ranker — generates reasoning cache")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--shortlist",  default="artifacts/top_500_ids.json",
                        help="JSON list of candidate_ids to re-rank")
    parser.add_argument("--out",        default="artifacts/reasoning_cache.json")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Only process first N candidates (for testing)")
    args = parser.parse_args()

    # ── API key ────────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ERROR] Set your API key first:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # ── Load shortlist IDs ─────────────────────────────────────────────────────
    if not os.path.exists(args.shortlist):
        print(f"[ERROR] Shortlist file not found: {args.shortlist}")
        print("Run rank.py first with --top 500 to generate it:")
        print("  python rank.py --candidates data/candidates.jsonl --top 500 --out artifacts/top500.csv")
        print("  Then extract IDs:  python src/extract_ids.py artifacts/top500.csv artifacts/top_500_ids.json")
        sys.exit(1)

    shortlist_ids = json.load(open(args.shortlist))
    if args.limit:
        shortlist_ids = shortlist_ids[:args.limit]

    print(f"Shortlist: {len(shortlist_ids)} candidates to re-rank\n")

    # ── Load existing cache (resume-friendly) ─────────────────────────────────
    cache = {}
    if os.path.exists(args.out):
        cache = json.load(open(args.out))
        print(f"Loaded existing cache: {len(cache)} entries. "
              f"Will skip already-processed candidates.\n")

    # ── Load full candidate records ────────────────────────────────────────────
    candidates_by_id = load_shortlist_candidates(args.candidates, shortlist_ids)

    # ── Re-rank loop ───────────────────────────────────────────────────────────
    to_process = [cid for cid in shortlist_ids if cid not in cache]
    print(f"Processing {len(to_process)} candidates (skipping {len(cache)} cached)...")
    print(f"Estimated time: {len(to_process) / REQUESTS_PER_SECOND / 60:.1f} minutes\n")

    interval = 1.0 / REQUESTS_PER_SECOND

    for i, cid in enumerate(to_process, start=1):
        candidate = candidates_by_id.get(cid)
        if not candidate:
            print(f"  [{i}/{len(to_process)}] {cid} — NOT FOUND in candidates file, skipping")
            continue

        profile_text = candidate_to_profile_text(candidate)
        prompt = RERANK_PROMPT_TEMPLATE.format(
            jd_summary=JD_SUMMARY,
            profile_text=profile_text,
        )

        t0 = time.time()
        reasoning = call_claude_api(prompt, api_key)
        elapsed = time.time() - t0

        if reasoning:
            # Truncate to 300 chars for CSV safety
            reasoning = reasoning.strip()[:300]
            cache[cid] = reasoning
            title = candidate["profile"].get("current_title", "?")
            print(f"  [{i}/{len(to_process)}] {cid} ({title}): {reasoning[:80]}...")
        else:
            print(f"  [{i}/{len(to_process)}] {cid} — API call failed, using fallback")
            cache[cid] = f"Profile reviewed; included based on composite score."

        # Save checkpoint every 25 candidates
        if i % 25 == 0:
            with open(args.out, "w") as f:
                json.dump(cache, f, indent=2)
            print(f"  [checkpoint saved: {len(cache)} entries → {args.out}]")

        # Rate limit
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Final save ─────────────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\nDone. Saved {len(cache)} reasoning entries → {args.out}")
    print("\nNext step: python rank.py --candidates data/candidates.jsonl --out submission.csv")


if __name__ == "__main__":
    main()
