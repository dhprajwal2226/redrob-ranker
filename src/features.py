"""
features.py
─────────────────────────────────────────────────────
WHAT THIS FILE DOES (read before anything else):

  This is the heart of the whole project. For every single candidate
  (out of 100,000), this file answers 5 questions:

    1. semantic_score    — does their career STORY match the JD's meaning?
                            (uses an AI embedding model, not keyword search)
    2. skill_score        — what fraction of required skills do they have?
    3. structural_score   — right experience range, location, company type?
    4. is_honeypot         — is this a fake/impossible profile? (auto-reject)
    5. is_disqualified     — does the JD's hard-reject rules apply to them?

  Every function below is independent and testable on its own.
  At the bottom, run this file directly to test on 10 real candidates.

HOW TO TEST:
  python src/features.py

REAL CANDIDATE JSON SHAPE (so the code below makes sense):
  {
    "candidate_id": "CAND_0000001",
    "profile": {
        "headline": "Backend Engineer | SQL, Spark, Cloud",
        "summary": "...",
        "location": "Toronto",
        "country": "Canada",
        "years_of_experience": 6.9,
        "current_title": "Backend Engineer",
        "current_company": "Mindtree",
        "current_company_size": "10001+",
        "current_industry": "IT Services"
    },
    "career_history": [
        {"company": "...", "title": "...", "duration_months": 27,
         "description": "...", "is_current": true}, ...
    ],
    "skills": [
        {"name": "NLP", "proficiency": "advanced", "duration_months": 26}, ...
    ],
    "redrob_signals": {
        "last_active_date": "2026-05-20",
        "open_to_work_flag": true,
        "recruiter_response_rate": 0.34,
        "notice_period_days": 60,
        ...
    }
  }
─────────────────────────────────────────────────────
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — Turning a candidate into plain text (for embedding)
# ═══════════════════════════════════════════════════════════════════

def get_candidate_text(candidate: dict) -> str:
    """
    Combines all the candidate's text into one big string.
    This string gets converted into an embedding (a list of numbers)
    so we can compare its MEANING to the job description's meaning.

    WHY combine everything? Because the JD wants "the full picture" —
    not just the skills list, but their actual career narrative.
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")

    # Join all job descriptions from career history into one string
    career_descriptions = " ".join(
        job.get("description", "") for job in candidate.get("career_history", [])
    )

    # Join all skill names into one string
    skill_names = " ".join(
        skill.get("name", "") for skill in candidate.get("skills", [])
    )

    return f"{headline}. {summary}. {career_descriptions}. Skills: {skill_names}"


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — Skill matching (simple, rule-based, fast)
# ═══════════════════════════════════════════════════════════════════

def compute_skill_score(candidate_skills: list, required_skills: list) -> float:
    """
    Counts what fraction of the JD's required skills this candidate has.

    EXAMPLE:
      required_skills = ["python", "embeddings", "vector database"]
      candidate has: ["Python", "NLP", "Docker"]
      → matches "python" only → score = 1/3 = 0.33

    WHY substring match (not exact match)?
      JD says "vector database", candidate's skill might be "Vector Databases"
      or "Milvus" (a specific vector DB). We check both directions to catch
      specific product names too (handled by SKILL_SYNONYMS below).
    """
    if not required_skills:
        return 0.0

    # Lowercase everything for comparison (avoids "Python" != "python" bugs)
    candidate_skill_names = {s.get("name", "").lower() for s in candidate_skills}
    candidate_skill_blob = " ".join(candidate_skill_names)

    # Some required skills are broad categories. We map them to specific
    # product names a candidate's resume would actually use.
    SKILL_SYNONYMS = {
        "vector database": ["pinecone", "weaviate", "qdrant", "milvus",
                             "opensearch", "elasticsearch", "faiss", "vector database", "vector db"],
        "embeddings": ["embeddings", "sentence-transformers", "bge", "e5",
                       "openai embeddings", "word2vec", "embedding"],
        "retrieval systems": ["retrieval", "rag", "search", "information retrieval"],
        "ranking evaluation": ["ndcg", "mrr", "map", "ranking", "evaluation framework"],
        "hybrid search": ["hybrid search", "bm25"],
        "semantic search": ["semantic search"],
        "information retrieval": ["information retrieval", "ir", "search engine"],
    }

    matches = 0
    for required in required_skills:
        required_lower = required.lower()
        synonyms = SKILL_SYNONYMS.get(required_lower, [required_lower])

        # Check if ANY synonym appears in the candidate's skill list
        if any(syn in candidate_skill_blob for syn in synonyms):
            matches += 1

    return matches / len(required_skills)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — Structural fit (rule-based scoring)
# ═══════════════════════════════════════════════════════════════════

def compute_structural_score(candidate: dict, jd_requirements: dict) -> float:
    """
    Scores 0.0 to 1.0 based on hard facts: years of experience,
    location, and company type. No ML here — just clear rules.

    Score breakdown (adds up to max 1.0):
      0.4 — years of experience in the ideal range
      0.3 — located in a preferred city
      0.3 — NOT at a consulting-only company currently
    """
    score = 0.0
    profile = candidate.get("profile", {})

    # ─ Experience check ─
    yoe = profile.get("years_of_experience", 0)
    min_yoe = jd_requirements.get("experience_min_years", 5)
    max_yoe = jd_requirements.get("experience_max_years", 9)

    if min_yoe <= yoe <= max_yoe:
        score += 0.4
    elif (min_yoe - 1) <= yoe <= (max_yoe + 2):
        # Close to the range — partial credit (JD says range is a guideline)
        score += 0.2

    # ─ Location check ─
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    preferred_cities = [c.lower() for c in jd_requirements.get("location_preferred", [])]

    if any(city in location for city in preferred_cities):
        score += 0.3
    elif country == "india":
        score += 0.15  # in India but not a preferred city — partial credit

    # ─ Company type check ─
    current_company = profile.get("current_company", "").lower()
    consulting_firms = [f.lower() for f in jd_requirements.get("consulting_firms", [])]

    if not any(firm in current_company for firm in consulting_firms):
        score += 0.3

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — Honeypot detection (catch fake/impossible profiles)
# ═══════════════════════════════════════════════════════════════════

def is_honeypot(candidate: dict) -> bool:
    """
    Detects profiles that are logically impossible — these are traps
    planted in the dataset. If our system ranks these highly, we lose
    points (or get disqualified). Returns True if this looks fake.

    CHECKS:
      1. A single job lasting more than 20 years (240 months) — suspicious
         for someone with relatively few total years of experience
      2. Many skills marked "expert" with 0 months of actual usage
         (you can't be an expert in something you've used for 0 months)
      3. Total career history months wildly exceeds stated years_of_experience
    """
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})

    # Check 1: any single job absurdly long
    for job in career_history:
        if job.get("duration_months", 0) > 240:  # over 20 years at ONE job
            return True

    # Check 2: "expert" skills claimed with zero experience using them
    expert_zero_months = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero_months) >= 3:
        return True

    # Check 3: total months worked vastly exceeds claimed years of experience
    total_months_worked = sum(job.get("duration_months", 0) for job in career_history)
    stated_years = profile.get("years_of_experience", 0)
    stated_months = stated_years * 12

    # Allow some overlap (people sometimes have 2 jobs at once), but if total
    # months worked is more than DOUBLE what they claim, it's suspicious
    if stated_months > 0 and total_months_worked > (stated_months * 2.5):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — Hard disqualification (JD's explicit reject rules)
# ═══════════════════════════════════════════════════════════════════

def is_disqualified(candidate: dict, jd_requirements: dict) -> bool:
    """
    Checks the JD's explicit "hard reject" rules. Returns True if the
    candidate should be automatically scored 0 — no matter how good
    their other scores look.

    RULES CHECKED:
      1. Entire career spent only at consulting firms (TCS, Wipro, etc.)
      2. Primary expertise is computer vision / speech / robotics,
         with NO NLP / information-retrieval skill overlap
    """
    career_history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])

    consulting_firms = [f.lower() for f in jd_requirements.get("consulting_firms", [])]
    wrong_domains = [d.lower() for d in jd_requirements.get("wrong_domains", [])]

    # ─ Rule 1: career-only consulting ─
    if career_history:
        all_consulting = all(
            any(firm in job.get("company", "").lower() for firm in consulting_firms)
            for job in career_history
        )
        if all_consulting:
            return True

    # ─ Rule 2: wrong domain with no NLP/IR overlap ─
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    headline_and_summary = f"{headline} {summary}"

    is_wrong_domain_primary = any(domain in headline_and_summary for domain in wrong_domains)

    if is_wrong_domain_primary:
        # Check if they ALSO have NLP/retrieval skills — if so, don't disqualify
        skill_names = " ".join(s.get("name", "").lower() for s in skills)
        nlp_overlap_terms = ["nlp", "embeddings", "retrieval", "ranking",
                              "llm", "transformer", "language model", "search"]
        has_nlp_overlap = any(term in skill_names for term in nlp_overlap_terms)

        if not has_nlp_overlap:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — Semantic similarity (the ML part — embeddings)
# ═══════════════════════════════════════════════════════════════════
#
# NOTE: We load the embedding model only when needed (lazy loading)
# because loading it takes a few seconds and we don't want to slow
# down simple tests that don't need it.

_embedding_model = None  # cached so we only load once


def get_embedding_model():
    """
    Loads the sentence-transformers model once and reuses it.
    This is the AI model that converts text into a list of numbers
    representing its MEANING. Similar meanings = similar numbers.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (BAAI/bge-small-en-v1.5)... this takes ~10 seconds")
        _embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _embedding_model


def compute_semantic_score(candidate_text: str, jd_embedding: np.ndarray) -> float:
    """
    Compares the MEANING of the candidate's career text to the JD's meaning.
    Returns a number from 0 (totally unrelated) to 1 (identical meaning).

    HOW IT WORKS:
      1. Convert candidate_text into an embedding (list of 384 numbers)
      2. Compare it to jd_embedding using cosine similarity
      3. Cosine similarity measures the ANGLE between two vectors —
         small angle = similar meaning, large angle = different meaning
    """
    model = get_embedding_model()
    candidate_embedding = model.encode(candidate_text)

    # Reshape because cosine_similarity expects 2D arrays (lists of vectors)
    similarity = cosine_similarity(
        [candidate_embedding],
        [jd_embedding]
    )[0][0]

    return float(similarity)


def compute_jd_embedding(jd_text: str) -> np.ndarray:
    """
    Embeds the job description text ONCE. This embedding gets reused
    for all 100,000 candidates (we don't want to recompute it each time).
    """
    model = get_embedding_model()
    return model.encode(jd_text)


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — Behavioral score placeholder
# ═══════════════════════════════════════════════════════════════════
# NOTE: Full behavioral scoring lives in src/behavioral.py (teammate's file).
# This is a simple fallback so features.py can be tested standalone
# before behavioral.py exists.

def compute_behavioral_score_fallback(redrob_signals: dict) -> float:
    """
    Simple fallback behavioral score, used only if behavioral.py
    isn't ready yet. Real version (with all 23 signals) is in
    src/behavioral.py — written by your teammate.
    """
    score = 0.5  # neutral baseline

    if redrob_signals.get("open_to_work_flag"):
        score += 0.2

    response_rate = redrob_signals.get("recruiter_response_rate", 0)
    score += response_rate * 0.3  # 0 to 0.3 bonus

    notice_days = redrob_signals.get("notice_period_days", 90)
    if notice_days <= 30:
        score += 0.2
    elif notice_days <= 60:
        score += 0.1

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — Self-test (runs only when you execute this file directly)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import sys
    import os

    # Make sure we can import jd_parser from the same src/ folder
    sys.path.insert(0, os.path.dirname(__file__))
    from jd_parser import load_requirements

    print("Loading JD requirements...")
    jd_requirements = load_requirements("artifacts/jd_requirements.json")

    print("Loading first 10 candidates from data/candidates.jsonl...\n")

    candidates = []
    with open("data/candidates.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            candidates.append(json.loads(line))

    print(f"Testing feature extraction on {len(candidates)} candidates...\n")
    print("=" * 70)

    for candidate in candidates:
        cid = candidate["candidate_id"]
        profile = candidate["profile"]

        skill_score = compute_skill_score(candidate["skills"], jd_requirements["hard_skills"])
        structural_score = compute_structural_score(candidate, jd_requirements)
        honeypot = is_honeypot(candidate)
        disqualified = is_disqualified(candidate, jd_requirements)
        behavioral_fallback = compute_behavioral_score_fallback(candidate["redrob_signals"])

        print(f"\n{cid} — {profile['current_title']} at {profile['current_company']}")
        print(f"  Location: {profile['location']}, {profile['country']}")
        print(f"  Experience: {profile['years_of_experience']} years")
        print(f"  skill_score:        {skill_score:.2f}")
        print(f"  structural_score:   {structural_score:.2f}")
        print(f"  behavioral (fallback): {behavioral_fallback:.2f}")
        print(f"  is_honeypot:        {honeypot}")
        print(f"  is_disqualified:    {disqualified}")

    print("\n" + "=" * 70)
    print("\nNOTE: semantic_score not tested here — it requires loading the")
    print("embedding model which takes longer. That gets tested in rank.py")
    print("(Step 1.3), which is the next file we build.")
    print("\nIf the numbers above look reasonable, features.py is working correctly.")