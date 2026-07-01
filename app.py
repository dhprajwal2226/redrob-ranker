"""
app.py — Redrob Hackathon Streamlit Sandbox

WHAT THIS IS:
  A hosted web app that lets anyone run the ranker on a small
  sample of candidates and see ranked results. Required by
  submission_spec.docx Section 10.5.

  It does NOT need to handle the full 100K pool — small-sample
  reproducibility is what the organizers are checking.

HOW TO RUN LOCALLY:
  streamlit run app.py
  → Opens browser at http://localhost:8501

HOW TO DEPLOY (free):
  Option 1: streamlit.io/cloud
    → Connect GitHub → select this repo → Deploy app.py

  Option 2: HuggingFace Spaces
    → New Space → Streamlit → upload files

WHAT IT DOES:
  1. User uploads a .jsonl file (≤100 candidates, e.g. sample_candidates.jsonl)
  2. Clicks "Run Ranker"
  3. Sees ranked table with candidate_id, score, title, reasoning
  4. Can download the ranked CSV
"""

import json
import os
import sys
import tempfile
import time
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

# ── Add project root to path for imports ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features   import (
    compute_skill_score, compute_structural_score,
    is_honeypot, is_disqualified
)
from src.behavioral import compute_behavioral_score
from src.jd_parser  import load_requirements

# ── JD requirements (embedded so the app is self-contained) ──────────────────
JD_REQ_PATH = "artifacts/jd_requirements.json"


@st.cache_resource
def get_jd_requirements():
    """Load JD requirements once and cache."""
    if os.path.exists(JD_REQ_PATH):
        return load_requirements(JD_REQ_PATH)
    # Fallback: hardcoded minimal version
    return {
        "hard_skills": ["embeddings", "vector database", "python", "retrieval systems",
                        "ranking evaluation", "hybrid search", "semantic search"],
        "consulting_firms": ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"],
        "wrong_domains": ["computer vision", "speech recognition", "robotics"],
        "location_preferred": ["Pune", "Noida", "Hyderabad", "Mumbai", "Delhi NCR", "Bengaluru"],
        "experience_min_years": 5,
        "experience_max_years": 9,
    }


@st.cache_resource
def get_embedding_model():
    """Load sentence-transformer model once."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("BAAI/bge-small-en-v1.5")
    except Exception as e:
        st.warning(f"Could not load embedding model: {e}. Semantic score will be 0.")
        return None


def get_candidate_text(candidate: dict) -> str:
    """Combines candidate fields into a single text for embedding."""
    p = candidate.get("profile", {})
    career_text = " ".join(
        j.get("description", "") for j in candidate.get("career_history", [])
    )
    skills_text = " ".join(s.get("name", "") for s in candidate.get("skills", []))
    return (
        f"{p.get('current_title', '')} at {p.get('current_company', '')}. "
        f"{p.get('headline', '')}. {p.get('summary', '')}. "
        f"{career_text}. Skills: {skills_text}"
    )


JD_TEXT_FOR_EMBEDDING = """
Senior AI Engineer Founding Team Redrob AI Series A.
Production embeddings retrieval sentence-transformers BGE E5 vector databases
Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS hybrid search
Python ranking evaluation NDCG MRR MAP. LLM fine-tuning LoRA QLoRA PEFT.
Product company experience. Pune Noida India.
"""

W_SEMANTIC   = 0.35
W_SKILL      = 0.20
W_STRUCTURAL = 0.20
W_BEHAVIORAL = 0.25

NON_TECH_TITLES = {
    "hr manager", "marketing manager", "content writer", "graphic designer",
    "accountant", "civil engineer", "mechanical engineer", "sales executive",
    "operations manager", "customer support", "business analyst", "project manager",
}


def score_candidate(candidate: dict, jd_req: dict, model, jd_embedding) -> dict:
    """Score a single candidate and return a result dict."""
    if is_honeypot(candidate):
        return None
    if is_disqualified(candidate, jd_req):
        return None

    cand_text = get_candidate_text(candidate)

    if model is not None and jd_embedding is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        cand_emb = model.encode(cand_text, normalize_embeddings=True)
        sem = float(cosine_similarity([cand_emb], [jd_embedding])[0][0])
    else:
        sem = 0.0

    sk  = compute_skill_score(candidate.get("skills", []), jd_req.get("hard_skills", []))
    st_ = compute_structural_score(candidate, jd_req)
    beh = compute_behavioral_score(candidate.get("redrob_signals", {}))

    title = candidate.get("profile", {}).get("current_title", "").lower()
    is_non_tech = any(nt in title for nt in NON_TECH_TITLES)
    penalty = 0.25 if is_non_tech else 1.0

    composite = (W_SEMANTIC * sem + W_SKILL * sk + W_STRUCTURAL * st_ + W_BEHAVIORAL * beh) * penalty

    p = candidate.get("profile", {})
    reasoning = (
        f"{p.get('current_title')} at {p.get('current_company')} "
        f"({p.get('years_of_experience'):.1f}y, {p.get('location')}). "
        f"Semantic={sem:.2f}, skills={sk:.2f}, behavioral={beh:.2f}."
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "title": p.get("current_title", ""),
        "company": p.get("current_company", ""),
        "location": f"{p.get('location', '')}, {p.get('country', '')}",
        "experience": p.get("years_of_experience", 0),
        "composite_score": round(composite, 4),
        "semantic": round(sem, 3),
        "skill": round(sk, 3),
        "structural": round(st_, 3),
        "behavioral": round(beh, 3),
        "penalty": round(penalty, 2),
        "reasoning": reasoning,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎯 Redrob AI — Candidate Ranker")
st.markdown("""
**Hackathon Sandbox** — Upload a sample of candidates (`.jsonl` format, ≤100 per run)
and rank them against the Senior AI Engineer JD.
""")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Role:** Senior AI Engineer — Founding Team  
    **Company:** Redrob AI (Series A)  
    **Location:** Pune / Noida, India  
    **Experience:** 5–9 years  

    ---

    **Scoring Weights:**
    - Semantic (embedding similarity): 35%
    - Skill match: 20%
    - Structural fit: 20%
    - Behavioral signals: 25%

    ---

    **Hard Filters:**
    - Honeypot profiles → excluded
    - Consulting-only career → excluded
    - Wrong domain (CV/speech) → excluded
    - Non-tech title penalty: ×0.25
    """)

    st.header("Sample File")
    st.markdown("""
    Use `sample_candidates.json` from the hackathon bundle, or any
    subset of `candidates.jsonl`.
    """)

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload candidates file (.jsonl or .json)",
    type=["jsonl", "json"],
    help="JSON Lines file: one candidate JSON per line. Or a JSON array."
)

if uploaded_file is not None:
    # Parse uploaded file
    try:
        content = uploaded_file.read().decode("utf-8").strip()
        if content.startswith("["):
            candidates = json.loads(content)
        else:
            candidates = [json.loads(line) for line in content.splitlines() if line.strip()]
        st.success(f"Loaded {len(candidates)} candidates from {uploaded_file.name}")
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        st.stop()

    if len(candidates) > 200:
        st.warning(f"File has {len(candidates)} candidates — showing results for first 200.")
        candidates = candidates[:200]

    # ── Run ranker ────────────────────────────────────────────────────────────
    if st.button("🚀 Run Ranker", type="primary"):
        jd_req = get_jd_requirements()

        with st.spinner("Loading embedding model (first time takes ~20s)..."):
            model = get_embedding_model()

        jd_embedding = None
        if model is not None:
            with st.spinner("Embedding JD..."):
                jd_embedding = model.encode(JD_TEXT_FOR_EMBEDDING, normalize_embeddings=True)

        results = []
        n_honeypot    = 0
        n_disqualified = 0

        progress = st.progress(0, text="Scoring candidates...")
        t0 = time.time()

        for i, candidate in enumerate(candidates):
            result = score_candidate(candidate, jd_req, model, jd_embedding)
            if result is None:
                if is_honeypot(candidate):
                    n_honeypot += 1
                else:
                    n_disqualified += 1
            else:
                results.append(result)
            progress.progress((i + 1) / len(candidates),
                              text=f"Scored {i+1}/{len(candidates)} candidates...")

        elapsed = time.time() - t0
        progress.empty()

        # Sort and rank
        results.sort(key=lambda r: (-r["composite_score"], r["candidate_id"]))
        for rank_idx, row in enumerate(results, start=1):
            row["rank"] = rank_idx

        top_n = min(100, len(results))
        top_results = results[:top_n]

        # ── Summary stats ─────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates processed", len(candidates))
        col2.metric("Eligible", len(results))
        col3.metric("Excluded (honeypot)", n_honeypot)
        col4.metric("Excluded (disqualified)", n_disqualified)

        st.caption(f"Ranked {len(results)} eligible candidates in {elapsed:.1f}s")

        # ── Results table ─────────────────────────────────────────────────────
        st.subheader(f"Top {top_n} Candidates")

        df = pd.DataFrame(top_results)[
            ["rank", "candidate_id", "title", "company", "location",
             "experience", "composite_score", "semantic", "skill",
             "structural", "behavioral", "reasoning"]
        ]

        # Color-code by score
        st.dataframe(
            df.style.background_gradient(subset=["composite_score"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )

        # ── Score distribution chart ──────────────────────────────────────────
        st.subheader("Score Distribution")
        score_df = df[["rank", "composite_score", "semantic", "skill",
                        "structural", "behavioral"]].head(30)
        st.bar_chart(score_df.set_index("rank")[["composite_score"]])

        # ── Download button ────────────────────────────────────────────────────
        import csv as csv_module
        output = StringIO()
        writer = csv_module.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in top_results:
            writer.writerow([
                row["candidate_id"], row["rank"],
                row["composite_score"], row["reasoning"]
            ])

        st.download_button(
            label="⬇️ Download ranked CSV",
            data=output.getvalue(),
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )

        # ── Top 10 detail view ────────────────────────────────────────────────
        st.subheader("Top 10 — Detailed View")
        for row in top_results[:10]:
            with st.expander(
                f"#{row['rank']} — {row['candidate_id']}  |  "
                f"{row['title']} at {row['company']}  |  "
                f"Score: {row['composite_score']:.4f}"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Location:** {row['location']}")
                    st.markdown(f"**Experience:** {row['experience']:.1f} years")
                    st.markdown(f"**Composite:** {row['composite_score']:.4f}")
                with col_b:
                    st.markdown(f"**Semantic:** {row['semantic']:.3f}")
                    st.markdown(f"**Skill match:** {row['skill']:.3f}")
                    st.markdown(f"**Structural:** {row['structural']:.3f}")
                    st.markdown(f"**Behavioral:** {row['behavioral']:.3f}")
                    if row["penalty"] < 1.0:
                        st.warning(f"⚠️ Title mismatch penalty: ×{row['penalty']}")
                st.markdown(f"**Reasoning:** {row['reasoning']}")

else:
    st.info("👆 Upload a `.jsonl` or `.json` candidates file to get started.")
    st.markdown("---")
    st.markdown("**Expected format:** One candidate JSON object per line (JSONL), "
                "matching the `candidate_schema.json` provided in the hackathon bundle.")
