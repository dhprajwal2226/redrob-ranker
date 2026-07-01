"""
app.py — Redrob AI Candidate Ranker — Streamlit Sandbox
Run with: streamlit run app.py
"""

import streamlit as st
import json
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

# ── Styles ──
st.markdown("""
<style>
.metric-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
    border-left: 4px solid #4f46e5;
}
.score-high { color: #16a34a; font-weight: 600; }
.score-mid  { color: #d97706; font-weight: 600; }
.score-low  { color: #dc2626; font-weight: 600; }
.tag {
    display: inline-block;
    background: #ede9fe;
    color: #4f46e5;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("🎯 Redrob AI — Intelligent Candidate Ranker")
st.markdown("**Senior AI Engineer** · Ranks candidates by semantic fit, skill match, structural fit, and behavioral signals.")
st.divider()

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ About this System")
    st.markdown("""
**Pipeline:**
1. JD parsed by Claude API → structured requirements
2. Each candidate scored on 4 dimensions
3. Weighted composite score → ranked shortlist
4. Honeypots detected and eliminated

**Score Weights:**
- 🧠 Semantic fit: 35%
- 🛠️ Skill match: 25%
- 🏢 Structural fit: 20%
- 📊 Behavioral signals: 20%

**Disqualifiers:**
- Entire career at consulting firms
- CV/speech/robotics only (no NLP)
- Impossible profiles (honeypots)
""")
    st.divider()
    st.markdown("Built for **Redrob India Data & AI Challenge**")

# ── Load precomputed results ──
PRECOMPUTED = "artifacts/precomputed_scores.csv"
SUBMISSION  = "team_redrob.csv"

def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def color_score(val):
    v = float(val)
    if v >= 0.6:
        return f'<span class="score-high">{v:.3f}</span>'
    elif v >= 0.4:
        return f'<span class="score-mid">{v:.3f}</span>'
    else:
        return f'<span class="score-low">{v:.3f}</span>'

# ── Tab layout ──
tab1, tab2, tab3 = st.tabs(["🏆 Top 100 Ranked", "📊 Score Explorer", "📁 Upload & Rank"])

# ═══ TAB 1 — Top 100 from submission ═══
with tab1:
    st.subheader("Final Ranked Shortlist — Top 100 Candidates")

    if os.path.exists(SUBMISSION):
        rows = load_csv(SUBMISSION)

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
        with col2:
            show_n = st.selectbox("Show top N", [10, 25, 50, 100], index=1)

        filtered = [r for r in rows if float(r["score"]) >= min_score][:show_n]

        st.markdown(f"Showing **{len(filtered)}** candidates")

        for r in filtered:
            with st.expander(
                f"#{r['rank']}  {r['candidate_id']}  —  score: {float(r['score']):.3f}"
            ):
                st.markdown(f"**Reasoning:** {r['reasoning']}")

    else:
        st.warning("team_redrob.csv not found. Run: `python src/generate_submission.py --input artifacts/top100.csv --out team_redrob.csv`")

# ═══ TAB 2 — Score Explorer ═══
with tab2:
    st.subheader("Score Breakdown — Top 200 Candidates")

    if os.path.exists(PRECOMPUTED):
        rows = load_csv(PRECOMPUTED)

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        scores = [float(r["composite_score"]) for r in rows]
        c1.metric("Candidates scored", len(rows))
        c2.metric("Highest score", f"{max(scores):.3f}")
        c3.metric("Average score", f"{sum(scores)/len(scores):.3f}")
        c4.metric("Above 0.6", sum(1 for s in scores if s >= 0.6))

        st.divider()

        # Score breakdown table
        show_n = st.slider("Show top N candidates", 5, len(rows), 20)
        rows_sorted = sorted(rows, key=lambda r: float(r["composite_score"]), reverse=True)

        table_data = []
        for i, r in enumerate(rows_sorted[:show_n]):
            table_data.append({
                "Rank": i + 1,
                "Candidate ID": r["candidate_id"],
                "Title": r.get("current_title", "")[:30],
                "Company": r.get("current_company", "")[:20],
                "Location": r.get("location", ""),
                "YoE": r.get("years_of_experience", ""),
                "Total": float(r["composite_score"]),
                "Semantic": float(r.get("semantic_score", 0)),
                "Skill": float(r.get("skill_score", 0)),
                "Structural": float(r.get("structural_score", 0)),
                "Behavioral": float(r.get("behavioral_score", 0)),
            })

        st.dataframe(
            table_data,
            use_container_width=True,
            column_config={
                "Total": st.column_config.ProgressColumn("Total", min_value=0, max_value=1),
                "Semantic": st.column_config.ProgressColumn("Semantic", min_value=0, max_value=1),
                "Skill": st.column_config.ProgressColumn("Skill", min_value=0, max_value=1),
                "Structural": st.column_config.ProgressColumn("Structural", min_value=0, max_value=1),
                "Behavioral": st.column_config.ProgressColumn("Behavioral", min_value=0, max_value=1),
            }
        )
    else:
        st.warning("precomputed_scores.csv not found in artifacts/. Run precompute_embeddings.py first.")

# ═══ TAB 3 — Upload & Rank ═══
with tab3:
    st.subheader("Upload a Sample JSONL — See Live Ranking")
    st.markdown("Upload a small `.jsonl` file (up to 100 candidates) to see how our system ranks them.")

    uploaded = st.file_uploader("Upload candidates.jsonl (sample)", type=["jsonl", "json"])

    if uploaded:
        from features import (
            compute_skill_score, compute_structural_score,
            is_honeypot, is_disqualified, compute_behavioral_score_fallback
        )
        from jd_parser import load_requirements

        jd = load_requirements("artifacts/jd_requirements.json")

        candidates = []
        for line in uploaded:
            try:
                candidates.append(json.loads(line))
            except Exception:
                pass

        st.info(f"Loaded {len(candidates)} candidates. Scoring (no ML embeddings — rule-based only)...")

        results = []
        for c in candidates:
            if is_honeypot(c):
                continue
            if is_disqualified(c, jd):
                results.append({
                    "candidate_id": c["candidate_id"],
                    "title": c["profile"].get("current_title", ""),
                    "company": c["profile"].get("current_company", ""),
                    "score": 0.0,
                    "status": "❌ Disqualified"
                })
                continue

            skill = compute_skill_score(c["skills"], jd["hard_skills"])
            structural = compute_structural_score(c, jd)
            behavioral = compute_behavioral_score_fallback(c.get("redrob_signals", {}))
            score = 0.4 * skill + 0.35 * structural + 0.25 * behavioral

            results.append({
                "candidate_id": c["candidate_id"],
                "title": c["profile"].get("current_title", ""),
                "company": c["profile"].get("current_company", ""),
                "score": round(score, 4),
                "status": "✅ Viable"
            })

        results.sort(key=lambda r: r["score"], reverse=True)

        st.success(f"Ranked {len(results)} candidates")
        st.dataframe(results, use_container_width=True)
    else:
        st.info("Upload a JSONL file to see live ranking. You can use the first 50 lines of candidates.jsonl as a sample.")
        st.code("""# Create a sample file (run in terminal):
python -c "
import json
with open('data/candidates.jsonl') as f:
    lines = [f.readline() for _ in range(50)]
with open('sample_50.jsonl', 'w') as f:
    f.writelines(lines)
print('Created sample_50.jsonl')
"
""")