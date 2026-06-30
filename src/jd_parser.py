"""
jd_parser.py
─────────────────────────────────────────────────────
WHAT THIS FILE DOES (read this before anything else):

  The job description (JD) is a long text document written in English.
  A human recruiter reads it and understands:
    - "They need someone who built vector search"
    - "People from Wipro-only careers are disqualified"
    - "5-9 years experience but flexible"

  This file makes Claude do that same reading — once — and saves
  the result as a clean JSON file: artifacts/jd_requirements.json

  Every other file in the project uses that JSON.
  You only run this file ONCE (it calls the Claude API, costs tokens).

HOW TO RUN:
  python src/jd_parser.py

OUTPUT:
  artifacts/jd_requirements.json

─────────────────────────────────────────────────────
"""

import json
import os
import requests

# ─── The full JD text (hardcoded so this file is self-contained) ─────────────
# We hardcode it here because the ranking step has NO network / file access.
# This way everything needed is in the code itself.

JD_TEXT = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid) | Open to relocation from Tier-1 Indian cities
Experience Required: 5–9 years

WHAT YOU ABSOLUTELY NEED:
- Production experience with embeddings-based retrieval systems (sentence-transformers, 
  OpenAI embeddings, BGE, E5, or similar) deployed to real users.
- Production experience with vector databases or hybrid search infrastructure — 
  Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or similar.
- Strong Python. Code quality matters.
- Hands-on experience designing evaluation frameworks for ranking systems — 
  NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

THINGS WE'D LIKE BUT WON'T REJECT YOU FOR:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space

EXPLICIT DISQUALIFIERS (these are hard rejections):
- Spent career in pure research environments without production deployment
- AI experience is only recent (<12 months) LangChain/OpenAI API wrappers 
  with no pre-LLM ML production experience
- Senior engineer who hasn't written production code in last 18 months 
  (moved to architecture-only roles)
- Entire career at consulting firms only: TCS, Infosys, Wipro, Accenture, 
  Cognizant, Capgemini
- Primary expertise is computer vision, speech, or robotics with no NLP/IR exposure
- Work has been entirely on closed-source proprietary systems for 5+ years 
  with no external validation

LOCATION PREFERENCES (in order):
Pune, Noida, Hyderabad, Mumbai, Delhi NCR, Bengaluru — all in India
Outside India: case-by-case, no visa sponsorship

NOTICE PERIOD:
Sub-30 days ideal. Can buy out up to 30 days. 30+ days lowers the bar.

COMPANY TYPE PREFERENCE:
Product companies. Consulting-only background is a hard disqualifier.
Currently at consulting but has prior product company experience = fine.

CULTURE / BEHAVIORAL SIGNALS THAT MATTER:
- Ships code fast, doesn't wait for perfect solutions
- Writes a lot, communicates async
- Plans to stay 3+ years (not a title-chaser)
- Active on job platform (last login recency matters)
"""

# ─── The prompt we send to Claude ─────────────────────────────────────────────
# We tell Claude exactly what JSON structure we want back.
# "Return ONLY a JSON object" — this prevents Claude from adding extra text.

EXTRACTION_PROMPT = f"""
You are a senior technical recruiter. Read this job description carefully and extract 
structured requirements.

Return ONLY a valid JSON object — no explanation, no markdown, no backticks.
Use exactly these keys:

{{
  "role_title": "string — the job title",
  "company_stage": "string — funding stage or company type",
  
  "hard_skills": ["list of must-have technical skills as lowercase strings"],
  
  "soft_signals": ["list of preferred qualities / nice-to-haves as strings"],
  
  "disqualifiers": [
    "list of exact disqualification conditions as clear strings"
  ],
  
  "consulting_firms": ["list of consulting firm names that disqualify if career-only"],
  
  "wrong_domains": ["domains that disqualify if primary expertise with no NLP/IR"],
  
  "location_preferred": ["ordered list of preferred city names"],
  "location_india_only": true,
  
  "experience_min_years": 5,
  "experience_max_years": 9,
  "experience_note": "string — nuance on the range",
  
  "notice_period_ideal_days": 30,
  "notice_period_hard_limit_days": 90,
  
  "company_type_preferred": "string — product/startup/etc",
  
  "behavioral_signals_important": ["list of behavioral signals the JD explicitly mentions"],
  
  "keyword_trap_warning": "string — what the JD says about keyword matching being a trap"
}}

JOB DESCRIPTION:
{JD_TEXT}
"""


def call_claude_api(prompt: str) -> str:
    """
    Calls the Claude API and returns the raw text response.
    
    WHAT IS HAPPENING HERE:
    - We send an HTTP POST request to Anthropic's API
    - The body is JSON with our prompt
    - Claude reads the prompt and returns a response
    - We extract the text from the response
    
    If this fails, it usually means:
    - No internet connection (check your wifi)
    - The API key is not set (see setup instructions below)
    """
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            # NOTE: When running from your terminal, set your API key:
            #   export ANTHROPIC_API_KEY="sk-ant-..."
            # The API key is read from environment variables — never hardcode it
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01"
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        timeout=30  # wait max 30 seconds
    )

    # Check if the request succeeded
    if response.status_code != 200:
        raise Exception(
            f"API call failed with status {response.status_code}.\n"
            f"Response: {response.text}\n"
            f"Make sure ANTHROPIC_API_KEY is set in your terminal:\n"
            f"  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    data = response.json()

    # The response has a "content" list. Each item is a block of text.
    # We join all text blocks into one string.
    return "".join(
        block["text"]
        for block in data["content"]
        if block["type"] == "text"
    )


def parse_jd() -> dict:
    """
    Main function. Calls Claude, parses the JSON response, returns a dict.
    
    WHAT IS A DICT?
    A Python dictionary = key-value pairs, like a JSON object.
    Example: {"hard_skills": ["python", "embeddings"], "experience_min_years": 5}
    
    We save this dict to a JSON file so other scripts can load it without
    calling the API again.
    """
    print("Calling Claude API to parse the job description...")
    print("(This runs once. Output is saved so future runs don't call the API.)\n")

    raw_response = call_claude_api(EXTRACTION_PROMPT)

    # Claude returns a JSON string. We convert it to a Python dict.
    # json.loads() = "load string" = parse JSON text into Python object
    try:
        requirements = json.loads(raw_response)
    except json.JSONDecodeError:
        # Sometimes Claude adds a tiny bit of text before/after the JSON.
        # We try to extract just the JSON part.
        import re
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            requirements = json.loads(json_match.group())
        else:
            raise Exception(
                f"Could not parse Claude's response as JSON.\n"
                f"Raw response was:\n{raw_response}"
            )

    return requirements


def save_requirements(requirements: dict, output_path: str = "artifacts/jd_requirements.json"):
    """
    Saves the parsed requirements dict to a JSON file.
    indent=2 makes the file human-readable (nicely formatted).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(requirements, f, indent=2)

    print(f"Saved to {output_path}")


def load_requirements(path: str = "artifacts/jd_requirements.json") -> dict:
    """
    Loads the saved requirements JSON.
    All other scripts call this instead of calling the API again.
    
    Usage in other files:
        from src.jd_parser import load_requirements
        jd = load_requirements()
        print(jd["hard_skills"])   # ['python', 'embeddings', ...]
    """
    with open(path) as f:
        return json.load(f)


# ─── WHAT HAPPENS WHEN YOU RUN THIS FILE ──────────────────────────────────────
# Python runs the block below only when you execute this file directly.
# It does NOT run when another file does "from src.jd_parser import load_requirements"

if __name__ == "__main__":

    output_path = "artifacts/jd_requirements.json"

    # If the file already exists, skip the API call (save tokens/money)
    if os.path.exists(output_path):
        print(f"Found existing {output_path}. Loading it.")
        print("(Delete this file if you want to re-parse the JD.)\n")
        requirements = load_requirements(output_path)
    else:
        # Parse the JD using Claude API
        requirements = parse_jd()
        save_requirements(requirements, output_path)

    # Print a summary so you can verify it looks right
    print("\n── EXTRACTED REQUIREMENTS ──────────────────────────────\n")
    print(f"Role:           {requirements.get('role_title')}")
    print(f"Experience:     {requirements.get('experience_min_years')}–{requirements.get('experience_max_years')} years")
    print(f"Location:       {', '.join(requirements.get('location_preferred', []))}")
    print(f"\nHard skills ({len(requirements.get('hard_skills', []))}):")
    for skill in requirements.get("hard_skills", []):
        print(f"  • {skill}")
    print(f"\nDisqualifiers ({len(requirements.get('disqualifiers', []))}):")
    for d in requirements.get("disqualifiers", []):
        print(f"  ✗ {d}")
    print(f"\nConsulting firms (career-only = disqualified):")
    for firm in requirements.get("consulting_firms", []):
        print(f"  ✗ {firm}")
    print(f"\nWrong domains (if primary, no NLP):")
    for domain in requirements.get("wrong_domains", []):
        print(f"  ✗ {domain}")
    print(f"\nBehavioral signals the JD mentions:")
    for sig in requirements.get("behavioral_signals_important", []):
        print(f"  → {sig}")
    print(f"\nKeyword trap warning from JD:")
    print(f"  \"{requirements.get('keyword_trap_warning')}\"")
    print("\n────────────────────────────────────────────────────────")
    print("\nDone. Other scripts will call load_requirements() to use this data.")