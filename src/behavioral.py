"""
behavioral.py
─────────────────────────────────────────────────────
WHAT THIS FILE DOES:

  Converts all 23 Redrob behavioral signals for a candidate into a
  single behavioural_score between 0.0 and 1.0.

  This score acts as a MULTIPLIER on the composite ranking score.
  The JD explicitly says:
    "A perfect-on-paper candidate who hasn't logged in for 6 months
     and has a 5% recruiter response rate is, for hiring purposes,
     not actually available."

  So even a brilliant engineer gets down-weighted if they look unavailable.

HOW TO TEST:
  python src/behavioral.py

SIGNAL GROUPS (weighted contributions to 1.0):
  Availability signals   — 0.40  (are they actually reachable?)
  Engagement signals     — 0.30  (are they active and interested?)
  Profile quality        — 0.20  (have they invested in their profile?)
  Trust signals          — 0.10  (are they verified?)
─────────────────────────────────────────────────────
"""

from datetime import datetime, date


# ── Reference date: the "today" for all recency calculations ──────────────────
# We fix this so the score is deterministic regardless of when you run it.
# Update this if you re-run after the competition closes.
REFERENCE_DATE = date(2026, 7, 1)


def _days_since(date_str: str) -> int:
    """
    Converts a date string like "2026-05-20" into number of days ago.
    Returns 999 if the string is missing or unparseable (treated as very stale).
    """
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except ValueError:
        return 999


def _score_last_active(days_ago: int) -> float:
    """
    Last login recency score.
    The JD says login recency is a strong availability signal.

    Scoring:
      0-14 days  → 1.0  (very recently active)
      15-30 days → 0.85
      31-60 days → 0.65
      61-90 days → 0.45
      91-180 days → 0.20
      180+ days  → 0.05  (treat as unavailable but don't fully exclude)
    """
    if days_ago <= 14:
        return 1.0
    elif days_ago <= 30:
        return 0.85
    elif days_ago <= 60:
        return 0.65
    elif days_ago <= 90:
        return 0.45
    elif days_ago <= 180:
        return 0.20
    else:
        return 0.05


def _score_notice_period(days: int) -> float:
    """
    Notice period score.
    JD says: "Sub-30 days ideal. Can buy out up to 30 days.
               30+ day notice candidates are still in scope but
               the bar gets higher."

    Scoring:
      0 days        → 1.0  (immediately available)
      1-30 days     → 0.90  (buy-out window, ideal range)
      31-60 days    → 0.55  (acceptable but penalised)
      61-90 days    → 0.30  (significant drag)
      90+ days      → 0.10  (very hard to hire quickly)
    """
    if days == 0:
        return 1.0
    elif days <= 30:
        return 0.90
    elif days <= 60:
        return 0.55
    elif days <= 90:
        return 0.30
    else:
        return 0.10


def _score_response_rate(rate: float) -> float:
    """
    Recruiter response rate (0.0 to 1.0).
    If they never reply to recruiters, we can't hire them.

    Scoring: linear but with a strong penalty below 0.2
      >= 0.7  → 1.0
      0.5-0.7 → 0.8
      0.3-0.5 → 0.6
      0.2-0.3 → 0.4
      0.1-0.2 → 0.2
      < 0.1   → 0.05
    """
    if rate >= 0.7:
        return 1.0
    elif rate >= 0.5:
        return 0.8
    elif rate >= 0.3:
        return 0.6
    elif rate >= 0.2:
        return 0.4
    elif rate >= 0.1:
        return 0.2
    else:
        return 0.05


def _score_github(github_score: float) -> float:
    """
    GitHub activity score (-1 to 100).
    -1 means no GitHub linked. For a Senior AI Engineer role at a
    startup that explicitly values open-source contributions, no
    GitHub is a soft negative signal (not a hard disqualifier).

    Scoring:
      -1        → 0.2  (no GitHub — significant but not fatal)
      0-20      → 0.3  (has GitHub but very low activity)
      21-50     → 0.6
      51-75     → 0.8
      76-100    → 1.0
    """
    if github_score < 0:
        return 0.2
    elif github_score <= 20:
        return 0.3
    elif github_score <= 50:
        return 0.6
    elif github_score <= 75:
        return 0.8
    else:
        return 1.0


def _score_interview_completion(rate: float) -> float:
    """
    Interview completion rate (0.0 to 1.0).
    Candidates who ghost interviews cost everyone time.

    Scoring:
      >= 0.9 → 1.0
      >= 0.7 → 0.8
      >= 0.5 → 0.6
      >= 0.3 → 0.4
      < 0.3  → 0.2
    """
    if rate >= 0.9:
        return 1.0
    elif rate >= 0.7:
        return 0.8
    elif rate >= 0.5:
        return 0.6
    elif rate >= 0.3:
        return 0.4
    else:
        return 0.2


def _score_profile_completeness(pct: float) -> float:
    """
    Profile completeness score (0-100).
    A fully filled profile shows investment in the job search.

    Scoring: linear, normalised to 0-1.
    We apply a small bonus above 90 and penalty below 50.
    """
    if pct >= 90:
        return 1.0
    elif pct >= 70:
        return 0.8
    elif pct >= 50:
        return 0.6
    elif pct >= 30:
        return 0.35
    else:
        return 0.15


def compute_behavioral_score(redrob_signals: dict) -> float:
    """
    Main function. Takes the full redrob_signals dict for one candidate
    and returns a single float between 0.0 and 1.0.

    WEIGHT BREAKDOWN (totals 1.0):
      last_active_date        0.25  ← most important: are they even awake?
      recruiter_response_rate 0.20  ← can we reach them?
      notice_period_days      0.15  ← can we hire them quickly?
      open_to_work_flag       0.10  ← have they signalled intent?
      interview_completion    0.10  ← do they show up?
      github_activity_score   0.08  ← open-source signal (startup values this)
      profile_completeness    0.06  ← invested in job search?
      verified_email+phone    0.03  ← trust signal
      linkedin_connected      0.02  ← professional presence
      willing_to_relocate     0.01  ← small bonus for flexibility
    """

    # ── Extract raw values with safe defaults ────────────────────────────────

    last_active_str   = redrob_signals.get("last_active_date", "")
    response_rate     = float(redrob_signals.get("recruiter_response_rate", 0.0))
    notice_days       = int(redrob_signals.get("notice_period_days", 90))
    open_to_work      = bool(redrob_signals.get("open_to_work_flag", False))
    interview_rate    = float(redrob_signals.get("interview_completion_rate", 0.5))
    github_score      = float(redrob_signals.get("github_activity_score", -1))
    profile_pct       = float(redrob_signals.get("profile_completeness_score", 0))
    verified_email    = bool(redrob_signals.get("verified_email", False))
    verified_phone    = bool(redrob_signals.get("verified_phone", False))
    linkedin          = bool(redrob_signals.get("linkedin_connected", False))
    relocate          = bool(redrob_signals.get("willing_to_relocate", False))

    # ── Compute sub-scores ────────────────────────────────────────────────────

    days_since_login  = _days_since(last_active_str)
    s_active          = _score_last_active(days_since_login)
    s_response        = _score_response_rate(response_rate)
    s_notice          = _score_notice_period(notice_days)
    s_open            = 1.0 if open_to_work else 0.4
    s_interview       = _score_interview_completion(interview_rate)
    s_github          = _score_github(github_score)
    s_profile         = _score_profile_completeness(profile_pct)
    s_verified        = (0.5 if verified_email else 0) + (0.5 if verified_phone else 0)
    s_linkedin        = 1.0 if linkedin else 0.3
    s_relocate        = 1.0 if relocate else 0.5

    # ── Weighted sum ──────────────────────────────────────────────────────────

    score = (
        s_active     * 0.25
        + s_response   * 0.20
        + s_notice     * 0.15
        + s_open       * 0.10
        + s_interview  * 0.10
        + s_github     * 0.08
        + s_profile    * 0.06
        + s_verified   * 0.03
        + s_linkedin   * 0.02
        + s_relocate   * 0.01
    )

    return round(min(max(score, 0.0), 1.0), 4)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("Testing behavioral scorer on first 10 candidates...\n")
    print("=" * 65)

    with open("data/candidates.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            c = json.loads(line)
            cid     = c["candidate_id"]
            title   = c["profile"]["current_title"]
            signals = c["redrob_signals"]
            bscore  = compute_behavioral_score(signals)

            days = _days_since(signals.get("last_active_date", ""))
            print(f"{cid}  {title:<35}")
            print(f"  last_active={days}d ago  "
                  f"response_rate={signals.get('recruiter_response_rate'):.2f}  "
                  f"notice={signals.get('notice_period_days')}d  "
                  f"open={signals.get('open_to_work_flag')}")
            print(f"  → behavioral_score: {bscore:.4f}\n")

    print("=" * 65)
    print("\nIf scores vary between candidates and feel intuitive, it's working.")
    print("High score (~0.85+) = recently active, fast responder, open to work.")
    print("Low score (~0.25)   = inactive 6+ months, low response rate, long notice.")
