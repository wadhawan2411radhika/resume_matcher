"""
JD Extraction Agent Tools.

Each tool is a focused, single-responsibility LLM call.
The ReAct agent decides which to call and in what order.

Design principles:
- Domain-agnostic: no ML/AI assumptions
- Each tool returns a validated Pydantic sub-schema
- Failures are surfaced as structured errors, not exceptions
- Tools can be re-invoked if the agent detects low-quality output
"""

import logging
from src.utils.llm_client import extract_structured
from src.extraction.jd_extraction.schemas import (
    RoleIdentity, HardRequirements, SoftRequirements,
    RoleCharacter, Compensation
)

logger = logging.getLogger(__name__)


# ── Tool 1: Role Identity ──────────────────────────────────────────────────────

def tool_extract_role_identity(jd_text: str) -> RoleIdentity:
    """
    Extract basic role metadata: title, seniority, location, employment type.

    Agent use: always call this first — establishes the frame for all other extractions.
    """
    system = """
You extract factual role metadata from job descriptions. Be precise and literal.

Rules:
- job_title: Use the exact title from the JD. Do not paraphrase.
- seniority_level: Infer from title AND from responsibility language.
  "Lead", "define strategy", "mentor", "drive roadmap" = senior/lead/staff.
  "Support", "assist", "learn", "under guidance" = junior/intern.
  If unclear, use "unknown".
- company_stage: Infer from context if not stated.
  Mentions of "Series A/B/C", "post-IPO", "Fortune 500", "early stage", "stealth" are signals.
- remote_policy: Extract exactly. "3 days in office" → "hybrid 3 days/week onsite".
- If a field is genuinely absent, return null — do not invent.
"""
    user = f"Extract role identity metadata from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, RoleIdentity)


# ── Tool 2: Hard Requirements ──────────────────────────────────────────────────

def tool_extract_hard_requirements(jd_text: str) -> HardRequirements:
    """
    Extract non-negotiable requirements: required skills, YoE, education, certifications.

    Agent use: call after role identity. This is the highest-stakes extraction.
    """
    system = """
You extract hard (non-negotiable) requirements from job descriptions.

CRITICAL — SKILLS MUST BE ATOMIC TOKENS:
- Each skill must be 1-4 words. Never copy JD sentences into skill lists.
- ✓ Correct: ["Python", "SQL", "financial modeling", "clinical trial design", "Kubernetes", "GAAP accounting"]
- ✗ Wrong: ["experience building scalable systems", "strong communication skills"]
- Extract skills from ALL sections — "requirements", "qualifications", "you will need", "must have".
- For each skill ask: "Is this explicitly stated as required or clearly non-negotiable?"
  If yes → required_skills. If uncertain → leave for soft requirements.
- Include both technical skills (tools, languages, frameworks) AND functional skills
  (e.g. "stakeholder management", "P&L ownership", "clinical documentation") AND domain knowledge
  (e.g. "options trading", "HIPAA compliance", "IFRS accounting").
- required_years_of_experience: Extract the minimum number. If "5-8 years" → 5.0. If not stated → null.
- required_education: Be specific. "BS/MS in CS" not just "degree required".
- required_certifications: Only hard requirements. e.g. CPA, PMP, bar license, medical license.
"""
    user = f"Extract hard (non-negotiable) requirements from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, HardRequirements)


# ── Tool 3: Soft Requirements ──────────────────────────────────────────────────

def tool_extract_soft_requirements(jd_text: str) -> SoftRequirements:
    """
    Extract preferred (nice-to-have) requirements.

    Agent use: call after hard requirements. Differentiates candidates who exceed the bar.
    """
    system = """
You extract soft (preferred, nice-to-have) requirements from job descriptions.

Rules:
- Only include what is explicitly framed as optional: "nice to have", "bonus", "preferred",
  "plus", "ideally", "we'd love if", "familiarity with".
- Skills must be atomic tokens (1-4 words). Same rules as hard requirements.
- preferred_company_background: Signals about where the candidate came from.
  e.g. "startup experience preferred", "Big 4 background a plus", "prior agency experience helpful".
- If the JD has no nice-to-haves section, return empty lists — do not invent.
"""
    user = f"Extract soft (preferred/nice-to-have) requirements from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, SoftRequirements)


# ── Tool 4: Role Character ─────────────────────────────────────────────────────

def tool_extract_role_character(jd_text: str) -> RoleCharacter:
    """
    Extract signals about ownership style, work pace, culture, and autonomy.

    Agent use: call after requirements. Captures what kind of person thrives here.
    """
    system = """
You extract role character signals from job descriptions — what kind of person thrives here.

ownership_style guidelines:
- "executor": given clear tasks, expected to execute well. Language: "implement", "support", "maintain".
- "ic_owner": owns deliverables end-to-end. Language: "own", "drive", "responsible for", "define".
- "tech_lead": sets direction for others. Language: "lead the team", "set standards", "architect".
- "player_coach": leads AND does individual work. Language: "lead a small team while still coding".
- "manager": primarily people management. Language: "hire", "performance reviews", "build the team".

work_style:
- "research": experimentation, exploration, papers. Language: "investigate", "prototype", "publish".
- "execution": build and ship. Language: "deploy", "production", "deliver", "launch".
- "hybrid": mix of both.

pace_signal: Read between the lines.
- "fast-paced", "move fast", "0 to 1", "scrappy" → early-stage intensity
- "scale existing", "mature product", "established processes" → scaling/enterprise pace
- "research-heavy", "explore and experiment" → slower, more deliberate

red_flag_signals: Be honest. These are real signals, not criticisms.
- "wear many hats" → high cognitive load, likely under-resourced
- "results-oriented culture" (no mention of process) → pressure-heavy
- "self-starter" + "minimal guidance" → possibly disorganized management
- "immediate impact expected" → no ramp-up time
- "competitive compensation" without disclosing range → possible underpayment signal
"""
    user = f"Extract role character signals from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, RoleCharacter)


# ── Tool 5: Compensation ───────────────────────────────────────────────────────

def tool_extract_compensation(jd_text: str) -> Compensation:
    """
    Extract compensation and benefits information.

    Agent use: call last — lowest priority but useful signal for candidate fit.
    """
    system = """
You extract compensation and benefits information from job descriptions.

Rules:
- salary_range_min / max: Numbers only, no currency symbols. Annualize if given monthly.
- If only one number is given (e.g. "up to $120K"), set that as max, min = null.
- equity_mentioned: True if stock, options, RSUs, equity are mentioned anywhere.
- benefits_highlights: Only notable or differentiating benefits. Skip generic ones like "health insurance".
  Include: "unlimited PTO", "4-day work week", "remote-first", "$X learning budget", "sabbatical policy".
- compensation_transparency: "full" = exact range given. "partial" = vague ("competitive", "market rate").
  "none" = not mentioned at all.
- If compensation is entirely absent from the JD, return all nulls/empty/false.
"""
    user = f"Extract compensation and benefits information from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, Compensation)
