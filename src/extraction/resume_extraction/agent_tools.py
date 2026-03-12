"""
Resume Extraction Agent Tools.

Five focused, single-responsibility extraction tools.
Each handles one natural section of a resume.

Design principles:
- Atomic skill tokens only — never sentence-length skill entries
- Evidence-based — only extract what is actually present
- Domain-agnostic — works for any professional background
"""

import logging
from src.utils.llm_client import extract_structured
from src.extraction.resume_extraction.schemas import (
    CandidateIdentity, WorkHistory, SkillsProfile,
    EducationAndCredentials, QualitySignals
)

logger = logging.getLogger(__name__)


# ── Tool 1: Identity ───────────────────────────────────────────────────────────

def tool_extract_identity(resume_text: str) -> CandidateIdentity:
    """
    Extract candidate identity from resume header and summary.
    Always call first — establishes the frame for all other tools.
    """
    system = """
You extract candidate identity information from the top of a resume.

Rules:
- full_name: Exactly as written. null if not found.
- current_title: Their most recent or stated job title. null if absent.
- current_seniority: Infer from title AND from the responsibilities described.
  "Staff", "Principal", "Director" in title → staff/principal/director.
  "Lead X" or "owned end-to-end" → lead/senior.
  "Junior", "Associate", "Graduate" → junior.
  PhD postdocs with no industry → treat as mid unless they have clear seniority signals.
- summary_present: True only if there is a dedicated summary/objective paragraph.
- Extract URLs exactly as written.
- If a field is absent, return null — never invent contact details.
"""
    user = f"Extract identity information from this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, CandidateIdentity)


# ── Tool 2: Work History ───────────────────────────────────────────────────────

def tool_extract_work_history(resume_text: str) -> WorkHistory:
    """
    Extract structured work history: all roles, durations, impact, trajectory.
    The most information-dense tool — handles the experience section.
    """
    system = """
You extract work history from resumes with high precision.

COMPANY TIER:
- tier_1: FAANG + elite peers: Google, Meta, Apple, Amazon, Microsoft, Netflix, OpenAI,
  Anthropic, DeepMind, Cohere, Stripe, Palantir, Databricks, Snowflake, Goldman Sachs (top banks),
  McKinsey/BCG/Bain (MBB), top law firms (Cravath, Skadden), top hospitals (Mayo, Cleveland Clinic).
- tier_2: Strong known companies, funded unicorns ($1B+ valuation), well-known regional firms.
- tier_3: Mid-market, consulting firms not in MBB, lesser-known companies.
- tier_4: Unknown, very small, or stealth companies.
- academic: Universities, research labs, postdocs, fellowships.

DURATION:
- Compute from date ranges. Sum non-overlapping periods for total_years_experience.
- "Present" = assume up to today. Use 0.5 for stints under 6 months or unclear.
- Do NOT trust any self-reported total like "10+ years experience".

ROLES:
- key_responsibilities: Summarize in your own words (1-2 sentences). Do NOT copy bullet points verbatim.
- impact_highlights: Only clearly quantified or objectively significant achievements.
  e.g. "$2M cost savings", "reduced latency by 40%", "led team of 8". Not "improved processes".
- has_quantified_impact: True if ANY bullet has a number/metric/dollar amount.
- skills_demonstrated: Atomic skill tokens (1-4 words) actually used in this role. Evidence-based only.

TRAJECTORY:
- ascending: Clear progression — promotions, title upgrades, increasing scope.
- plateauing: Same level for 4+ years with no meaningful scope increase.
- pivoting: Clear domain/industry/function change.
- lateral: Sideways moves without clear growth signal.

LEADERSHIP:
- has_leadership_experience: Led a team, project, or initiative (not just senior IC work).
- has_people_management: Had direct reports, conducted performance reviews, or hired.
"""
    user = f"Extract complete work history from this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, WorkHistory)


# ── Tool 3: Skills ─────────────────────────────────────────────────────────────

def tool_extract_skills(resume_text: str) -> SkillsProfile:
    """
    Extract skills with recency and depth signals.
    Separates explicit (listed) from implicit (demonstrated) skills.
    """
    system = """
You extract skills from resumes with precision and honesty.

CRITICAL RULES:
1. All skills must be ATOMIC TOKENS: 1-4 words each. Never phrases or sentences.
   ✓ "Python", "RAG", "financial modeling", "Kubernetes", "A/B testing", "IFRS accounting"
   ✗ "experience building scalable ML systems", "strong communication and leadership skills"

2. explicit_skills: ONLY from a dedicated "Skills", "Technical Skills", or "Core Competencies"
   section. Exactly what is listed — no inference, no additions.

3. implicit_skills: Skills clearly demonstrated in work history, projects, or education
   but NOT listed in a skills section. Only include if the resume provides clear evidence.
   IMPORTANT: Include broad domain concepts when the evidence is overwhelming.
   Examples:
   - Someone with 8 years of ML engineering roles → add "machine learning" as implicit
   - Someone with NLP publications and NLP job titles → add "Natural Language Processing" as implicit  
   - Someone who built ETL pipelines in every role → add "ETL", "data processing" as implicit
   - Someone who deployed to AWS/GCP in every role → add "cloud platforms" as implicit
   - Someone with MLOps job titles or infra work → add "MLOps" as implicit
   Do NOT hallucinate skills. But DO include domain-level concepts that are clearly demonstrated
   across multiple roles even if never explicitly listed in a skills section.

4. skill_recency_entries: For each significant skill, provide a {skill, recency} object.
   - "recent": used in a role/project from the last 2 years
   - "established": used 2-5 years ago
   - "dated": last used 5+ years ago
   Base on the actual dates of the jobs/projects, not guesses.
   Example: [{"skill": "Python", "recency": "recent"}, {"skill": "Spark", "recency": "established"}]

5. skill_depth_signals: Skills where the candidate shows genuine mastery.
   Write a short phrase describing the depth evidence.
   e.g. "PyTorch: fine-tuned 13B model on 32 A100s with FSDP"
   Only include 2-5 genuinely strong signals — be selective.

6. Domain knowledge counts as a skill: "options trading", "HIPAA compliance",
   "clinical trial design", "GAAP accounting", "criminal litigation" are all valid skills.

7. Functional skills count: "stakeholder management", "P&L ownership", "product roadmap",
   "performance reviews" are extractable if clearly evidenced.
"""
    user = f"Extract skills with recency and depth signals from this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, SkillsProfile)


# ── Tool 4: Education & Credentials ───────────────────────────────────────────

def tool_extract_education(resume_text: str) -> EducationAndCredentials:
    """
    Extract education, certifications, publications, OSS, patents, talks.
    """
    system = """
You extract education and credentials from resumes.

INSTITUTION TIER:
- "tier_1": Top 20 globally — MIT, Stanford, Harvard, Oxford, Cambridge, ETH Zurich,
  IIT (top campuses), NUS, Caltech, UCL, Imperial, CMU, Princeton, Yale, etc.
- "tier_2": Strong regional universities, well-regarded state schools, top programs
  (even if not globally elite — e.g. UT Austin CS, UWaterloo, UIUC, Purdue).
- "tier_3": Lesser-known or unranked institutions.
- "online_only": Coursera, edX, Udemy, bootcamps. These are certifications, not degrees.

DEGREES:
- highest_degree: Full degree name. e.g. "PhD Computer Science", "MBA", "LLB", "MD".
- List postdocs under highest_degree if they represent the terminal qualification.
- additional_degrees: All other completed degrees in full.

PUBLICATIONS:
- Include paper title and venue if mentioned. e.g. "RAG-Dialogue — ACL 2023".
- Conference abbreviations are fine: ACL, NeurIPS, EMNLP, ICLR, ICML, CVPR, SIGIR.
- Set has_publications = True if ANY publications are present.

OPEN SOURCE:
- Include project name + any signal of adoption (stars, downloads, PyPI).
- Set has_open_source = True if ANY OSS contributions are present.

CERTIFICATIONS:
- Professional certs only: CPA, CFA, PMP, AWS Solutions Architect, GCP Professional ML Engineer,
  Bar License, Medical License, CKA, CISSP, etc.
- Do NOT include online course certificates like "Coursera ML Specialization" as certifications —
  these belong in education notes, not certifications.
"""
    user = f"Extract education and credentials from this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, EducationAndCredentials)


# ── Tool 5: Quality Signals ────────────────────────────────────────────────────

def tool_assess_quality_signals(resume_text: str) -> QualitySignals:
    """
    Assess resume quality meta-signals: exaggeration, specificity, consistency.
    This is an honest analytical assessment, not a judgment of the person.
    """
    system = """
You assess the quality and honesty of claims made in a resume.
This is an analytical task — assess the WRITING and CLAIMS, not the person.

exaggeration_index (0.0–1.0):
- Count vague superlatives used WITHOUT supporting evidence:
  "passionate about", "expert in", "highly skilled", "results-driven",
  "innovative", "strategic thinker", "thought leader", "world-class".
- If these phrases appear WITH evidence → they don't count as exaggeration.
- 0.0 = every claim is backed by specifics. 1.0 = almost all claims are unsupported adjectives.
- Typical range: 0.1–0.4 for solid resumes. >0.6 is a red flag.

specificity_score (0.0–1.0):
- How concrete and detailed are the claims?
- 0.0: "Worked on machine learning projects to improve business outcomes."
- 0.5: "Built recommendation engine that improved CTR by 14%."
- 1.0: "Fine-tuned Llama-2 13B on 50K domain-specific examples using QLoRA on 4×A100s;
        achieved 12% ROUGE improvement over GPT-3.5 at 60% lower inference cost."
- Typical range: 0.3–0.7. Strong senior resumes often score 0.6+.

buzzword_density (0.0–1.0):
- Ratio of trendy-but-shallow terms to substantive content.
- Flag: "AI-powered", "synergy", "leveraged", "cutting-edge", "transformative",
  "next-generation", "disruptive", "holistic", "robust solutions".
- 0.0 = all substance. 1.0 = mostly buzzwords.

quantified_impact_ratio:
- Count ALL bullet points across the entire resume.
- Divide number of bullets containing a metric (%, $, x, users, latency, time saved)
  by total bullet count. Be strict — "improved processes" is NOT quantified.

has_unexplained_gaps:
- True if employment gaps of 6+ months exist with no explanation in the resume.
- Do not flag gaps that are explained (education, parental leave, freelancing noted).

consistency_issues:
- Only flag genuine issues, not minor things.
- Examples to flag: "Title says Director but described responsibilities sound like IC analyst",
  "Two roles with overlapping date ranges without 'concurrent' noted",
  "Claims 10 years experience but dates only add up to 6".
- Be conservative — only flag clear inconsistencies, not speculative concerns.
"""
    user = f"Assess the quality signals of this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, QualitySignals)