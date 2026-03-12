"""
Domain-agnostic JD extraction schemas.

Designed to work across any job domain — engineering, finance, legal,
marketing, healthcare, etc. No ML/AI-specific assumptions baked in.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SeniorityLevel(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    VP = "vp"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"


class OwnershipStyle(str, Enum):
    EXECUTOR = "executor"         # Given tasks, executes them well
    IC_OWNER = "ic_owner"         # Owns deliverables end-to-end independently
    TECH_LEAD = "tech_lead"       # Drives technical/functional direction for a team
    PLAYER_COACH = "player_coach" # Leads a team but still does individual work
    MANAGER = "manager"           # Primarily people management
    UNKNOWN = "unknown"


class WorkStyle(str, Enum):
    RESEARCH = "research"         # Exploration, experimentation, publication
    EXECUTION = "execution"       # Build and ship to production
    HYBRID = "hybrid"             # Mix of both
    UNKNOWN = "unknown"


# ── Sub-schemas (assembled by agent tools) ────────────────────────────────────

class RoleIdentity(BaseModel):
    """Basic factual role metadata."""
    job_title: str
    seniority_level: SeniorityLevel
    department: Optional[str] = None
    team_or_org: Optional[str] = None
    location: Optional[str] = None
    remote_policy: Optional[str] = Field(
        None, description="e.g. 'fully remote', 'hybrid 3 days', 'onsite only'"
    )
    employment_type: Optional[str] = Field(
        None, description="e.g. 'full-time', 'contract', 'part-time'"
    )
    company_stage: Optional[str] = Field(
        None, description="e.g. 'early-stage startup', 'series B', 'public enterprise'"
    )


class HardRequirements(BaseModel):
    """Non-negotiable requirements — failing these is disqualifying."""
    required_skills: list[str] = Field(
        description=(
            "Atomic skill/technology/domain tokens. 1-4 words each. "
            "Only what is explicitly required or clearly non-negotiable. "
            "Examples: 'Python', 'SQL', 'financial modeling', 'stakeholder management', "
            "'Kubernetes', 'GAAP accounting', 'clinical trials', 'Figma'"
        )
    )
    required_years_of_experience: Optional[float] = Field(
        None, description="Minimum years. Use the lower bound if a range is given. Null if not stated."
    )
    required_education: Optional[str] = Field(
        None, description="e.g. \"Bachelor's in CS\", \"CPA required\", \"MD or equivalent\""
    )
    required_domain_experience: list[str] = Field(
        default_factory=list,
        description="Industry or domain experience explicitly required. e.g. ['fintech', 'healthcare', 'B2B SaaS']"
    )
    required_certifications: list[str] = Field(
        default_factory=list,
        description="Mandatory certifications or licenses. e.g. ['CFA', 'PMP', 'AWS Solutions Architect']"
    )


class SoftRequirements(BaseModel):
    """Preferred but not mandatory — differentiators, not gates."""
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Atomic skill tokens from 'nice to have', 'bonus', 'preferred' sections."
    )
    preferred_domain_experience: list[str] = Field(
        default_factory=list,
        description="Domains that would strengthen the application but aren't required."
    )
    preferred_education: Optional[str] = Field(
        None, description="Preferred but not required education level."
    )
    preferred_company_background: list[str] = Field(
        default_factory=list,
        description="e.g. ['startup experience', 'FAANG background', 'Big 4 consulting']"
    )


class RoleCharacter(BaseModel):
    """Signals about what kind of person thrives in this role."""
    ownership_style: OwnershipStyle
    work_style: WorkStyle
    pace_signal: Optional[str] = Field(
        None,
        description="e.g. 'fast-paced 0-to-1', 'scaling existing systems', 'steady enterprise', 'research-heavy'"
    )
    collaboration_surface: Optional[str] = Field(
        None,
        description="Who they work with. e.g. 'cross-functional with PM and design', 'primarily technical team', 'external clients'"
    )
    autonomy_level: Optional[str] = Field(
        None,
        description="e.g. 'high autonomy, define your own roadmap', 'structured with clear requirements', 'collaborative scoping'"
    )
    red_flag_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Implicit warnings embedded in JD language. "
            "e.g. 'wear many hats' = high load, 'results-oriented' with no mention of process = pressure culture, "
            "'immediate impact' = minimal onboarding"
        )
    )
    growth_opportunity: Optional[str] = Field(
        None,
        description="What career growth or learning the role offers, if mentioned."
    )


class Compensation(BaseModel):
    """Compensation and benefits signals, if present."""
    salary_range_min: Optional[float] = Field(None, description="Annual salary minimum in USD (or local currency)")
    salary_range_max: Optional[float] = Field(None, description="Annual salary maximum in USD")
    currency: Optional[str] = Field(None, description="Currency if not USD")
    equity_mentioned: bool = Field(default=False)
    bonus_mentioned: bool = Field(default=False)
    benefits_highlights: list[str] = Field(
        default_factory=list,
        description="Notable benefits mentioned. e.g. ['unlimited PTO', '401k match', 'remote stipend']"
    )
    compensation_transparency: str = Field(
        default="none",
        description="'full' = exact range given, 'partial' = vague range, 'none' = not mentioned"
    )


class JDProfile(BaseModel):
    """
    Complete structured profile of a job description.
    Assembled from sub-schemas extracted by the ReAct agent.
    Domain-agnostic — works for any role type.
    """
    # Sub-schemas
    role_identity: RoleIdentity
    hard_requirements: HardRequirements
    soft_requirements: SoftRequirements
    role_character: RoleCharacter
    compensation: Compensation

    # Synthesized by agent
    ideal_candidate_persona: str = Field(
        description=(
            "2-3 sentences describing what the ideal hire actually looks like. "
            "Specific and concrete — not generic HR language."
        )
    )
    role_in_one_line: str = Field(
        description="One sentence that captures the essence of the role for a candidate. "
                    "e.g. 'Lead the design and scaling of our fraud detection platform as the senior IC owner on a 6-person ML team.'"
    )
    extraction_confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Agent's self-assessed confidence in the extraction quality. "
            "1.0 = JD was detailed and unambiguous. "
            "0.5 = JD had vague language or missing sections. "
            "0.2 = JD was extremely sparse or poorly written."
        )
    )
    extraction_notes: list[str] = Field(
        default_factory=list,
        description="Any caveats, ambiguities, or low-confidence extractions the agent wants to flag."
    )
