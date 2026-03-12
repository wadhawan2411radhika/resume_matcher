"""
Domain-agnostic resume extraction schemas.

Designed to work across any candidate background — engineering, finance,
legal, marketing, healthcare, etc. No domain-specific assumptions.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────

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


class CompanyTier(str, Enum):
    TIER_1 = "tier_1"   # FAANG, top AI labs, elite firms
    TIER_2 = "tier_2"   # Strong known companies, funded unicorns
    TIER_3 = "tier_3"   # Mid-market, consulting, lesser-known
    TIER_4 = "tier_4"   # Unknown / very small / stealth
    ACADEMIC = "academic"


class CareerTrajectory(str, Enum):
    ASCENDING = "ascending"     # Clear promotions / increasing scope
    PLATEAUING = "plateauing"   # Same level for 4+ years
    PIVOTING = "pivoting"       # Domain or role change
    LATERAL = "lateral"         # Sideways moves, no clear growth
    UNKNOWN = "unknown"


class SkillRecency(str, Enum):
    RECENT = "recent"           # Used in last 2 years
    ESTABLISHED = "established" # Used 2-5 years ago
    DATED = "dated"             # Last used 5+ years ago


# ── Sub-schemas (assembled by agent tools) ─────────────────────────────────────

class CandidateIdentity(BaseModel):
    """Basic candidate metadata extracted from header/summary."""
    full_name: Optional[str] = None
    current_title: Optional[str] = None
    current_seniority: SeniorityLevel = SeniorityLevel.UNKNOWN
    location: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    personal_site: Optional[str] = None
    summary_present: bool = Field(
        default=False,
        description="Whether the resume has a professional summary section"
    )


class WorkRole(BaseModel):
    """A single job/role entry."""
    company: str
    title: str
    duration_years: float = Field(description="Computed duration. Use 0.5 for <6 months.")
    company_tier: CompanyTier
    domain: Optional[str] = Field(None, description="Industry/domain of the company")
    seniority_at_role: SeniorityLevel = SeniorityLevel.UNKNOWN
    key_responsibilities: list[str] = Field(
        default_factory=list,
        description="1-2 sentence summaries of what they actually did. NOT copied bullets."
    )
    impact_highlights: list[str] = Field(
        default_factory=list,
        description="Quantified or clearly significant achievements only."
    )
    has_quantified_impact: bool = Field(
        description="True if any bullet contains a number, %, $, or measurable metric."
    )
    skills_demonstrated: list[str] = Field(
        default_factory=list,
        description="Atomic skill tokens clearly used in this role."
    )


class WorkHistory(BaseModel):
    """Full work history analysis."""
    roles: list[WorkRole] = Field(default_factory=list)
    total_years_experience: float = Field(
        description=(
            "Computed by summing non-overlapping durations. "
            "Do NOT trust any self-reported claim. "
            "Use 0.5 for short stints without clear dates."
        )
    )
    career_trajectory: CareerTrajectory
    has_leadership_experience: bool = Field(
        description="True if they led a team, project, or people at any point."
    )
    has_people_management: bool = Field(
        description="True if they had direct reports or hiring responsibility."
    )
    highest_company_tier: CompanyTier
    domains_worked_in: list[str] = Field(
        description="All industry domains across career. e.g. ['fintech', 'healthcare', 'e-commerce']"
    )


class SkillRecencyEntry(BaseModel):
    """Single skill + recency pair. List form avoids dict[str,str] which OpenAI Structured Outputs rejects."""
    skill: str = Field(description="Skill name, matching exactly how it appears in explicit_skills or implicit_skills.")
    recency: SkillRecency = Field(description="recent = used last 2yr | established = 2-5yr ago | dated = 5+yr ago")


class SkillsProfile(BaseModel):
    """Comprehensive skills extraction."""
    explicit_skills: list[str] = Field(
        description=(
            "Atomic tokens (1-4 words) from a dedicated Skills section ONLY. "
            "Exactly what is listed — no inference, no hallucination."
        )
    )
    implicit_skills: list[str] = Field(
        description=(
            "Atomic tokens demonstrated in work history or projects "
            "but NOT listed in a skills section. Only include if clearly evidenced."
        )
    )
    skill_recency_entries: list[SkillRecencyEntry] = Field(
        default_factory=list,
        description=(
            "Recency classification for each significant skill. "
            "Cover all explicit_skills and any implicit skills used recently. "
            "Based on the actual dates of roles/projects where the skill was used."
        )
    )
    skill_depth_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Skills where the candidate shows clear depth — not just listed but "
            "demonstrated at scale, in production, or with measurable outcomes. "
            "e.g. 'PyTorch: fine-tuned 13B model on 4xA100s with FSDP'"
        )
    )

    @property
    def skill_recency_map(self) -> dict[str, str]:
        """Backward-compatible dict view of skill_recency_entries."""
        return {e.skill: e.recency.value for e in self.skill_recency_entries}


class EducationAndCredentials(BaseModel):
    """Education, certifications, publications, open source."""
    highest_degree: Optional[str] = Field(None, description="e.g. 'PhD Computer Science', 'MBA', 'BS Mechanical Engineering'")
    institution: Optional[str] = None
    institution_tier: Optional[str] = Field(
        None,
        description="'tier_1' (top 20 global), 'tier_2' (strong regional), 'tier_3' (unknown), 'online_only'"
    )
    graduation_year: Optional[int] = None
    additional_degrees: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(
        default_factory=list,
        description="Professional certifications. e.g. ['CPA', 'AWS Solutions Architect', 'PMP']"
    )
    publications: list[str] = Field(
        default_factory=list,
        description="Published papers, articles, or research. Include venue if available."
    )
    open_source_contributions: list[str] = Field(
        default_factory=list,
        description="OSS projects, GitHub repos with stars, notable contributions."
    )
    patents: list[str] = Field(default_factory=list)
    speaking_engagements: list[str] = Field(
        default_factory=list,
        description="Conference talks, podcasts, notable public appearances."
    )
    has_publications: bool = False
    has_open_source: bool = False


class QualitySignals(BaseModel):
    """
    Meta-signals about resume quality and candidate substance.
    These are NOT personality judgments — they measure claim quality.
    """
    exaggeration_index: float = Field(
        ge=0.0, le=1.0,
        description=(
            "0.0 = all claims concrete and supported by evidence. "
            "1.0 = almost entirely vague superlatives with no proof. "
            "Flag: 'passionate', 'results-driven', 'innovative', 'expert in X' with no evidence."
        )
    )
    specificity_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "0.0 = generic ('worked on ML models', 'improved performance'). "
            "1.0 = highly specific ('fine-tuned Llama-2 13B on 50K examples with FSDP, "
            "12% ROUGE improvement over GPT-3.5 baseline')."
        )
    )
    buzzword_density: float = Field(
        ge=0.0, le=1.0,
        description=(
            "0.0 = all substance. 1.0 = mostly trendy terms with no depth. "
            "Flag: 'AI-powered', 'synergy', 'leveraged', 'cutting-edge', 'world-class'."
        )
    )
    quantified_impact_ratio: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of ALL bullet points across the resume that contain a number or metric."
    )
    has_unexplained_gaps: bool = Field(
        default=False,
        description="True if there are employment gaps of 6+ months with no explanation."
    )
    consistency_issues: list[str] = Field(
        default_factory=list,
        description=(
            "Inconsistencies or concerns found. "
            "e.g. 'Claims senior title but describes junior responsibilities', "
            "'Dates overlap across two concurrent roles without explanation'."
        )
    )


# ── Final assembled profile ────────────────────────────────────────────────────

class ResumeProfile(BaseModel):
    """
    Complete structured profile of a resume.
    Assembled from sub-schemas extracted by the ReAct agent.
    Domain-agnostic — works for any candidate type.
    """
    identity: CandidateIdentity
    work_history: WorkHistory
    skills: SkillsProfile
    education: EducationAndCredentials
    quality_signals: QualitySignals

    # Synthesized by agent
    career_archetype: str = Field(
        description=(
            "One concise label for the candidate's professional identity. "
            "e.g. 'ML Platform Engineer', 'Generalist Builder', 'Researcher-Practitioner', "
            "'Enterprise Sales Leader', 'Full-Stack Product Engineer', 'Clinical Data Scientist'"
        )
    )
    career_narrative: str = Field(
        description="2-3 sentences telling the story of this person's career arc. Be specific."
    )
    green_flags: list[str] = Field(
        default_factory=list,
        description="Genuine standout signals. Be selective — only real differentiators."
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Real concerns: gaps, stagnation, all buzz no substance, "
            "claims without evidence, inconsistencies."
        )
    )
    extraction_confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Agent self-assessed confidence. "
            "1.0 = resume was detailed and clear. "
            "0.5 = some sections sparse or ambiguous. "
            "0.2 = very sparse resume, many fields inferred."
        )
    )
    extraction_notes: list[str] = Field(
        default_factory=list,
        description="Ambiguities, missing sections, or low-confidence extractions to flag."
    )

    # Convenience properties for backward compatibility with scoring layer
    @property
    def candidate_name(self) -> Optional[str]:
        return self.identity.full_name

    @property
    def current_seniority(self) -> SeniorityLevel:
        return self.identity.current_seniority

    @property
    def total_years_experience(self) -> float:
        return self.work_history.total_years_experience

    @property
    def explicit_skills(self) -> list[str]:
        return self.skills.explicit_skills

    @property
    def implicit_skills(self) -> list[str]:
        return self.skills.implicit_skills

    @property
    def domains_worked_in(self) -> list[str]:
        return self.work_history.domains_worked_in

    @property
    def highest_company_tier(self) -> CompanyTier:
        return self.work_history.highest_company_tier

    @property
    def career_trajectory(self) -> CareerTrajectory:
        return self.work_history.career_trajectory

    @property
    def has_leadership_experience(self) -> bool:
        return self.work_history.has_leadership_experience

    @property
    def has_open_source(self) -> bool:
        return self.education.has_open_source

    @property
    def has_publications(self) -> bool:
        return self.education.has_publications

    @property
    def exaggeration_index(self) -> float:
        return self.quality_signals.exaggeration_index

    @property
    def specificity_score(self) -> float:
        return self.quality_signals.specificity_score

    @property
    def buzzword_density(self) -> float:
        return self.quality_signals.buzzword_density

    @property
    def quantified_impact_ratio(self) -> float:
        return self.quality_signals.quantified_impact_ratio

    @property
    def has_unexplained_gaps(self) -> bool:
        return self.quality_signals.has_unexplained_gaps

    @property
    def work_experiences(self) -> list[WorkRole]:
        return self.work_history.roles