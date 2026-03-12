"""
ReAct Agent for Resume Extraction.

Mirrors the JD extraction agent architecture:
    Thought → Action (tool call) → Observation → Thought → ...
    until all sections extracted → Synthesize → ResumeProfile

Why ReAct for resumes:
- Resumes vary wildly in structure — some have no skills section, some are
  academic CVs with 20 publications, some are one-page career changers.
- The agent can skip tools that are irrelevant (e.g. no education section)
  and spend more iterations on dense sections.
- Tool failures are isolated — a bad quality signal extraction doesn't
  corrupt the work history.
- Full reasoning trace logged → debuggable.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from openai import OpenAI

from src.extraction.resume_extraction.schemas import (
    ResumeProfile, CandidateIdentity, WorkHistory, SkillsProfile,
    EducationAndCredentials, QualitySignals, SeniorityLevel,
    CareerTrajectory, CompanyTier
)
from src.extraction.resume_extraction.agent_tools import (
    tool_extract_identity,
    tool_extract_work_history,
    tool_extract_skills,
    tool_extract_education,
    tool_assess_quality_signals,
)
from src.utils.llm_client import extract_structured
from config import config

logger = logging.getLogger(__name__)


# ── Agent State ────────────────────────────────────────────────────────────────

@dataclass
class ResumeAgentState:
    """Mutable state carried through the ReAct loop."""
    resume_text: str
    steps: list[dict] = field(default_factory=list)

    # Accumulated tool results
    identity: CandidateIdentity | None = None
    work_history: WorkHistory | None = None
    skills: SkillsProfile | None = None
    education: EducationAndCredentials | None = None
    quality_signals: QualitySignals | None = None

    # Agent control
    iteration: int = 0
    max_iterations: int = 10
    done: bool = False
    error: str | None = None

    def log_step(self, thought: str, action: str, observation: str):
        self.steps.append({
            "iteration": self.iteration,
            "thought": thought,
            "action": action,
            "observation": observation,
        })
        logger.info(f"[Step {self.iteration}] Action: {action}")
        logger.debug(f"  Thought: {thought}")
        logger.debug(f"  Observation: {observation[:200]}")

    def tools_completed(self) -> list[str]:
        completed = []
        if self.identity: completed.append("extract_identity")
        if self.work_history: completed.append("extract_work_history")
        if self.skills: completed.append("extract_skills")
        if self.education: completed.append("extract_education")
        if self.quality_signals: completed.append("assess_quality_signals")
        return completed

    def all_required_complete(self) -> bool:
        """Minimum needed before synthesis: identity + work history + skills."""
        return bool(self.identity and self.work_history and self.skills)

    def all_complete(self) -> bool:
        return bool(
            self.identity and self.work_history and self.skills
            and self.education and self.quality_signals
        )


# ── Tool Registry ──────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "extract_identity": tool_extract_identity,
    "extract_work_history": tool_extract_work_history,
    "extract_skills": tool_extract_skills,
    "extract_education": tool_extract_education,
    "assess_quality_signals": tool_assess_quality_signals,
}

TOOL_DESCRIPTIONS = {
    "extract_identity": "Extract candidate name, current title, seniority, location, and contact info from the resume header/summary.",
    "extract_work_history": "Extract all roles, companies, durations, impact highlights, trajectory, and leadership signals. Most information-dense tool.",
    "extract_skills": "Extract explicit (listed) and implicit (demonstrated) skills as atomic tokens, with recency map and depth signals.",
    "extract_education": "Extract degrees, institution, certifications, publications, open source contributions, patents, speaking engagements.",
    "assess_quality_signals": "Assess exaggeration index, specificity score, buzzword density, quantified impact ratio, and consistency issues.",
    "synthesize": "Finalize extraction — synthesize all gathered sub-schemas into a complete ResumeProfile. Call when enough tools are done.",
}


# ── Reasoner ───────────────────────────────────────────────────────────────────

def _get_next_action(state: ResumeAgentState) -> dict:
    """Ask LLM what tool to call next given current state."""
    completed = state.tools_completed()
    remaining = [t for t in TOOL_REGISTRY if t not in completed]

    system = """
You are the reasoning component of a resume extraction agent.
Decide which tool to call next to extract information from a resume.

Available tools:
{tool_descriptions}

Decision rules:
- Always call extract_identity first if not done.
- Always call extract_work_history early — it is the richest section.
- extract_skills should be called after work_history (implicit skills depend on it).
- extract_education is important for academic CVs and senior candidates — always call it.
- assess_quality_signals should be called after work_history and skills are done.
- Call "synthesize" when you have identity + work_history + skills at minimum.
  You may synthesize before education/quality if the resume is sparse.
- If a section is clearly absent (e.g. resume has no skills section, no education listed),
  you may skip that tool and proceed to synthesize.

Respond with ONLY a JSON object:
{{"tool": "<tool_name>", "reasoning": "<one sentence why>"}}
""".format(
        tool_descriptions="\n".join(f"- {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())
    )

    user = f"""
Resume preview (first 400 chars): {state.resume_text[:400]}

Completed: {completed if completed else "none"}
Remaining: {remaining}
Iteration: {state.iteration}/{state.max_iterations}

What tool should I call next?
"""
    client = _get_client()
    response = client.chat.completions.create(
        model=config.model.llm_model,
        temperature=0.0,
        max_tokens=150,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(response.choices[0].message.content)


def _get_client() -> OpenAI:
    provider = config.model.llm_provider
    if provider == "groq":
        return OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── Synthesizer ────────────────────────────────────────────────────────────────

def _synthesize(state: ResumeAgentState) -> ResumeProfile:
    """
    Final step: assemble all tool outputs into a ResumeProfile.
    Fills defaults for any skipped tools, then asks LLM for narrative fields.
    """
    from pydantic import BaseModel as PydanticBaseModel

    # Safe defaults for any skipped tools
    identity = state.identity or CandidateIdentity()
    work_history = state.work_history or WorkHistory(
        total_years_experience=0.0,
        career_trajectory=CareerTrajectory.UNKNOWN,
        has_leadership_experience=False,
        has_people_management=False,
        highest_company_tier=CompanyTier.TIER_4,
        domains_worked_in=[],
    )
    skills = state.skills or SkillsProfile(explicit_skills=[], implicit_skills=[])
    education = state.education or EducationAndCredentials()
    quality = state.quality_signals or QualitySignals(
        exaggeration_index=0.3,
        specificity_score=0.5,
        buzzword_density=0.2,
        quantified_impact_ratio=0.3,
    )

    # Ask LLM for synthesis fields
    class SynthMeta(PydanticBaseModel):
        career_archetype: str
        career_narrative: str
        green_flags: list[str]
        red_flags: list[str]
        extraction_confidence: float
        extraction_notes: list[str]

    system = """
Synthesize the final narrative fields for a resume extraction.

career_archetype: One concise professional identity label. Be specific.
  e.g. "ML Platform Engineer", "Generalist Builder", "Researcher-Practitioner",
  "Enterprise Sales Leader", "Full-Stack Product Engineer", "Clinical Data Scientist",
  "DevOps/MLOps Specialist", "Early-Stage Startup Founder"

career_narrative: 2-3 specific sentences about this person's career arc. Not generic.

green_flags: Genuine standouts only. 2-5 items. Be selective.
  e.g. "Staff MLE at Cohere with 50M+ query RAG system in production",
  "PhD from Stanford NLP Group with 4 ACL/EMNLP publications",
  "Open source library with 480 GitHub stars"

red_flags: Real concerns only.
  e.g. "No quantified impact across 6 years of experience",
  "All claims are vague — no metrics, no specific technologies mentioned",
  "4-year gap in employment with no explanation"

extraction_confidence: 0.0–1.0. How complete and parseable was the resume?

extraction_notes: Caveats, ambiguities, or fields where you had low confidence.
"""

    user = f"""
Name: {identity.full_name}
Title: {identity.current_title} ({identity.current_seniority})
Total YoE: {work_history.total_years_experience}
Trajectory: {work_history.career_trajectory}
Highest company tier: {work_history.highest_company_tier}
Domains: {work_history.domains_worked_in}
Explicit skills: {skills.explicit_skills[:15]}
Implicit skills: {skills.implicit_skills[:10]}
Education: {education.highest_degree} from {education.institution}
Publications: {education.has_publications} | OSS: {education.has_open_source}
Exaggeration index: {quality.exaggeration_index}
Specificity score: {quality.specificity_score}
Quantified impact ratio: {quality.quantified_impact_ratio}
Consistency issues: {quality.consistency_issues}

Resume excerpt: {state.resume_text[:600]}

Write the synthesis fields.
"""

    meta = extract_structured(system, user, SynthMeta)

    return ResumeProfile(
        identity=identity,
        work_history=work_history,
        skills=skills,
        education=education,
        quality_signals=quality,
        career_archetype=meta.career_archetype,
        career_narrative=meta.career_narrative,
        green_flags=meta.green_flags,
        red_flags=meta.red_flags,
        extraction_confidence=meta.extraction_confidence,
        extraction_notes=meta.extraction_notes,
    )


# ── ReAct Loop ─────────────────────────────────────────────────────────────────

def run_resume_extraction_agent(resume_text: str) -> tuple[ResumeProfile, list[dict]]:
    """
    Run the ReAct agent to extract structured information from a resume.

    Args:
        resume_text: Raw resume text (parsed from PDF/DOCX/TXT).

    Returns:
        (ResumeProfile, reasoning_trace)
    """
    logger.info("Starting resume extraction agent...")
    state = ResumeAgentState(resume_text=resume_text)

    while not state.done and state.iteration < state.max_iterations:
        state.iteration += 1
        time.sleep(0.3)  # brief pause to avoid rate limit bursts

        # ── Thought + Action decision ──────────────────────────────────────
        try:
            action_decision = _get_next_action(state)
            tool_name = action_decision.get("tool", "synthesize")
            reasoning = action_decision.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Reasoner failed at step {state.iteration}: {e}. Forcing synthesize.")
            tool_name = "synthesize"
            reasoning = "Reasoner error — falling back to synthesis."

        # ── Execute Action ─────────────────────────────────────────────────
        if tool_name == "synthesize":
            try:
                # Force required tools if not done
                if not state.identity:
                    logger.info("Force-running extract_identity before synthesis")
                    state.identity = tool_extract_identity(resume_text)
                if not state.work_history:
                    logger.info("Force-running extract_work_history before synthesis")
                    state.work_history = tool_extract_work_history(resume_text)
                if not state.skills:
                    logger.info("Force-running extract_skills before synthesis")
                    state.skills = tool_extract_skills(resume_text)

                profile = _synthesize(state)
                state.log_step(
                    thought=reasoning,
                    action="synthesize",
                    observation=f"Complete. Archetype: {profile.career_archetype}. "
                               f"Confidence: {profile.extraction_confidence:.2f}"
                )
                state.done = True
                logger.info(
                    f"Resume agent complete in {state.iteration} steps | "
                    f"{profile.identity.full_name or 'Unknown'} | "
                    f"{profile.career_archetype} | "
                    f"YoE: {profile.total_years_experience} | "
                    f"Confidence: {profile.extraction_confidence:.2f}"
                )
                return profile, state.steps

            except Exception as e:
                state.error = str(e)
                state.log_step(reasoning, "synthesize", f"ERROR: {e}")
                raise RuntimeError(f"Resume synthesis failed: {e}") from e

        elif tool_name in TOOL_REGISTRY:
            tool_fn = TOOL_REGISTRY[tool_name]
            try:
                result = tool_fn(resume_text)
                observation = f"Success. Keys: {list(result.model_dump().keys())}"

                if tool_name == "extract_identity":
                    state.identity = result
                elif tool_name == "extract_work_history":
                    state.work_history = result
                elif tool_name == "extract_skills":
                    state.skills = result
                elif tool_name == "extract_education":
                    state.education = result
                elif tool_name == "assess_quality_signals":
                    state.quality_signals = result

                state.log_step(reasoning, tool_name, observation)

            except Exception as e:
                state.log_step(reasoning, tool_name, f"ERROR: {e}")
                logger.warning(f"Tool {tool_name} failed: {e}")
                time.sleep(1.0)

        else:
            logger.warning(f"Unknown tool: {tool_name}. Skipping.")
            state.log_step(reasoning, f"UNKNOWN:{tool_name}", "Tool not in registry. Skipping.")

    # Max iterations — synthesize whatever we have
    logger.warning(f"Max iterations ({state.max_iterations}) reached. Synthesizing partial data.")
    profile = _synthesize(state)
    return profile, state.steps


# ── Public interface ───────────────────────────────────────────────────────────

def extract_resume(resume_text: str) -> ResumeProfile:
    """
    Public interface for resume extraction.
    Uses the ReAct agent internally.
    Drop-in replacement for the old single-call extractor.

    Args:
        resume_text: Raw resume text (parsed from PDF/DOCX/TXT).

    Returns:
        ResumeProfile: Fully structured, domain-agnostic resume representation.
    """
    profile, _ = run_resume_extraction_agent(resume_text)
    return profile