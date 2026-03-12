"""
ReAct Agent for JD Extraction.

Implements the Reasoning + Acting loop for structured JD extraction.
The agent reasons about what to extract, calls focused tools, observes
results, and decides whether to continue or synthesize.

Architecture:
    Thought → Action (tool call) → Observation → Thought → ...
    until all required information is extracted → Synthesize → JDProfile

Why ReAct over a single LLM call:
- Each tool is focused → better extraction quality per field
- Agent can retry a tool if it detects low-quality output
- Agent can skip tools if information is clearly absent (e.g. no compensation)
- Full reasoning trace is logged → debuggable
- Adding a new extraction dimension = add a tool, not modify a giant prompt
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI
from pydantic import BaseModel

from src.extraction.jd_extraction.schemas import (
    JDProfile, RoleIdentity, HardRequirements, SoftRequirements,
    RoleCharacter, Compensation
)
from src.extraction.jd_extraction.agent_tools import (
    tool_extract_role_identity,
    tool_extract_hard_requirements,
    tool_extract_soft_requirements,
    tool_extract_role_character,
    tool_extract_compensation,
)
from src.utils.llm_client import extract_structured
from config import config

logger = logging.getLogger(__name__)


# ── Agent State ────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable state carried through the ReAct loop."""
    jd_text: str
    steps: list[dict] = field(default_factory=list)  # Full reasoning trace

    # Accumulated tool results
    role_identity: RoleIdentity | None = None
    hard_requirements: HardRequirements | None = None
    soft_requirements: SoftRequirements | None = None
    role_character: RoleCharacter | None = None
    compensation: Compensation | None = None

    # Agent control
    iteration: int = 0
    max_iterations: int = 8
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
        if self.role_identity: completed.append("extract_role_identity")
        if self.hard_requirements: completed.append("extract_hard_requirements")
        if self.soft_requirements: completed.append("extract_soft_requirements")
        if self.role_character: completed.append("extract_role_character")
        if self.compensation: completed.append("extract_compensation")
        return completed

    def all_required_complete(self) -> bool:
        """Core extraction is done when role + requirements are extracted."""
        return bool(self.role_identity and self.hard_requirements)

    def all_complete(self) -> bool:
        return bool(
            self.role_identity and self.hard_requirements and
            self.soft_requirements and self.role_character and
            self.compensation
        )


# ── Tool Registry ──────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "extract_role_identity": tool_extract_role_identity,
    "extract_hard_requirements": tool_extract_hard_requirements,
    "extract_soft_requirements": tool_extract_soft_requirements,
    "extract_role_character": tool_extract_role_character,
    "extract_compensation": tool_extract_compensation,
}

TOOL_DESCRIPTIONS = {
    "extract_role_identity": "Extract job title, seniority level, location, remote policy, employment type, company stage.",
    "extract_hard_requirements": "Extract required skills (atomic tokens), minimum years of experience, required education, required certifications, required domain experience.",
    "extract_soft_requirements": "Extract preferred/nice-to-have skills, preferred company background, preferred education.",
    "extract_role_character": "Extract ownership style, work style, pace signals, autonomy level, collaboration surface, red flag signals in the JD language.",
    "extract_compensation": "Extract salary range, equity, bonus, benefits highlights. Return nulls if compensation is not mentioned.",
    "synthesize": "Finalize the extraction — synthesize all gathered information into a complete JDProfile. Call this when all needed tools have been run.",
}


# ── Reasoner: decides next action ─────────────────────────────────────────────

def _get_next_action(state: AgentState) -> dict:
    """
    Ask the LLM to reason about what to do next given current state.
    Returns a structured action: {tool: str, reasoning: str}
    """
    completed = state.tools_completed()
    remaining = [t for t in TOOL_REGISTRY if t not in completed]

    system = """
You are the reasoning component of a JD extraction agent.
Your job: decide what tool to call next to extract information from a job description.

Available tools and what they extract:
{tool_descriptions}

Rules:
- Always call extract_role_identity first if not done.
- Always call extract_hard_requirements early — it is the most important.
- Call extract_soft_requirements, extract_role_character, extract_compensation as needed.
- Call "synthesize" only when you have enough information (at minimum: role_identity + hard_requirements).
- If a section is clearly absent from the JD (e.g. no compensation mentioned), you may skip that tool and call synthesize.
- If a previous tool result looks low quality or suspicious, you may re-call it (but max once per tool).

Respond with ONLY a JSON object:
{{"tool": "<tool_name>", "reasoning": "<one sentence why>"}}
""".format(
        tool_descriptions="\n".join(f"- {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())
    )

    user = f"""
JD (first 500 chars): {state.jd_text[:500]}

Completed tools: {completed if completed else "none yet"}
Remaining tools: {remaining}
Iteration: {state.iteration}/{state.max_iterations}

What tool should I call next?
"""

    client = _get_client()
    response = client.chat.completions.create(
        model=config.model.llm_model,
        temperature=0.0,
        max_tokens=200,
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

class _SynthesisOutput(JDProfile):
    """Used for structured synthesis — same as JDProfile."""
    pass


def _synthesize(state: AgentState) -> JDProfile:
    """
    Final step: assemble all tool outputs into a complete JDProfile.
    Fills in any missing sub-schemas with sensible defaults.
    """
    from src.extraction.jd_extraction.schemas import OwnershipStyle, WorkStyle

    # Build defaults for any skipped tools
    role_char = state.role_character or RoleCharacter(
        ownership_style=OwnershipStyle.UNKNOWN,
        work_style=WorkStyle.UNKNOWN,
    )
    comp = state.compensation or Compensation()

    # Ask LLM to synthesize persona + role_in_one_line + confidence
    class SynthMeta(BaseModel):
        ideal_candidate_persona: str
        role_in_one_line: str
        extraction_confidence: float
        extraction_notes: list[str]

    system = """
You are finalizing a structured JD extraction. Based on the extracted sub-schemas,
write the synthesis fields.

ideal_candidate_persona: 2-3 concrete sentences about what the ideal hire looks like.
  Not generic HR language — be specific about experience, traits, and background.
role_in_one_line: One punchy sentence summarizing the role for a potential candidate.
extraction_confidence: 0.0-1.0. How complete and unambiguous was the JD?
  1.0 = detailed, specific, easy to extract. 0.5 = vague in places. 0.2 = very sparse.
extraction_notes: List any ambiguities, missing sections, or low-confidence extractions.
"""

    hard_req = state.hard_requirements
    role_id = state.role_identity

    user = f"""
Job: {role_id.job_title if role_id else 'Unknown'} ({role_id.seniority_level if role_id else 'Unknown'})
Required skills: {hard_req.required_skills if hard_req else []}
Required YoE: {hard_req.required_years_of_experience if hard_req else None}
Required domain: {hard_req.required_domain_experience if hard_req else []}
Preferred skills: {state.soft_requirements.preferred_skills if state.soft_requirements else []}
Ownership style: {role_char.ownership_style}
Work style: {role_char.work_style}
Pace: {role_char.pace_signal}
Red flags: {role_char.red_flag_signals}

JD excerpt: {state.jd_text[:600]}

Write the synthesis fields.
"""

    meta = extract_structured(system, user, SynthMeta)

    return JDProfile(
        role_identity=state.role_identity,
        hard_requirements=state.hard_requirements,
        soft_requirements=state.soft_requirements or SoftRequirements(),
        role_character=role_char,
        compensation=comp,
        ideal_candidate_persona=meta.ideal_candidate_persona,
        role_in_one_line=meta.role_in_one_line,
        extraction_confidence=meta.extraction_confidence,
        extraction_notes=meta.extraction_notes,
    )


# ── ReAct Loop ─────────────────────────────────────────────────────────────────

def run_jd_extraction_agent(jd_text: str) -> tuple[JDProfile, list[dict]]:
    """
    Run the ReAct agent to extract structured information from a JD.

    Args:
        jd_text: Raw job description text (any domain).

    Returns:
        (JDProfile, reasoning_trace) — the structured profile and full agent trace.
    """
    logger.info("Starting JD extraction agent...")
    state = AgentState(jd_text=jd_text)

    while not state.done and state.iteration < state.max_iterations:
        state.iteration += 1
        time.sleep(0.3)  # brief pause to avoid rate limit bursts

        # ── Thought + Action decision ──────────────────────────────────────
        try:
            action_decision = _get_next_action(state)
            tool_name = action_decision.get("tool", "synthesize")
            reasoning = action_decision.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Reasoner failed: {e}. Forcing synthesize.")
            tool_name = "synthesize"
            reasoning = "Reasoner error — falling back to synthesis."

        # ── Execute Action ─────────────────────────────────────────────────
        if tool_name == "synthesize":
            try:
                if not state.all_required_complete():
                    # Force-run missing required tools before synthesizing
                    if not state.role_identity:
                        state.role_identity = tool_extract_role_identity(jd_text)
                    if not state.hard_requirements:
                        state.hard_requirements = tool_extract_hard_requirements(jd_text)

                profile = _synthesize(state)
                state.log_step(
                    thought=reasoning,
                    action="synthesize",
                    observation=f"Synthesis complete. Confidence: {profile.extraction_confidence:.2f}"
                )
                state.done = True
                logger.info(f"Agent complete in {state.iteration} steps. "
                           f"Confidence: {profile.extraction_confidence:.2f}")
                return profile, state.steps

            except Exception as e:
                state.error = str(e)
                state.log_step(reasoning, "synthesize", f"ERROR: {e}")
                raise RuntimeError(f"Synthesis failed: {e}") from e

        elif tool_name in TOOL_REGISTRY:
            tool_fn = TOOL_REGISTRY[tool_name]
            try:
                result = tool_fn(jd_text)
                observation = f"Success. Extracted: {result.model_dump()}"

                # Store result in state
                if tool_name == "extract_role_identity":
                    state.role_identity = result
                elif tool_name == "extract_hard_requirements":
                    state.hard_requirements = result
                elif tool_name == "extract_soft_requirements":
                    state.soft_requirements = result
                elif tool_name == "extract_role_character":
                    state.role_character = result
                elif tool_name == "extract_compensation":
                    state.compensation = result

                state.log_step(reasoning, tool_name, observation[:300])

            except Exception as e:
                observation = f"ERROR: {e}"
                state.log_step(reasoning, tool_name, observation)
                logger.warning(f"Tool {tool_name} failed: {e}")
                time.sleep(1)

        else:
            logger.warning(f"Unknown tool requested: {tool_name}. Skipping.")
            state.log_step(reasoning, f"UNKNOWN:{tool_name}", "Tool not found. Skipping.")

    # Max iterations hit — synthesize whatever we have
    logger.warning("Max iterations reached. Synthesizing with partial data.")
    profile = _synthesize(state)
    return profile, state.steps


# ── Public interface (drop-in replacement for old extract_jd) ─────────────────

def extract_jd(jd_text: str) -> JDProfile:
    """
    Public interface for JD extraction.
    Uses the ReAct agent internally.
    Drop-in replacement for the old single-call extractor.

    Args:
        jd_text: Raw job description text.

    Returns:
        JDProfile: Fully structured, domain-agnostic JD representation.
    """
    profile, trace = run_jd_extraction_agent(jd_text)

    logger.info(
        f"JD extracted: '{profile.role_identity.job_title}' | "
        f"Seniority: {profile.role_identity.seniority_level} | "
        f"Required skills: {len(profile.hard_requirements.required_skills)} | "
        f"Agent steps: {len(trace)} | "
        f"Confidence: {profile.extraction_confidence:.2f}"
    )

    return profile