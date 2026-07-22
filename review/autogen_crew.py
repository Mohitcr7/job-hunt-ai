# review/autogen_crew.py
#
# WHAT THIS FILE DOES:
# A two-persona quality review pass over each cover letter, in the style
# of an AutoGen/CrewAI "crew":
#
#   Persona 1 — REVIEWER: a picky hiring manager who critiques the letter
#   Persona 2 — EDITOR  : rewrites the letter to address the critique
#
# WHY NOT THE ag2/AutoGen FRAMEWORK ITSELF:
# ag2's multi-agent chat needs an OpenAI-compatible endpoint config, extra
# setup, and gives us little here — the reviewer→editor exchange is a
# fixed two-step conversation, so two plain LLM calls express it exactly,
# work with Gemini out of the box, and are far easier to debug. Same
# pattern, fewer moving parts.
#
# Toggle with ENABLE_REVIEW_CREW=false in .env to save one LLM round-trip
# per kit.

import os
from typing import List

from loguru import logger
from langchain_core.prompts import PromptTemplate

from config import get_llm
from graph.state import ApplicationKit, PipelineState

ENABLE_REVIEW_CREW = os.getenv("ENABLE_REVIEW_CREW", "true").lower() == "true"

_REVIEWER_PROMPT = PromptTemplate(
    template="""
You are a skeptical hiring manager at {company} reviewing a cover letter
for the {job_title} position. You read hundreds of these.

Critique this letter in 3-5 specific bullet points. Look for:
- generic filler that could be sent to any company
- claims not backed by anything concrete
- weak or cliché openings/closings
- anything that sounds AI-generated
- missed opportunities to connect the candidate's background to the role

COVER LETTER:
{cover_letter}

Return only the bullet-point critique, nothing else.
""",
    input_variables=["company", "job_title", "cover_letter"],
)

_EDITOR_PROMPT = PromptTemplate(
    template="""
You are an expert editor. Rewrite this cover letter to address every point
in the reviewer's critique, while keeping all factual claims exactly as
they are — do not invent anything new. Keep it 250-350 words.

ORIGINAL COVER LETTER:
{cover_letter}

REVIEWER CRITIQUE:
{critique}

Return only the improved cover letter text, nothing else.
""",
    input_variables=["cover_letter", "critique"],
)


def review_cover_letter(kit: ApplicationKit) -> str:
    """Runs the reviewer→editor exchange, returns the improved letter."""
    llm_critic = get_llm(temperature=0.3)
    llm_editor = get_llm(temperature=0.5)

    critique = (_REVIEWER_PROMPT | llm_critic).invoke({
        "company": kit["company"],
        "job_title": kit["job_title"],
        "cover_letter": kit["cover_letter"],
    }).content

    improved = (_EDITOR_PROMPT | llm_editor).invoke({
        "cover_letter": kit["cover_letter"],
        "critique": critique,
    }).content

    return improved.strip()


def review_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node between Tailor and Applier.
    Reads: kits. Writes: kits (with improved cover letters), status.
    """
    errors = list(state.get("errors", []))
    kits: List[ApplicationKit] = state.get("kits", [])

    if not ENABLE_REVIEW_CREW or not kits:
        status = "Review: skipped" if not ENABLE_REVIEW_CREW else "Review: no kits"
        return {"status": status, "errors": errors}

    # Each kit's reviewer→editor exchange is independent of the others, so run
    # them CONCURRENTLY (bounded by LLM_CONCURRENCY). The two calls WITHIN a kit
    # stay ordered (the editor needs the critique) inside review_cover_letter.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from config import LLM_CONCURRENCY

    reviewed = 0
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futures = {pool.submit(review_cover_letter, kit): kit for kit in kits}
        for future in as_completed(futures):
            kit = futures[future]
            try:
                kit["cover_letter"] = future.result()
                reviewed += 1
            except Exception as e:
                # Keep the original letter — a failed review is not a lost kit
                logger.warning(f"Review failed for {kit['job_title']}: {e}")
                errors.append(f"Review failed for {kit['job_title']}: {e}")

    status = f"Review: polished {reviewed}/{len(kits)} cover letters"
    logger.info(status)
    return {"kits": kits, "status": status, "errors": errors}
