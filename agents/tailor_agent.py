# agents/tailor_agent.py
#
# WHAT THIS AGENT DOES:
# For every matched job, generates a tailored "application kit":
#   - a rewritten professional summary aimed at that specific role
#   - resume bullet points re-ordered/re-worded to mirror the JD language
#   - a full cover letter
#   - a list of JD keywords the resume is missing (useful for ATS systems)
#
# IMPORTANT: it rewords and emphasises what is genuinely on the resume.
# The prompt explicitly forbids inventing experience — a tailored resume
# that lies gets candidates blacklisted.

from typing import List

from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import get_llm
from graph.state import ApplicationKit, PipelineState

# Cap LLM spend per run — tailor only the best N matches
MAX_KITS_PER_RUN = 8


class TailoredMaterials(BaseModel):
    tailored_summary: str = Field(..., description="2-3 sentence professional summary targeting this job")
    tailored_bullets: List[str] = Field(..., description="5-7 resume bullet points, most relevant first")
    cover_letter: str = Field(..., description="Complete cover letter, 250-350 words")
    keywords_to_add: List[str] = Field(
        default_factory=list,
        description="Important JD keywords/skills missing from the resume",
    )


_TAILOR_PROMPT = PromptTemplate(
    template="""
You are an expert resume writer and career coach.

Create tailored application materials for this candidate and job.

STRICT RULES:
- NEVER invent experience, skills, projects, or numbers not present in the resume.
- Only re-word, re-order, and emphasise what the candidate actually has.
- Mirror the job description's terminology where the resume supports it
  (e.g. if the JD says "LLM applications" and the resume has "RAG systems",
  use "LLM applications (RAG)").
- The cover letter must sound human and specific — no "I am writing to
  express my interest" openers. Reference the company and role concretely.
- Keep the cover letter 250-350 words.

CANDIDATE RESUME:
Name: {name}
Summary: {summary}
Skills: {skills}
Experience:
{experience}
Education:
{education}

JOB POSTING:
Title: {job_title}
Company: {company}
Description:
{job_description}

{format_instructions}
Return ONLY valid JSON.
""",
    input_variables=[
        "name", "summary", "skills", "experience", "education",
        "job_title", "company", "job_description",
    ],
    partial_variables={
        "format_instructions": JsonOutputParser(
            pydantic_object=TailoredMaterials
        ).get_format_instructions()
    },
)


def build_kit(resume, match) -> ApplicationKit:
    """One LLM call: produce the full application kit for one job."""
    parser = JsonOutputParser(pydantic_object=TailoredMaterials)
    # Higher temperature than parsing/scoring — this is creative writing
    chain = _TAILOR_PROMPT | get_llm(temperature=0.7) | parser

    experience_text = "\n".join(
        f"- {w.role} at {w.company} ({w.duration}): {w.description}"
        for w in resume.work_experience
    ) or "(none listed)"
    education_text = "\n".join(
        f"- {e.degree}, {e.institution} ({e.year})" for e in resume.education
    ) or "(none listed)"

    result = chain.invoke({
        "name": resume.name,
        "summary": resume.summary,
        "skills": ", ".join(resume.skills),
        "experience": experience_text,
        "education": education_text,
        "job_title": match.job.title,
        "company": match.job.company,
        "job_description": match.job.description[:6000],
    })
    materials = TailoredMaterials.model_validate(result)

    return ApplicationKit(
        job_url=match.job.url,
        job_title=match.job.title,
        company=match.job.company,
        match_score=match.score_percent,
        tailored_summary=materials.tailored_summary,
        tailored_bullets=materials.tailored_bullets,
        cover_letter=materials.cover_letter,
        keywords_to_add=materials.keywords_to_add,
    )


def tailor_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node. Reads: resume, matches.
    Writes: kits, status, errors.
    """
    errors = list(state.get("errors", []))
    resume = state.get("resume")
    matches = state.get("matches", [])

    if not resume or not matches:
        status = "Tailor: no matches to tailor for"
        logger.info(status)
        return {"kits": [], "status": status, "errors": errors}

    # Each kit is an independent (large) LLM generation, so build them
    # CONCURRENTLY — bounded by LLM_CONCURRENCY to respect provider rate limits.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from config import LLM_CONCURRENCY

    to_build = matches[:MAX_KITS_PER_RUN]
    for match in to_build:
        logger.info(f"Tailoring for: {match.job.title} @ {match.job.company}")

    kits: List[ApplicationKit] = []
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futures = {pool.submit(build_kit, resume, m): m for m in to_build}
        for future in as_completed(futures):
            match = futures[future]
            try:
                kits.append(future.result())
            except Exception as e:
                logger.error(f"Tailoring failed for {match.job.url}: {e}")
                errors.append(f"Tailoring failed for {match.job.title}: {e}")

    skipped = max(0, len(matches) - MAX_KITS_PER_RUN)
    status = f"Tailor: built {len(kits)} application kits" + (
        f" ({skipped} matches skipped by per-run cap)" if skipped else ""
    )
    logger.info(status)
    return {"kits": kits, "status": status, "errors": errors}
