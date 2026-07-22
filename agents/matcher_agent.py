# agents/matcher_agent.py
#
# WHAT THIS AGENT DOES:
# Takes the scraped jobs + parsed resume and ranks how well each job
# fits, in two stages:
#
#   Stage 1 (free, fast)  : FAISS semantic similarity between the resume
#                           and every job description. Filters out the
#                           clearly irrelevant ones.
#   Stage 2 (LLM, cheap)  : the top candidates get a proper LLM review —
#                           the model reads the JD against the resume and
#                           returns a 0-100 fit score with reasoning.
#
# Two stages because embeddings are great at "roughly similar topic" but
# blind to hard requirements ("8+ years", "must know Rust"). The LLM
# catches those, and we only pay for it on jobs that already look close.

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import get_llm, MATCH_THRESHOLD
from graph.state import PipelineState
from tools.vector_store_tool import JobMatch, JobVectorStore

# How many FAISS candidates get the (paid) LLM review
LLM_RERANK_TOP_K = 15


class FitAssessment(BaseModel):
    """What the LLM returns for each job it reviews."""
    score: int = Field(..., description="Overall fit score 0-100")
    reasoning: str = Field("", description="One-sentence justification")
    missing_requirements: List[str] = Field(
        default_factory=list,
        description="Hard requirements in the JD the candidate does not meet",
    )


_FIT_PROMPT = PromptTemplate(
    template="""
You are a technical recruiter assessing candidate-job fit.

CANDIDATE PROFILE:
{resume_summary}

Skills: {skills}
Experience: {experience_years} years

JOB POSTING:
Title: {job_title}
Company: {company}
Description:
{job_description}

Score the fit from 0-100:
- 90-100: near-perfect match, apply immediately
- 70-89 : strong match, worth applying
- 50-69 : partial match, missing some requirements
- 0-49  : poor match

Penalise hard mismatches (required years of experience far above the
candidate's, mandatory skills the candidate lacks, wrong domain).
Reward overlapping core skills and domain alignment.

{format_instructions}
Return ONLY valid JSON.
""",
    input_variables=[
        "resume_summary", "skills", "experience_years",
        "job_title", "company", "job_description",
    ],
    partial_variables={
        "format_instructions": JsonOutputParser(
            pydantic_object=FitAssessment
        ).get_format_instructions()
    },
)


def llm_score_job(resume, job, experience_years: int) -> FitAssessment:
    """One LLM call: read the JD against the resume, return a FitAssessment."""
    parser = JsonOutputParser(pydantic_object=FitAssessment)
    chain = _FIT_PROMPT | get_llm(temperature=0.0) | parser

    result = chain.invoke({
        "resume_summary": resume.summary,
        "skills": ", ".join(resume.skills),
        "experience_years": experience_years,
        "job_title": job.title,
        "company": job.company,
        # Truncate very long JDs — the useful signal is at the top
        "job_description": job.description[:6000],
    })
    return FitAssessment.model_validate(result)


def faiss_prefilter(resume, jobs, min_score: int = 50, top_k: int = None) -> List[JobMatch]:
    """
    STAGE 1 (fast, free, no LLM): embed the jobs and rank them by semantic
    similarity to the resume, returning everything above `min_score`.

    This is the quick pass whose results we can show the user immediately while
    the paid LLM rerank runs in the background.
    """
    store = JobVectorStore()
    store.add_jobs(jobs)

    resume_text = (
        f"{resume.summary}\nSkills: {', '.join(resume.skills)}\n{resume.raw_text[:3000]}"
    )
    k = top_k if top_k is not None else LLM_RERANK_TOP_K * 3
    return store.find_matching_jobs(resume_text, top_k=k, min_score=min_score)


def rerank_candidates(resume, candidates, top_n=None) -> List[JobMatch]:
    """
    STAGE 2 (free, off the generation budget): sharpen the FAISS shortlist with a
    dedicated cross-encoder reranker and keep the top_n most relevant for the LLM.
    When the reranker is disabled/unavailable it falls back to the FAISS order.
    This narrows how many jobs reach the expensive LLM fit-scoring.
    """
    from config import ENABLE_RERANK, RERANK_TOP_N

    n = top_n if top_n is not None else RERANK_TOP_N
    if not ENABLE_RERANK or not candidates:
        return candidates[:LLM_RERANK_TOP_K]

    from tools.reranker_tool import rerank_jobs

    query = f"{resume.summary} Skills: {', '.join(resume.skills)}"
    return rerank_jobs(query, candidates, top_n=n)


def llm_rerank(resume, candidates, min_score, experience_years, errors=None) -> List[JobMatch]:
    """
    STAGE 3 (paid): the LLM reads each shortlisted JD against the resume and
    returns a real fit score — catching hard requirements embeddings miss
    ("8+ years required"). Falls back to the embedding score if a call fails.

    The scoring calls run CONCURRENTLY (bounded by LLM_CONCURRENCY); results are
    merged in the main thread, so parallelism never causes a race.
    """
    from config import LLM_CONCURRENCY

    if errors is None:
        errors = []
    cands = candidates[:LLM_RERANK_TOP_K]
    matches: List[JobMatch] = []

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futures = {pool.submit(llm_score_job, resume, c.job, experience_years): c for c in cands}
        for future in as_completed(futures):
            cand = futures[future]
            try:
                assessment = future.result()
                if assessment.score >= min_score:
                    matches.append(JobMatch(
                        job=cand.job,
                        score=assessment.score / 100.0,
                        score_percent=assessment.score,
                    ))
                    logger.info(f"  {assessment.score}% — {cand.job.title} @ {cand.job.company}")
            except Exception as e:
                logger.warning(f"LLM scoring failed for {cand.job.url}: {e}")
                errors.append(f"LLM scoring failed for {cand.job.title}: {e}")
                if cand.score_percent >= min_score:
                    matches.append(cand)

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def matcher_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node (used by the CLI/graph). Runs both stages back-to-back.
    Reads: resume, jobs, min_score. Writes: matches, status, errors.
    """
    errors = list(state.get("errors", []))
    resume = state.get("resume")
    jobs = state.get("jobs", [])
    min_score = state.get("min_score", MATCH_THRESHOLD)

    if not resume or not jobs:
        status = "Matcher: nothing to match (no resume or no jobs)"
        logger.warning(status)
        return {"matches": [], "status": status, "errors": errors}

    from config import JOB_PREFERENCES
    experience_years = JOB_PREFERENCES.get("experience_years", 0)

    # Stage 1: low-floor FAISS prefilter (removes obvious noise).
    # Stage 2: reranker narrows to the most relevant few.
    # Stage 3: LLM judges fit on that shortlist.
    candidates = faiss_prefilter(resume, jobs, min_score=30)
    candidates = rerank_candidates(resume, candidates)
    logger.info(f"Matcher: {len(candidates)} candidates go to LLM review")
    matches = llm_rerank(resume, candidates, min_score, experience_years, errors)

    status = f"Matcher: {len(matches)} jobs above {min_score}% threshold"
    logger.info(status)
    return {"matches": matches, "status": status, "errors": errors}
