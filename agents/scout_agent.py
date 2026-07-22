# agents/scout_agent.py
#
# WHAT THIS AGENT DOES:
# The Scout is the first node in the pipeline. It has two jobs:
#   1. Parse the resume PDF into structured data (once, cached to JSON)
#   2. Scrape fresh job postings from all configured sources
#
# It writes `resume` and `jobs` into the pipeline state.

import json
from pathlib import Path

from loguru import logger

from config import JOB_PREFERENCES
from graph.state import PipelineState
from tools.resume_parser_tool import ParsedResume, load_and_parse_resume
from tools.scraper_tool import scrape_all_jobs

PARSED_RESUME_CACHE = "data/parsed_resume.json"


def get_resume(resume_path: str, force_reparse: bool = False) -> ParsedResume:
    """
    Returns the parsed resume, using the JSON cache when possible.
    Parsing costs one LLM call, so we only re-parse when the PDF
    is newer than the cache (or when forced).
    """
    cache = Path(PARSED_RESUME_CACHE)
    pdf = Path(resume_path)

    if not force_reparse and cache.exists():
        pdf_is_newer = pdf.exists() and pdf.stat().st_mtime > cache.stat().st_mtime
        if not pdf_is_newer:
            logger.info(f"Loading cached parsed resume from {cache}")
            with open(cache) as f:
                return ParsedResume.model_validate(json.load(f))

    resume = load_and_parse_resume(resume_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(resume.model_dump(), f, indent=2)
    return resume


def scout_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node. Reads: resume_path, search_terms, locations, hours_old.
    Writes: resume, jobs, status, errors.
    """
    errors = list(state.get("errors", []))

    # 1. Resume
    resume_path = state.get("resume_path", "data/resume.pdf")
    try:
        resume = get_resume(resume_path)
    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        return {
            "resume": None,
            "jobs": [],
            "errors": errors + [f"Resume parsing failed: {e}"],
            "status": "Failed: could not parse resume",
        }

    # 2. Jobs
    search_terms = state.get("search_terms") or JOB_PREFERENCES["roles"]
    locations = state.get("locations") or JOB_PREFERENCES["locations"]
    hours_old = state.get("hours_old", 24)

    try:
        jobs = scrape_all_jobs(
            search_terms=search_terms,
            locations=locations,
            hours_old=hours_old,
            enable_naukri=True,
            enable_company_pages=False,  # selectors go stale fast; enable once verified
        )
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        jobs = []
        errors.append(f"Scraping failed: {e}")

    status = f"Scout: parsed resume, found {len(jobs)} jobs"
    logger.info(status)
    return {"resume": resume, "jobs": jobs, "errors": errors, "status": status}
