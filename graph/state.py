# graph/state.py
#
# WHAT THIS FILE DOES:
# Defines the shared "state" object that flows through the LangGraph
# pipeline. Every agent (node) receives this state, reads what it needs,
# and writes its results back into it.
#
# KEY CONCEPT:
# - TypedDict : a dictionary with a declared shape. LangGraph uses it to
#   know which keys exist in the state and to merge updates from each node.
#   Unlike a Pydantic model, it stays a plain dict at runtime — fast and
#   easy to serialize.

from typing import List, Optional, TypedDict

from tools.resume_parser_tool import ParsedResume
from tools.vector_store_tool import Job, JobMatch


class ApplicationKit(TypedDict):
    """Everything the Tailor Agent produces for one matched job."""
    job_url: str
    job_title: str
    company: str
    match_score: int
    tailored_summary: str        # rewritten resume summary targeting this job
    tailored_bullets: List[str]  # resume bullet points emphasising relevant skills
    cover_letter: str            # full cover letter text
    keywords_to_add: List[str]   # JD keywords missing from the resume


class PipelineState(TypedDict, total=False):
    """
    The single source of truth passed between agents.

    total=False means every key is optional — nodes only set the keys
    they are responsible for, and LangGraph merges the updates.
    """
    # --- inputs ---
    resume_path: str                  # path to the PDF resume
    search_terms: List[str]           # roles to search for
    locations: List[str]              # locations to search in
    hours_old: int                    # freshness window for job postings
    min_score: int                    # match threshold (0-100)

    # --- produced by nodes ---
    resume: Optional[ParsedResume]    # Scout: parsed resume
    jobs: List[Job]                   # Scout: raw scraped jobs
    matches: List[JobMatch]           # Matcher: jobs above threshold
    kits: List[ApplicationKit]        # Tailor: application materials
    applied: List[str]                # Applier: job URLs recorded in tracker

    # --- bookkeeping ---
    errors: List[str]                 # non-fatal errors collected along the way
    status: str                       # human-readable progress message
