# job_hunt_agent.py
#
# WHAT THIS FILE DOES:
# The Amazon Bedrock AgentCore Runtime entrypoint. AgentCore wraps our existing
# LangGraph crew and gives it serverless deployment, session isolation, memory,
# identity, and observability — WITHOUT rewriting any agent logic.
#
# The crew itself is unchanged: this file just adapts one invocation
# (resume text in → tailored kits out) to run each agent node in sequence,
# sourcing job postings from a Bedrock Managed Knowledge Base when deployed
# (falling back to the local scraper for dev).
#
# Deploy:
#   agentcore configure --entrypoint job_hunt_agent.py
#   agentcore launch --env MANAGED_KB_ID=... --env GUARDRAIL_ID=... --env GUARDRAIL_VERSION=...
# Invoke:
#   agentcore invoke '{"resume": "<resume text>"}'
#
# Local smoke test (no AWS):  python job_hunt_agent.py

import dataclasses
from typing import Optional

from loguru import logger

from bedrock_agentcore import BedrockAgentCoreApp

from config import MATCH_THRESHOLD, MANAGED_KB_ID, JOB_PREFERENCES
from tools.resume_parser_tool import parse_resume_with_llm
from tools.vector_store_tool import Job

app = BedrockAgentCoreApp()


def _jobs_from_payload(raw_jobs: list) -> list:
    """Build Job objects from postings passed directly in the invocation payload."""
    jobs = []
    for j in raw_jobs:
        jobs.append(Job(
            title=j.get("title", ""),
            company=j.get("company", ""),
            location=j.get("location", ""),
            description=j.get("description", ""),
            url=j.get("url", ""),
            platform=j.get("platform", "payload"),
            posted_date=j.get("posted_date", ""),
            salary=j.get("salary", ""),
            job_id=j.get("job_id", j.get("url", "")),
        ))
    return jobs


def _get_jobs(resume, payload: dict) -> list:
    """
    Resolve the job source, in priority order:
      1. postings supplied in the payload      (caller already retrieved them)
      2. Bedrock Managed Knowledge Base         (the deployed RAG path)
      3. live scraping                          (local/dev fallback)
    """
    if payload.get("jobs"):
        logger.info("Using job postings supplied in the payload")
        return _jobs_from_payload(payload["jobs"])

    query = f"{resume.summary} Skills: {', '.join(resume.skills)}"

    if MANAGED_KB_ID:
        from tools.bedrock_kb_tool import retrieve_jobs_from_kb
        return retrieve_jobs_from_kb(query, top_k=payload.get("top_k", 15))

    # Dev fallback — requires jobspy/Playwright and outbound network.
    logger.info("No Managed KB configured — falling back to live scraping")
    from tools.scraper_tool import scrape_all_jobs
    return scrape_all_jobs(
        search_terms=payload.get("search_terms") or JOB_PREFERENCES["roles"],
        locations=payload.get("locations") or JOB_PREFERENCES["locations"],
        hours_old=payload.get("hours_old", 24),
        enable_naukri=False,          # keep the serverless path light
        enable_company_pages=False,
    )


@app.entrypoint
def invoke(payload: dict, context: Optional[object] = None) -> dict:
    """
    AgentCore entrypoint. One invocation = one candidate's job-match run.

    payload:
      resume        (str, required)  raw resume text
      search_terms  (list[str])      override target roles
      locations     (list[str])      override target locations
      min_score     (int)            match threshold, default from config
      jobs          (list[dict])     optional pre-retrieved postings
      top_k         (int)            KB retrieval size

    returns a JSON-serialisable summary: matches + tailored application kits.
    """
    # AgentCore passes the raw JSON payload; accept a couple of key aliases.
    resume_text = (payload or {}).get("resume") or (payload or {}).get("prompt")
    if not resume_text:
        return {"error": "payload must include a 'resume' field with resume text"}

    # Import the agent nodes lazily so module import stays cheap on cold start.
    from agents.matcher_agent import matcher_node
    from agents.tailor_agent import tailor_node
    from review.autogen_crew import review_node

    logger.info("AgentCore invocation started — parsing resume")
    resume = parse_resume_with_llm(resume_text)
    resume.raw_text = resume_text

    jobs = _get_jobs(resume, payload)
    if not jobs:
        return {"matches": [], "kits": [], "status": "No job postings found for this run"}

    # Run the crew: match → tailor → review. (Persistence/notifications are
    # handled by AgentCore Memory + the caller, so we skip the local applier.)
    state = {
        "resume": resume,
        "jobs": jobs,
        "min_score": payload.get("min_score", MATCH_THRESHOLD),
        "errors": [],
    }
    state.update(matcher_node(state))
    state.update(tailor_node(state))
    state.update(review_node(state))

    matches = state.get("matches", [])
    return {
        "status": state.get("status", "done"),
        "jobs_considered": len(jobs),
        "matches": [
            {
                "title": m.job.title,
                "company": m.job.company,
                "location": m.job.location,
                "score": m.score_percent,
                "url": m.job.url,
            }
            for m in matches
        ],
        # ApplicationKit is a TypedDict (plain dict) — already JSON-serialisable.
        "kits": state.get("kits", []),
        "errors": state.get("errors", []),
    }


if __name__ == "__main__":
    # `python job_hunt_agent.py` starts the AgentCore dev server locally so you
    # can `agentcore invoke` against it before deploying.
    app.run()
