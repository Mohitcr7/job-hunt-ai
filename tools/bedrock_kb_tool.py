# tools/bedrock_kb_tool.py
#
# WHAT THIS FILE DOES:
# When deployed on Amazon Bedrock AgentCore, job postings live in a Bedrock
# **Managed Knowledge Base** (S3 → auto chunk/embed/index) instead of the local
# FAISS index. This module retrieves the postings most relevant to a resume
# from that KB — the "R" in RAG, but managed by AWS.
#
# It's the drop-in replacement for the Stage-1 FAISS pre-filter in
# tools/vector_store_tool.py: same job (semantic shortlist), different backend.
#
# boto3 is imported lazily so the rest of the project runs without AWS deps.

from typing import List, Optional

from loguru import logger

from config import BEDROCK_REGION, MANAGED_KB_ID
from tools.vector_store_tool import Job


def _location_uri(location: dict) -> str:
    """Best-effort extraction of a source URI from a retrieval result's location."""
    if not location:
        return ""
    for key in ("s3Location", "webLocation", "confluenceLocation"):
        loc = location.get(key)
        if isinstance(loc, dict):
            return loc.get("uri") or loc.get("url") or ""
    return ""


def retrieve_jobs_from_kb(
    query: str,
    top_k: int = 15,
    kb_id: Optional[str] = None,
    region: Optional[str] = None,
) -> List[Job]:
    """
    Semantic-search the Managed Knowledge Base for postings matching `query`
    (typically the resume summary + skills) and return them as Job objects.

    Mirrors JobVectorStore.find_matching_jobs() so the matcher agent can consume
    the results identically — the only difference is retrieval happens in Bedrock.
    """
    kb_id = kb_id or MANAGED_KB_ID
    if not kb_id:
        logger.warning("retrieve_jobs_from_kb called but MANAGED_KB_ID is not set")
        return []

    import boto3  # lazy — only needed on the Bedrock deployment path

    client = boto3.client("bedrock-agent-runtime", region_name=region or BEDROCK_REGION)

    resp = client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query[:1000]},  # Bedrock caps the query length
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": top_k}
        },
    )

    jobs: List[Job] = []
    for result in resp.get("retrievalResults", []):
        text = (result.get("content") or {}).get("text", "")
        # KB metadata is whatever you attached in the S3 ingest (title, company,
        # url…). We read it defensively and fall back to the source location.
        md = result.get("metadata") or {}
        url = md.get("url") or md.get("job_url") or _location_uri(result.get("location") or {})

        jobs.append(Job(
            title=str(md.get("title", "(from Knowledge Base)")),
            company=str(md.get("company", "")),
            location=str(md.get("location", "")),
            description=text,
            url=str(url or ""),
            platform="bedrock_kb",
            posted_date=str(md.get("posted_date", "")),
            salary=str(md.get("salary", "")),
            job_id=str(md.get("id", url or "")),
        ))

    logger.info(f"Bedrock KB retrieved {len(jobs)} postings for the resume query")
    return jobs
