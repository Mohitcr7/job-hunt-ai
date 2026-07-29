# tools/reranker_tool.py
#
# WHAT THIS FILE DOES:
# A dedicated relevance reranker for the matcher. After FAISS gives a broad
# semantic shortlist, this narrows it to the most relevant jobs using NVIDIA's
# Llama-Nemotron reranker via OpenRouter's /rerank endpoint — a single batched
# call that scores every candidate against the resume at once.
#
# WHY A RERANKER, NOT THE GENERATION LLM:
# A cross-encoder reranker judges relevance far more sharply than the FAISS
# bi-encoder, in ONE cheap call — and it runs on a FREE, separate budget, so it
# doesn't burn the generation LLM's rate limit that tailoring needs. It does NOT
# replace the LLM fit-scoring (which reasons about hard requirements like
# "8+ years"); it just feeds the LLM a shorter, better-ordered shortlist.
#
# Fails soft: no API key or any error → returns the FAISS order unchanged, so
# the pipeline always works with or without the reranker.

import os
from typing import List

from loguru import logger

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, RERANK_MODEL_ID
from tools.vector_store_tool import JobMatch

# The endpoint rejects more than 512 passages per request:
#   "List should have at most 512 items after validation, not 1000"
# Measured latency at the ceiling is ~9s, so chunking a full day's scrape into
# a couple of calls is cheap. Left a margin under the documented limit.
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "400"))


def _document_for(match: JobMatch) -> str:
    """The text the reranker judges — title and company first, they carry signal."""
    return f"{match.job.title} at {match.job.company}. {match.job.description[:1500]}"


def _rerank_batch(query: str, documents: List[str]) -> List[dict]:
    """POST one batch of ≤512 documents. Returns [] on any failure."""
    import requests

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/rerank",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "Job Hunt AI",
            },
            json={
                "model": RERANK_MODEL_ID,
                "query": query[:4000],
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        logger.warning(f"Rerank batch of {len(documents)} failed: {e}")
        return []


def score_all_jobs(query: str, candidates: List[JobMatch]) -> List[JobMatch]:
    """
    Score EVERY candidate with the cross-encoder and return them sorted.

    Different job from `rerank_jobs`: that one narrows a shortlist for the
    pipeline and throws the scores away. This keeps the score on each JobMatch,
    because the daily sheet shows a fit band for every row.

    Safe to chunk. The model scores each (query, document) pair independently —
    verified by scoring the same posting in batches with different neighbours and
    getting bit-identical results — so a job's score doesn't depend on what it
    was batched with. A listwise reranker would need the opposite treatment.

    Fails soft: any batch that errors keeps its embedding scores, so a rate limit
    degrades ranking quality instead of losing the run.
    """
    if not candidates:
        return candidates
    if not OPENROUTER_API_KEY:
        logger.info("Reranker: no OPENROUTER_API_KEY set — keeping embedding order")
        return candidates

    scored = []
    for start in range(0, len(candidates), RERANK_BATCH_SIZE):
        batch = candidates[start:start + RERANK_BATCH_SIZE]
        results = _rerank_batch(query, [_document_for(c) for c in batch])
        for result in results:
            index = result.get("index")
            score = result.get("relevance_score")
            if isinstance(index, int) and 0 <= index < len(batch) and score is not None:
                batch[index].score = float(score)
                scored.append(batch[index])
        logger.info(f"Reranked {start + len(batch)}/{len(candidates)} jobs")

    if not scored:
        logger.warning("Reranker scored nothing — keeping embedding order")
        return candidates

    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)

    # Convert to a percentile rank rather than score × 100. The raw relevance
    # this model returns is tiny and query-dependent — a day's real postings came
    # back inside 0.0003–0.0148, which multiplied by 100 collapses every row to 0
    # or 1 and destroys the ranking. The ordering is sound, the absolute scale
    # isn't, so express each job's position within the day's batch instead.
    total = len(ordered)
    for position, match in enumerate(ordered):
        match.score_percent = int(round(100 * (total - position) / total))

    logger.info(f"Reranker scored {len(scored)}/{len(candidates)} jobs")
    return ordered


def rerank_jobs(query: str, candidates: List[JobMatch], top_n: int = 8) -> List[JobMatch]:
    """
    Reorder `candidates` by relevance to `query` and return the top_n.

    `candidates` are FAISS JobMatch objects — we keep them intact and only
    reorder/narrow. Calls OpenRouter's POST /api/v1/rerank once for the whole
    batch (query + all job documents → per-job relevance scores).
    """
    if not candidates:
        return candidates
    if not OPENROUTER_API_KEY:
        logger.info("Reranker: no OPENROUTER_API_KEY set — keeping FAISS order")
        return candidates[:top_n]

    import requests  # lazy: only needed when the reranker is actually used

    # One document per candidate. The reranker scores each (query, document)
    # pair independently, so the number of documents isn't bounded by context.
    documents = [
        f"{c.job.title} at {c.job.company}. {c.job.description[:1500]}"
        for c in candidates
    ]

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/rerank",  # base is …/api/v1 → …/api/v1/rerank
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "Job Hunt AI",
            },
            json={
                "model": RERANK_MODEL_ID,
                "query": query[:4000],
                "documents": documents,
                "top_n": top_n,
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        logger.warning(f"Reranker failed ({e}) — keeping FAISS order")
        return candidates[:top_n]

    # results: [{"index": <position in documents>, "relevance_score": <float>}, …]
    reranked: List[JobMatch] = []
    for r in results:
        idx = r.get("index")
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            reranked.append(candidates[idx])

    if not reranked:
        logger.warning("Reranker returned no usable results — keeping FAISS order")
        return candidates[:top_n]

    logger.info(f"Reranker: narrowed {len(candidates)} → {len(reranked)} candidates")
    return reranked[:top_n]
