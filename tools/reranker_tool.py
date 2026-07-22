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

from typing import List

from loguru import logger

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, RERANK_MODEL_ID
from tools.vector_store_tool import JobMatch


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
