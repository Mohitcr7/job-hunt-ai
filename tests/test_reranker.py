# tests/test_reranker.py
#
# Unit tests for the reranker tool that don't require a network call or API key.
# We monkeypatch requests.post to return a Cohere-style rerank response and
# confirm the candidates are reordered/narrowed correctly, plus verify the
# fail-soft behaviour when the key is absent or the API errors.

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_candidates(n):
    from tools.vector_store_tool import Job, JobMatch
    out = []
    for i in range(n):
        job = Job(
            title=f"Job {i}", company=f"Co {i}", location="Remote",
            description=f"description {i}", url=f"http://x/{i}", platform="linkedin",
        )
        out.append(JobMatch(job=job, score=0.5, score_percent=50))
    return out


def test_rerank_reorders_and_narrows(monkeypatch):
    import config
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")

    import tools.reranker_tool as rt
    monkeypatch.setattr(rt, "OPENROUTER_API_KEY", "test-key")

    # Fake OpenRouter /rerank response: reverse relevance order, indices 3,2,1,0
    class FakeResp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"results": [
                {"index": 3, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.7},
                {"index": 0, "relevance_score": 0.6},
            ]}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp())

    cands = _make_candidates(4)
    ranked = rt.rerank_jobs("query", cands, top_n=2)

    assert len(ranked) == 2, "top_n should cap the results"
    assert ranked[0].job.title == "Job 3", "should be reordered by relevance"
    assert ranked[1].job.title == "Job 2"


def test_rerank_no_key_falls_back(monkeypatch):
    import tools.reranker_tool as rt
    monkeypatch.setattr(rt, "OPENROUTER_API_KEY", None)

    cands = _make_candidates(5)
    ranked = rt.rerank_jobs("query", cands, top_n=3)

    # No key → keep FAISS order, capped to top_n
    assert [m.job.title for m in ranked] == ["Job 0", "Job 1", "Job 2"]


def test_rerank_api_error_falls_back(monkeypatch):
    import tools.reranker_tool as rt
    monkeypatch.setattr(rt, "OPENROUTER_API_KEY", "test-key")

    import requests
    def boom(*a, **k):
        raise requests.RequestException("network down")
    monkeypatch.setattr(requests, "post", boom)

    cands = _make_candidates(4)
    ranked = rt.rerank_jobs("query", cands, top_n=2)

    # API error → keep FAISS order, never crash
    assert [m.job.title for m in ranked] == ["Job 0", "Job 1"]
