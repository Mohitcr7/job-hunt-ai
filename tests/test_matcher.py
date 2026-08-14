# tests/test_matcher.py
#
# Unit tests for the three-stage matching funnel.
#
# No network, no LLM, no torch. The embedding model is replaced with a
# deterministic bag-of-words encoder, so FAISS still does real vector maths and
# the assertions are about real ordering — but the tests run in a second and
# work in CI, which installs numpy and faiss-cpu without sentence-transformers.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Small enough to reason about by hand: each dimension is one keyword count.
VOCAB = [
    "python", "pytorch", "machine", "learning", "retrieval",
    "nursing", "patient", "kitchen", "audit", "kubernetes",
]


class _StubEncoder:
    """Counts vocabulary words. Similar texts land near each other, as intended."""

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        matrix = np.array(
            [[float(text.lower().count(word)) for word in VOCAB] for text in items],
            dtype="float32",
        )
        # A row of zeros would become NaN under L2 normalisation, which is a
        # property of the stub, not of the system under test.
        matrix += 0.01
        return matrix[0] if single else matrix


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    import tools.vector_store_tool as vs

    monkeypatch.setattr(vs, "_embedding_model", _StubEncoder())
    yield


def _resume(summary="python pytorch machine learning retrieval", skills=None):
    from tools.resume_parser_tool import ParsedResume

    return ParsedResume(
        summary=summary,
        skills=skills or ["python", "pytorch", "retrieval"],
        raw_text=summary,
    )


def _job(title, description, company="Acme", url=None):
    from tools.vector_store_tool import Job

    return Job(
        title=title, company=company, location="Remote",
        description=description, url=url or f"https://example.test/{title}",
        platform="eval",
    )


# --- stage 1: FAISS prefilter -----------------------------------------------

def test_prefilter_ranks_relevant_jobs_above_irrelevant_ones():
    from agents.matcher_agent import faiss_prefilter

    jobs = [
        _job("Nurse", "nursing patient patient nursing care"),
        _job("ML Engineer", "python pytorch machine learning retrieval"),
        _job("Chef", "kitchen kitchen kitchen menu"),
    ]
    matches = faiss_prefilter(_resume(), jobs, min_score=0, top_k=len(jobs))

    assert matches[0].job.title == "ML Engineer"
    assert [m.job.title for m in matches[1:]] != ["ML Engineer"]


def test_prefilter_returns_scores_in_descending_order():
    from agents.matcher_agent import faiss_prefilter

    jobs = [
        _job("Partial", "python audit audit"),
        _job("Strong", "python pytorch machine learning retrieval"),
        _job("None", "kitchen audit nursing"),
    ]
    matches = faiss_prefilter(_resume(), jobs, min_score=0, top_k=len(jobs))
    scores = [m.score for m in matches]

    assert scores == sorted(scores, reverse=True)


def test_prefilter_drops_everything_below_min_score():
    from agents.matcher_agent import faiss_prefilter

    jobs = [
        _job("Strong", "python pytorch machine learning retrieval"),
        _job("Unrelated", "kitchen kitchen nursing audit"),
    ]
    everything = faiss_prefilter(_resume(), jobs, min_score=0, top_k=2)
    filtered = faiss_prefilter(_resume(), jobs, min_score=90, top_k=2)

    assert len(everything) == 2
    assert len(filtered) < 2
    assert all(m.score_percent >= 90 for m in filtered)


def test_prefilter_on_an_empty_index_returns_nothing():
    """A scrape that found nothing must not blow up the matcher."""
    from agents.matcher_agent import faiss_prefilter

    assert faiss_prefilter(_resume(), [], min_score=0, top_k=5) == []


# --- stage 2: reranker ------------------------------------------------------

def test_rerank_keeps_faiss_order_when_disabled(monkeypatch):
    import config
    from agents.matcher_agent import faiss_prefilter, rerank_candidates

    monkeypatch.setattr(config, "ENABLE_RERANK", False)
    jobs = [_job(f"Job {i}", "python pytorch retrieval") for i in range(5)]
    candidates = faiss_prefilter(_resume(), jobs, min_score=0, top_k=5)

    assert rerank_candidates(_resume(), candidates) == candidates[:len(candidates)]


def test_rerank_of_nothing_is_nothing():
    from agents.matcher_agent import rerank_candidates

    assert rerank_candidates(_resume(), []) == []


# --- stage 3: LLM fit scoring -----------------------------------------------

def _candidates(n=3, score_percent=50):
    from tools.vector_store_tool import JobMatch

    return [
        JobMatch(job=_job(f"Job {i}", "python pytorch", url=f"https://example.test/{i}"),
                 score=score_percent / 100, score_percent=score_percent)
        for i in range(n)
    ]


def test_llm_score_replaces_the_embedding_score(monkeypatch):
    import agents.matcher_agent as ma
    from agents.matcher_agent import FitAssessment

    monkeypatch.setattr(ma, "llm_score_job",
                        lambda resume, job, years: FitAssessment(score=88, reasoning="fits"))

    matches = ma.llm_rerank(_resume(), _candidates(3), min_score=70, experience_years=1)

    assert len(matches) == 3
    assert all(m.score_percent == 88 for m in matches)


def test_llm_score_below_threshold_is_dropped(monkeypatch):
    import agents.matcher_agent as ma
    from agents.matcher_agent import FitAssessment

    scores = iter([95, 40, 80])
    monkeypatch.setattr(ma, "llm_score_job",
                        lambda resume, job, years: FitAssessment(score=next(scores)))

    matches = ma.llm_rerank(_resume(), _candidates(3), min_score=70, experience_years=1)

    assert sorted(m.score_percent for m in matches) == [80, 95]


def test_a_failed_llm_call_keeps_the_embedding_score(monkeypatch):
    """
    A rate limit mid-run must degrade the ranking, not lose the job.

    The candidate scored 75 by embedding, above the threshold, so it survives
    with that score rather than vanishing because one HTTP call failed.
    """
    import agents.matcher_agent as ma

    def boom(resume, job, years):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(ma, "llm_score_job", boom)
    errors = []

    matches = ma.llm_rerank(_resume(), _candidates(2, score_percent=75),
                            min_score=70, experience_years=1, errors=errors)

    assert len(matches) == 2
    assert all(m.score_percent == 75 for m in matches)
    assert len(errors) == 2


def test_a_failed_llm_call_below_threshold_is_still_dropped(monkeypatch):
    import agents.matcher_agent as ma

    def boom(resume, job, years):
        raise RuntimeError("timeout")

    monkeypatch.setattr(ma, "llm_score_job", boom)

    matches = ma.llm_rerank(_resume(), _candidates(2, score_percent=30),
                            min_score=70, experience_years=1)

    assert matches == []


def test_matches_come_back_sorted_by_score(monkeypatch):
    import agents.matcher_agent as ma
    from agents.matcher_agent import FitAssessment

    scores = iter([72, 96, 84])
    monkeypatch.setattr(ma, "llm_score_job",
                        lambda resume, job, years: FitAssessment(score=next(scores)))

    matches = ma.llm_rerank(_resume(), _candidates(3), min_score=70, experience_years=1)

    assert [m.score_percent for m in matches] == [96, 84, 72]


# --- the node itself --------------------------------------------------------

def test_matcher_node_short_circuits_without_jobs():
    from agents.matcher_agent import matcher_node

    result = matcher_node({"resume": _resume(), "jobs": [], "min_score": 70})

    assert result["matches"] == []
    assert "nothing to match" in result["status"]


def test_matcher_node_short_circuits_without_a_resume():
    from agents.matcher_agent import matcher_node

    result = matcher_node({"resume": None, "jobs": [_job("X", "python")], "min_score": 70})

    assert result["matches"] == []


def test_fit_assessment_rejects_a_missing_score():
    from pydantic import ValidationError
    from agents.matcher_agent import FitAssessment

    with pytest.raises(ValidationError):
        FitAssessment(reasoning="looks good")
