# tests/test_smoke.py
#
# Lightweight smoke tests that run without heavy ML dependencies,
# network access, or API keys. They cover the tracker (the project's
# only persistence layer) and basic config sanity.
#
# Run with: python -m pytest tests/ -q

import os
import sys
from pathlib import Path

# Make the project root importable when pytest runs from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the tracker at a throwaway database BEFORE importing config
os.environ["DATABASE_URL"] = "sqlite:///./data/test_applications.db"


def _sample_kit(url="https://example.com/job/1"):
    return {
        "job_url": url,
        "job_title": "ML Engineer",
        "company": "TestCorp",
        "match_score": 85,
        "tailored_summary": "A summary.",
        "tailored_bullets": ["Did X", "Built Y"],
        "cover_letter": "Dear team...",
        "keywords_to_add": ["MLOps"],
    }


def setup_module():
    Path("data").mkdir(exist_ok=True)


def teardown_module():
    Path("data/test_applications.db").unlink(missing_ok=True)


def test_config_loads():
    import config
    assert config.MATCH_THRESHOLD > 0
    assert "roles" in config.JOB_PREFERENCES


def test_tracker_roundtrip():
    from review import tracker

    # fresh insert succeeds, duplicate is rejected
    assert tracker.record_application(_sample_kit()) is True
    assert tracker.record_application(_sample_kit()) is False
    assert tracker.is_tracked("https://example.com/job/1")

    apps = tracker.list_applications()
    assert len(apps) >= 1
    app = next(a for a in apps if a["job_url"] == "https://example.com/job/1")
    assert app["status"] == "prepared"
    assert app["tailored_bullets"] == ["Did X", "Built Y"]

    # status lifecycle
    assert tracker.update_status(app["id"], "applied", notes="sent via portal")
    updated = next(
        a for a in tracker.list_applications() if a["id"] == app["id"]
    )
    assert updated["status"] == "applied"
    assert updated["notes"] == "sent via portal"

    # invalid status raises
    try:
        tracker.update_status(app["id"], "not-a-status")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_tracker_stats():
    from review import tracker

    tracker.record_application(_sample_kit("https://example.com/job/2"))
    stats = tracker.get_stats()
    assert stats["total"] >= 2
    assert stats["avg_match_score"] > 0


def test_notifier_formats_summary():
    from tools.notifier_tool import _format_summary

    text = _format_summary([_sample_kit()])
    assert "ML Engineer" in text
    assert "TestCorp" in text
    assert "85" in text
