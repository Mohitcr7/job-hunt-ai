# tests/test_enrichment.py
#
# Unit tests for company-page description enrichment. No network calls: the
# quality gate is a pure function, and the fetch path is exercised with a
# monkeypatched requests module.
#
# What matters here is the asymmetry — a placeholder description scores
# predictably low, but a *wrong* description looks like signal and silently
# corrupts matching. So these tests lean on the rejection cases.

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# A real posting: long paragraphs, no results count, title words present.
REAL_POSTING = """Back to jobs
Account Executive, Public Sector
Sydney, Australia
Apply

About the Role

As an Australia Public Sector Account Executive you'll drive the adoption of safe, frontier AI within the Australian government space, establishing and growing our presence in this strategic market and becoming a trusted partner to customers.

Responsibilities

Lead our market entry and expansion in the Australian public sector, establishing our presence and building durable relationships with agency stakeholders across federal and state government.

You may be a good fit if you have significant experience selling technical platforms to government buyers, can navigate long procurement cycles with patience and rigour, and are comfortable being the first person on the ground in a new territory.

Partner closely with go-to-market, product and marketing colleagues to define our market entry strategy and articulate a value proposition that holds up to the scrutiny of security-conscious public sector buyers with strict compliance requirements.

Represent customer needs internally, translating what you hear in the field into concrete product feedback so that the roadmap reflects the realities of deploying frontier AI inside government departments.

Deadline to apply: none. Applications will be reviewed on a rolling basis, and we encourage candidates from non-traditional backgrounds to apply even where they do not meet every listed requirement.
"""

# A career site whose description renders client-side: nav chrome only.
NAV_CHROME = "\n".join([
    "Skip to main content", "Careers", "Locations", "Professions",
    "Programs", "Sign in", "Jobs", "Search jobs", "Job cart", "0",
    "Work site", "Experience level", "All filters", "Clear All",
    "Job description", "Company and benefits", "Top skills",
])

# A results page we got bounced back to, announcing its own size.
LISTING_PAGE = REAL_POSTING + "\n1741 jobs\nSort: Latest\n"


def test_accepts_a_real_posting():
    from tools.scraper_tool import _looks_like_description
    assert _looks_like_description(REAL_POSTING, "Account Executive, Public Sector")


def test_rejects_nav_chrome_when_description_renders_client_side():
    from tools.scraper_tool import _looks_like_description
    assert not _looks_like_description(NAV_CHROME, "Software Engineer")


def test_rejects_a_search_results_page():
    """A job link that redirects back to the board must not be substituted in."""
    from tools.scraper_tool import _looks_like_description
    assert not _looks_like_description(LISTING_PAGE, "Account Executive, Public Sector")


def test_rejects_a_page_about_a_different_job():
    """Right shape, wrong posting — the card's title should appear in its page."""
    from tools.scraper_tool import _looks_like_description
    assert not _looks_like_description(REAL_POSTING, "Senior Kubernetes Platform Reliability")


def test_short_titles_do_not_gate_on_relevance():
    """Titles with no significant words fall back to the structural checks."""
    from tools.scraper_tool import _looks_like_description
    assert _looks_like_description(REAL_POSTING, "SDE II")


def test_enrichment_replaces_placeholder(monkeypatch):
    import tools.scraper_tool as st

    monkeypatch.setattr(st, "ENABLE_PAGE_ENRICHMENT", True)
    fake_requests = SimpleNamespace(
        get=lambda *a, **k: SimpleNamespace(status_code=200, text=REAL_POSTING)
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    from tools.vector_store_tool import Job
    job = Job(
        title="Account Executive, Public Sector", company="Acme", location="Sydney",
        description="Account Executive, Public Sector position at Acme",
        url="https://example.test/jobs/1", platform="company_page",
    )
    assert st._enrich_descriptions([job]) == 1
    assert "Responsibilities" in job.description


def test_enrichment_keeps_placeholder_when_fetch_fails(monkeypatch):
    """A dead link, a 404 or a timeout must leave the job untouched, not blank."""
    import tools.scraper_tool as st

    monkeypatch.setattr(st, "ENABLE_PAGE_ENRICHMENT", True)

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=boom))

    from tools.vector_store_tool import Job
    placeholder = "Software Engineer position at Acme"
    job = Job(
        title="Software Engineer", company="Acme", location="Remote",
        description=placeholder, url="https://example.test/jobs/2",
        platform="company_page",
    )
    assert st._enrich_descriptions([job]) == 0
    assert job.description == placeholder


def test_rate_limit_aborts_the_pass(monkeypatch):
    """One 429 stops the whole pass rather than hammering a free service."""
    import tools.scraper_tool as st

    monkeypatch.setattr(st, "ENABLE_PAGE_ENRICHMENT", True)
    monkeypatch.setattr(st, "ENRICH_CONCURRENCY", 1)
    calls = []

    def rate_limited(url, **k):
        calls.append(url)
        return SimpleNamespace(status_code=429, text="")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=rate_limited))

    from tools.vector_store_tool import Job
    jobs = [
        Job(title=f"Job {i}", company="Acme", location="Remote",
            description=f"Job {i} position at Acme", url=f"https://example.test/jobs/{i}",
            platform="company_page")
        for i in range(5)
    ]
    assert st._enrich_descriptions(jobs) == 0
    assert len(calls) < len(jobs), "enrichment kept fetching after being rate-limited"


def test_disabled_by_flag(monkeypatch):
    import tools.scraper_tool as st

    monkeypatch.setattr(st, "ENABLE_PAGE_ENRICHMENT", False)

    def boom(*a, **k):
        raise AssertionError("should not fetch when enrichment is disabled")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=boom))

    from tools.vector_store_tool import Job
    job = Job(title="X", company="Acme", location="Remote", description="X position at Acme",
              url="https://example.test/jobs/3", platform="company_page")
    assert st._enrich_descriptions([job]) == 0
