# tools/scraper_tool.py
#
# WHAT THIS FILE DOES:
# Fetches fresh job postings from three kinds of source and returns them as a
# unified, deduplicated list of Job objects:
#
#   1. LinkedIn + Indeed  → jobspy (a specialised search client — reliable)
#   2. Naukri             → Scrapling (JS render + self-healing selectors)
#   3. Company pages      → Scrapling (same, configurable per company)
#
# WHY SCRAPLING FOR (2) AND (3):
# Naukri and company career pages are JavaScript-heavy and change their HTML
# often. The old scraper hard-coded a single CSS selector per field, so the
# day Naukri renamed a class the scraper silently returned zero jobs.
# Scrapling fixes this two ways:
#   - `adaptive` selectors relocate an element by similarity after the site's
#     markup changes (it "self-heals" using a fingerprint saved on the last
#     good run), and
#   - we try a LIST of candidate selectors per field, newest-known first.
# Scrapling's DynamicFetcher is synchronous, so this whole file is now plain
# sequential code — no asyncio.
#
# POLITE BY DESIGN: every source keeps randomised rate-limit delays. We do not
# use Scrapling's anti-bot/stealth fetcher — this tool stays within reasonable,
# respectful scraping.

import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

from loguru import logger

from tools.vector_store_tool import Job

# How many (role × city) searches run at once. Higher = faster, but more likely
# to trip LinkedIn/Indeed rate limits. 4 is a safe-ish balance; tune via .env.
SCRAPE_CONCURRENCY = int(os.getenv("SCRAPE_CONCURRENCY", "4"))


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _clean(value) -> str:
    """
    Normalise a scraped field. jobspy returns pandas values, so a missing
    company arrives as the float NaN whose str() is the literal 'nan' — which
    is why job cards used to show "nan". This turns those into an empty string.
    """
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in ("nan", "none", "null") else text


def _abs_url(href: str, base: str) -> str:
    """Turn a relative '/jobs/123' href into an absolute URL using the page's base."""
    if href and href.startswith("/"):
        p = urlparse(base)
        return f"{p.scheme}://{p.netloc}{href}"
    return href or ""


def _first_text(element, selectors: List[str]) -> str:
    """Return the text of the first selector that matches inside `element`."""
    for sel in selectors:
        try:
            found = element.css(sel)
            if found:
                text = found[0].get_all_text().strip()
                if text:
                    return text
        except Exception:
            continue
    return ""


def _first_href(element, selectors: List[str]) -> str:
    """Return the href of the first anchor selector that matches inside `element`."""
    for sel in selectors:
        try:
            found = element.css(sel)
            if found:
                href = found[0].attrib.get("href", "")
                if href:
                    return href
        except Exception:
            continue
    return ""


def _find_cards(page, selectors: List[str]):
    """
    Find job-card elements, resiliently:
      1. try each known selector, saving a fingerprint of whatever matches so
         Scrapling can relocate it later (auto_save), then
      2. if none match (markup changed), ask Scrapling to self-heal using the
         fingerprint saved on a previous good run (adaptive).
    Falls back to a plain .css() call on Scrapling versions that don't accept
    the auto_save/adaptive keywords.
    """
    for sel in selectors:
        try:
            cards = page.css(sel, auto_save=True)
        except TypeError:
            try:
                cards = page.css(sel)
            except Exception:
                continue
        except Exception:
            continue
        if cards:
            return cards

    for sel in selectors:  # self-healing relocation
        try:
            cards = page.css(sel, adaptive=True)
            if cards:
                logger.info(f"Relocated job cards via self-healing selector: {sel}")
                return cards
        except Exception:
            continue
    return []


# ---------------------------------------------------------------------------
# SECTION 1: jobspy-based scraping (LinkedIn + Indeed) — the reliable source
# ---------------------------------------------------------------------------

def _jobspy_one_search(
    term: str, location: str, hours_old: int, results_per_term: int
) -> List[Job]:
    """
    Runs ONE jobspy query (one role in one city) and returns its jobs.
    This is the unit of work the thread pool parallelises.
    """
    from jobspy import scrape_jobs

    logger.info(f"Searching: '{term}' in '{location}'")
    # Small random stagger so N workers don't hammer the API in the same instant.
    time.sleep(random.uniform(0.0, 1.5))

    jobs: List[Job] = []
    try:
        df = scrape_jobs(
            site_name=["linkedin", "indeed"],
            search_term=term,
            location=location,
            results_wanted=results_per_term,
            hours_old=hours_old,
            country_indeed="India",
            linkedin_fetch_description=True,
        )
    except Exception as e:
        logger.error(f"jobspy failed for '{term}' in '{location}': {e}")
        return jobs

    if df is None or df.empty:
        logger.warning(f"No results for '{term}' in '{location}'")
        return jobs

    logger.info(f"Found {len(df)} raw results for '{term}' in '{location}'")
    for _, row in df.iterrows():
        url = _clean(row.get("job_url"))
        if not url:
            continue
        description = _clean(row.get("description"))
        if not description:
            continue  # can't match without a description
        jobs.append(Job(
            title=_clean(row.get("title")),
            company=_clean(row.get("company")),
            location=_clean(row.get("location")) or location,
            description=description,
            url=url,
            platform=_clean(row.get("site")) or "unknown",
            posted_date=_clean(row.get("date_posted")),
            salary=_clean(row.get("min_amount")),
            job_id=_clean(row.get("id")) or url,
        ))
    return jobs


def scrape_with_jobspy(
    search_terms: List[str],
    locations: List[str],
    hours_old: int = 24,
    results_per_term: int = 20,
) -> List[Job]:
    """
    Fetches jobs from LinkedIn + Indeed for every (role × city) combination.

    Searches run CONCURRENTLY (bounded by SCRAPE_CONCURRENCY) instead of one at
    a time, which cuts wall-clock latency roughly by the concurrency factor.
    Results are merged and de-duplicated by URL back in the main thread, so the
    parallelism never causes a data race.
    """
    try:
        import jobspy  # noqa: F401 — fail fast with a clear message if missing
    except ImportError:
        logger.error("jobspy not installed. Run: pip install python-jobspy")
        return []

    pairs = [(term, loc) for term in search_terms for loc in locations]
    all_jobs: List[Job] = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=SCRAPE_CONCURRENCY) as pool:
        futures = {
            pool.submit(_jobspy_one_search, term, loc, hours_old, results_per_term): (term, loc)
            for (term, loc) in pairs
        }
        # as_completed yields each search the moment it finishes, so we merge
        # results (and stream progress) as they arrive rather than in order.
        for future in as_completed(futures):
            term, loc = futures[future]
            try:
                for job in future.result():
                    if job.url and job.url not in seen_urls:
                        seen_urls.add(job.url)
                        all_jobs.append(job)
            except Exception as e:
                logger.error(f"Search worker failed for '{term}' in '{loc}': {e}")

    logger.info(
        f"jobspy total: {len(all_jobs)} unique jobs from {len(pairs)} searches "
        f"(up to {SCRAPE_CONCURRENCY} in parallel)"
    )
    return all_jobs


# ---------------------------------------------------------------------------
# SECTION 2: Naukri (Scrapling — JS render + self-healing selectors)
# ---------------------------------------------------------------------------
# Candidate selectors, newest-known first. Scrapling tries each in order and
# self-heals if all of them drift, so Naukri renaming a class no longer zeroes
# out the scraper.

NAUKRI_CARD_SELECTORS = [
    "div.srp-jobtuple-wrapper",
    "div.cust-job-tuple",
    "article.jobTuple",
]
NAUKRI_TITLE_SELECTORS = ["a.title", "a.jobTupleHeader", "a[class*='title']"]
NAUKRI_COMPANY_SELECTORS = ["a.comp-name", "a.subTitle", "span.comp-name", "a[class*='comp']"]
NAUKRI_LOCATION_SELECTORS = ["span.locWdth", "li.location span", "span[class*='loc']"]
NAUKRI_SALARY_SELECTORS = ["span.sal-wrap span", "li.salary span", "span[class*='sal']"]
NAUKRI_DESC_SELECTORS = ["span.job-desc", "div.job-description", "[class*='job-desc']"]


def scrape_naukri(
    search_terms: List[str],
    locations: List[str],
    max_pages: int = 2,
) -> List[Job]:
    """
    Scrapes Naukri using Scrapling's DynamicFetcher (headless browser, so the
    JS-rendered listings actually load). One browser session is reused across
    all pages for speed.
    """
    try:
        from scrapling.fetchers import DynamicSession
    except ImportError:
        logger.warning(
            "Scrapling browser fetcher unavailable — skipping Naukri (LinkedIn/Indeed "
            "still work). Enable with: pip install 'scrapling[fetchers]' && scrapling install"
        )
        return []

    all_jobs: List[Job] = []
    seen_urls = set()

    try:
        with DynamicSession(headless=True, network_idle=True) as session:
            for term in search_terms:
                for location in locations:
                    logger.info(f"Naukri: scraping '{term}' in '{location}'")

                    for page_num in range(1, max_pages + 1):
                        term_slug = term.lower().replace(" ", "-")
                        loc_slug = location.lower().replace(" ", "-")
                        url = (
                            f"https://www.naukri.com/{term_slug}-jobs-in-{loc_slug}"
                            f"?jobAge=1&pageNo={page_num}"
                        )

                        try:
                            page = session.fetch(url)
                        except Exception as e:
                            logger.error(f"Naukri fetch failed for {url}: {e}")
                            break

                        cards = _find_cards(page, NAUKRI_CARD_SELECTORS)
                        if not cards:
                            logger.warning(
                                f"Naukri: no job cards for '{term}' in '{location}' "
                                f"(page {page_num}). The layout may have changed, or "
                                f"this IP is being blocked — LinkedIn/Indeed still work."
                            )
                            break

                        logger.info(f"Found {len(cards)} cards on page {page_num}")

                        for card in cards:
                            href = _first_href(card, NAUKRI_TITLE_SELECTORS)
                            if not href or href in seen_urls:
                                continue
                            seen_urls.add(href)

                            title = _first_text(card, NAUKRI_TITLE_SELECTORS)
                            company = _first_text(card, NAUKRI_COMPANY_SELECTORS)
                            description = (
                                _first_text(card, NAUKRI_DESC_SELECTORS)
                                or f"{title} at {company}"
                            )

                            all_jobs.append(Job(
                                title=title,
                                company=company,
                                location=_first_text(card, NAUKRI_LOCATION_SELECTORS) or location,
                                description=description,
                                url=href,
                                platform="naukri",
                                posted_date=datetime.now().strftime("%Y-%m-%d"),
                                salary=_first_text(card, NAUKRI_SALARY_SELECTORS),
                                job_id=href,
                            ))

                        time.sleep(random.uniform(2.0, 4.0))  # stay polite

    except Exception as e:
        logger.error(f"Naukri scraping error: {e}")

    logger.info(f"Naukri total: {len(all_jobs)} jobs collected")
    return all_jobs


# ---------------------------------------------------------------------------
# SECTION 3: Company career pages (Scrapling)
# ---------------------------------------------------------------------------
# Each company exposes a few candidate selectors; Scrapling tries them in order
# and self-heals. To add a company, copy a block and fill in its selectors
# (inspect the careers page in your browser's dev tools).

COMPANY_CAREER_PAGES = [
    {
        "company": "Google",
        "url": "https://careers.google.com/jobs/results/?q={term}&location={location}",
        "card_selectors": ["li.lLd3Je", "div[jsname]"],
        "title_selectors": ["h3.QJPWVe", "h2.QJPWVe", "[class*='title']"],
        "link_selectors": ["a.WpHeLc", "a[href*='/jobs/results/']"],
    },
    {
        "company": "Microsoft",
        "url": "https://jobs.careers.microsoft.com/global/en/search?q={term}&lc={location}",
        "card_selectors": ["div.ms-List-cell", "div[class*='jobCard']"],
        "title_selectors": ["span[class*='title']", "h2"],
        "link_selectors": ["a[href*='/job/']", "a[aria-label]"],
    },
]


def scrape_company_pages(
    search_terms: List[str],
    locations: List[str],
) -> List[Job]:
    """Scrapes the configured company career pages with Scrapling."""
    try:
        from scrapling.fetchers import DynamicSession
    except ImportError:
        logger.warning(
            "Scrapling browser fetcher unavailable — skipping company pages. "
            "Enable with: pip install 'scrapling[fetchers]' && scrapling install"
        )
        return []

    all_jobs: List[Job] = []
    seen_urls = set()

    try:
        with DynamicSession(headless=True, network_idle=True) as session:
            for config in COMPANY_CAREER_PAGES:
                for term in search_terms[:2]:       # cap per company
                    for location in locations[:2]:
                        url = config["url"].format(
                            term=term.replace(" ", "+"),
                            location=location.replace(" ", "+"),
                        )
                        logger.info(f"Scraping {config['company']} careers: {term} in {location}")

                        try:
                            page = session.fetch(url)
                        except Exception as e:
                            logger.error(f"Failed to fetch {config['company']}: {e}")
                            continue

                        cards = _find_cards(page, config["card_selectors"])
                        logger.info(f"{config['company']}: found {len(cards)} cards")

                        for card in cards[:10]:
                            href = _first_href(card, config["link_selectors"])
                            if not href:
                                continue
                            href = _abs_url(href, url)
                            if href in seen_urls:
                                continue
                            seen_urls.add(href)

                            title = _first_text(card, config["title_selectors"])
                            all_jobs.append(Job(
                                title=title,
                                company=config["company"],
                                location=location,
                                description=f"{title} position at {config['company']}",
                                url=href,
                                platform="company_page",
                                posted_date=datetime.now().strftime("%Y-%m-%d"),
                                job_id=href,
                            ))

                        time.sleep(random.uniform(2.0, 4.0))

    except Exception as e:
        logger.error(f"Company page scraping error: {e}")

    logger.info(f"Company pages total: {len(all_jobs)} jobs")
    return all_jobs


# ---------------------------------------------------------------------------
# SECTION 4: Master scraper — calls all sources and merges results
# ---------------------------------------------------------------------------

def scrape_all_jobs(
    search_terms: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    hours_old: int = 24,
    enable_naukri: bool = True,
    enable_company_pages: bool = False,
) -> List[Job]:
    """
    The single function the Scout Agent calls. Runs every enabled source and
    returns one merged, deduplicated list. jobspy is always on (the reliable
    source); Naukri and company pages are best-effort via Scrapling.
    """
    from config import JOB_PREFERENCES

    if search_terms is None:
        search_terms = JOB_PREFERENCES["roles"]
    if locations is None:
        locations = JOB_PREFERENCES["locations"]

    logger.info("=" * 50)
    logger.info(f"Starting job scrape for: {search_terms}")
    logger.info(f"Locations: {locations}")
    logger.info(f"Fetching jobs posted in last {hours_old} hours")
    logger.info("=" * 50)

    all_jobs: List[Job] = []

    logger.info("\n--- Source 1: LinkedIn + Indeed (jobspy) ---")
    jobspy_jobs = scrape_with_jobspy(search_terms, locations, hours_old)
    all_jobs.extend(jobspy_jobs)
    logger.info(f"jobspy contributed {len(jobspy_jobs)} jobs")

    if enable_naukri:
        logger.info("\n--- Source 2: Naukri (Scrapling) ---")
        naukri_jobs = scrape_naukri(search_terms, locations)
        all_jobs.extend(naukri_jobs)
        logger.info(f"Naukri contributed {len(naukri_jobs)} jobs")

    if enable_company_pages:
        logger.info("\n--- Source 3: Company career pages (Scrapling) ---")
        company_jobs = scrape_company_pages(search_terms, locations)
        all_jobs.extend(company_jobs)
        logger.info(f"Company pages contributed {len(company_jobs)} jobs")

    # Final dedup by URL across all sources
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        if job.url and job.url not in seen:
            seen.add(job.url)
            unique_jobs.append(job)

    logger.info(f"\nTotal unique jobs collected: {len(unique_jobs)}")
    return unique_jobs


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Running scraper test...")
    jobs = scrape_all_jobs(
        search_terms=["Data Scientist", "ML Engineer"],
        locations=["Bangalore", "Remote"],
        hours_old=48,
        enable_naukri=True,
        enable_company_pages=False,
    )

    print(f"\n=== SCRAPE RESULTS: {len(jobs)} jobs ===")
    for job in jobs[:5]:
        print(f"\n{job.title} — {job.company}")
        print(f"  Platform : {job.platform}")
        print(f"  Location : {job.location}")
        print(f"  URL      : {job.url[:80]}")
