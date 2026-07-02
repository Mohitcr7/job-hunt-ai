# tools/scraper_tool.py
#
# WHAT THIS FILE DOES:
# Fetches fresh job postings from LinkedIn, Indeed, Naukri, and
# company career pages, then returns them as a unified list of Job objects.
#
# KEY CONCEPTS:
# - jobspy    : a library that queries LinkedIn/Indeed search internally.
#               It mimics what your browser does when you search jobs —
#               but in Python, cleanly, without needing your login.
# - Playwright: a browser automation library. It opens a real Chrome
#               browser invisibly (called "headless" mode) and navigates
#               pages just like a human would — click, scroll, read.
# - Rate limiting : deliberately adding time.sleep() pauses between
#               requests so we don't hammer servers and get IP-banned.
#               Think of it as being a polite visitor.
# - User-Agent : a string your browser sends to websites identifying itself.
#               We set a realistic one so we don't look like a bot.

import time
import random
from datetime import datetime, timedelta
from typing import List, Optional
from loguru import logger  # better than print() for logging — adds timestamps,
                           # log levels (INFO, WARNING, ERROR), and colours

from tools.vector_store_tool import Job  # reuse the Job dataclass we defined

# ---------------------------------------------------------------------------
# SECTION 1: jobspy-based scraping (LinkedIn + Indeed)
# ---------------------------------------------------------------------------
# jobspy returns a pandas DataFrame (think: a spreadsheet in Python).
# Each row is one job listing. We convert each row into our Job object.

def scrape_with_jobspy(
    search_terms: List[str],
    locations: List[str],
    hours_old: int = 24,         # only fetch jobs posted in last N hours
    results_per_term: int = 20,  # how many results per search query
) -> List[Job]:
    """
    Uses jobspy to fetch jobs from LinkedIn and Indeed.

    search_terms : list of role names, e.g. ["Data Scientist", "ML Engineer"]
    locations    : list of cities, e.g. ["Bangalore", "Remote"]
    hours_old    : skip jobs older than this many hours (keeps results fresh)
    """
    try:
        from jobspy import scrape_jobs
        import pandas as pd
    except ImportError:
        logger.error("jobspy not installed. Run: pip install python-jobspy")
        return []

    all_jobs: List[Job] = []
    seen_urls = set()  # for deduplication — avoid adding the same job twice

    for term in search_terms:
        for location in locations:
            logger.info(f"Searching: '{term}' in '{location}'")

            try:
                # scrape_jobs() is jobspy's main function.
                # It returns a pandas DataFrame with columns like:
                # title, company, location, job_url, description, date_posted, etc.
                df = scrape_jobs(
                    site_name=["linkedin", "indeed"],
                    search_term=term,
                    location=location,
                    results_wanted=results_per_term,
                    hours_old=hours_old,
                    country_indeed="India",   # keeps Indeed results India-specific
                    linkedin_fetch_description=True,  # get full JD, not just snippet
                )

                if df is None or df.empty:
                    logger.warning(f"No results for '{term}' in '{location}'")
                    continue

                logger.info(f"Found {len(df)} raw results")

                # Iterate over each row in the DataFrame
                # .iterrows() gives (index, row) pairs
                for _, row in df.iterrows():
                    url = str(row.get("job_url", ""))

                    # Skip if we've already seen this job URL
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Skip if description is missing — can't match without it
                    description = str(row.get("description", ""))
                    if not description or description == "nan":
                        continue

                    # row.get("field", default) safely reads a DataFrame column
                    # — returns the default if the column doesn't exist
                    job = Job(
                        title=str(row.get("title", "")),
                        company=str(row.get("company", "")),
                        location=str(row.get("location", location)),
                        description=description,
                        url=url,
                        platform=str(row.get("site", "unknown")),
                        posted_date=str(row.get("date_posted", "")),
                        salary=str(row.get("min_amount", "")) or "",
                        job_id=str(row.get("id", url)),
                    )
                    all_jobs.append(job)

                # Polite delay between requests — random so it's less bot-like
                # random.uniform(a, b) picks a random float between a and b
                delay = random.uniform(2.0, 4.0)
                logger.info(f"Waiting {delay:.1f}s before next search...")
                time.sleep(delay)

            except Exception as e:
                # Don't crash the whole pipeline if one search fails
                logger.error(f"jobspy failed for '{term}' in '{location}': {e}")
                continue

    logger.info(f"jobspy total: {len(all_jobs)} unique jobs collected")
    return all_jobs


# ---------------------------------------------------------------------------
# SECTION 2: Playwright-based scraping (Naukri)
# ---------------------------------------------------------------------------
# Naukri doesn't have a public API, so we use Playwright to open
# their website in a hidden browser and read the job listings.
#
# "async" and "await" — Playwright uses asynchronous code.
# Think of async like this: instead of waiting for each page to load
# before doing anything else, async code says "start loading this page,
# and while it loads, I can do other things." It's more efficient.
# async def = this function is asynchronous
# await = "pause here until this async operation finishes"

import asyncio  # Python's built-in library for async programming

async def scrape_naukri_async(
    search_terms: List[str],
    locations: List[str],
    max_pages: int = 2,          # how many result pages to go through
    jobs_per_page: int = 20,
) -> List[Job]:
    """
    Scrapes Naukri.com using Playwright in headless Chrome.
    'headless' means the browser runs invisibly — no window opens.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    all_jobs: List[Job] = []
    seen_urls = set()

    # async with = async version of Python's "with" statement
    # It ensures resources (the browser) are properly closed when done
    async with async_playwright() as p:
        # Launch a headless Chrome browser
        browser = await p.chromium.launch(
            headless=True,          # True = invisible; False = visible window (useful for debugging)
            args=["--no-sandbox"],  # required in some Linux environments
        )

        # A "context" is like a separate browser profile — its own cookies,
        # cache, and history. We set a realistic User-Agent so Naukri
        # thinks we're a regular Chrome user on a MacBook.
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )

        page = await context.new_page()

        for term in search_terms:
            for location in locations:
                logger.info(f"Naukri: scraping '{term}' in '{location}'")

                for page_num in range(1, max_pages + 1):
                    # Build the Naukri search URL
                    # %20 = URL-encoded space, - = separator Naukri uses
                    term_slug = term.lower().replace(" ", "-")
                    loc_slug = location.lower().replace(" ", "-")
                    url = (
                        f"https://www.naukri.com/{term_slug}-jobs-in-{loc_slug}"
                        f"?jobAge=1"        # posted in last 1 day
                        f"&pageNo={page_num}"
                    )

                    try:
                        # Navigate to the URL
                        # wait_until="networkidle" = wait until no network
                        # requests are in flight (page is fully loaded)
                        await page.goto(url, wait_until="networkidle", timeout=30000)

                        # Random human-like delay (1.5 to 3 seconds)
                        await asyncio.sleep(random.uniform(1.5, 3.0))

                        # querySelectorAll finds all elements matching a CSS selector.
                        # CSS selectors: "div.jobTuple" means <div class="jobTuple">
                        # Naukri's job cards have class "jobTuple" or "cust-job-tuple"
                        job_cards = await page.query_selector_all(
                            "article.jobTuple, div.cust-job-tuple"
                        )

                        if not job_cards:
                            logger.warning(f"No job cards found on page {page_num}")
                            break

                        logger.info(f"Found {len(job_cards)} cards on page {page_num}")

                        for card in job_cards:
                            try:
                                # inner_text() reads the visible text of an element
                                # query_selector() finds a child element by CSS selector
                                title_el = await card.query_selector("a.title")
                                company_el = await card.query_selector("a.subTitle")
                                location_el = await card.query_selector("li.location span")
                                salary_el = await card.query_selector("li.salary span")

                                title = await title_el.inner_text() if title_el else ""
                                company = await company_el.inner_text() if company_el else ""
                                job_location = await location_el.inner_text() if location_el else location
                                salary = await salary_el.inner_text() if salary_el else ""

                                # get_attribute("href") reads the href attribute of <a> tags
                                job_url = await title_el.get_attribute("href") if title_el else ""

                                if not job_url or job_url in seen_urls:
                                    continue
                                seen_urls.add(job_url)

                                # Get job description snippet from the card
                                desc_el = await card.query_selector("div.job-description")
                                description = await desc_el.inner_text() if desc_el else f"{title} at {company}"

                                job = Job(
                                    title=title.strip(),
                                    company=company.strip(),
                                    location=job_location.strip(),
                                    description=description.strip(),
                                    url=job_url,
                                    platform="naukri",
                                    posted_date=datetime.now().strftime("%Y-%m-%d"),
                                    salary=salary.strip(),
                                    job_id=job_url,
                                )
                                all_jobs.append(job)

                            except Exception as e:
                                logger.warning(f"Failed to parse a card: {e}")
                                continue

                        # Polite delay between pages
                        await asyncio.sleep(random.uniform(2.0, 4.0))

                    except Exception as e:
                        logger.error(f"Naukri page failed: {e}")
                        break

        await browser.close()

    logger.info(f"Naukri total: {len(all_jobs)} jobs collected")
    return all_jobs


def scrape_naukri(search_terms: List[str], locations: List[str]) -> List[Job]:
    """
    Synchronous wrapper around the async Naukri scraper.
    Why? The rest of our pipeline is synchronous (regular Python).
    asyncio.run() creates an event loop and runs the async function
    inside it, then returns the result synchronously.
    This is the standard pattern for calling async code from sync code.
    """
    return asyncio.run(scrape_naukri_async(search_terms, locations))


# ---------------------------------------------------------------------------
# SECTION 3: Company career pages (Playwright)
# ---------------------------------------------------------------------------
# Each company has a different careers page structure, so we define
# a small config per company and use a generic scraper.

COMPANY_CAREER_PAGES = [
    {
        "company": "Google",
        "url": "https://careers.google.com/jobs/results/?q={term}&location={location}",
        "job_card_selector": "li.lLd3Je",
        "title_selector": "h2.QJPWVe",
        "location_selector": "span.r0wTof",
        "link_selector": "a.WpHeLc",
    },
    {
        "company": "Microsoft",
        "url": "https://jobs.microsoft.com/en-us/search?q={term}&lc={location}",
        "job_card_selector": "div.ms-DocumentCard",
        "title_selector": "span.ms-DocumentCard-title",
        "location_selector": "span[data-automation-id='jobLocation']",
        "link_selector": "a[data-automation-id='jobTitle']",
    },
    # Add more companies here as needed
    # The pattern: find their careers page, inspect the HTML to find
    # CSS selectors for job cards, title, location, and link
]

async def scrape_company_pages_async(
    search_terms: List[str],
    locations: List[str],
) -> List[Job]:
    """Scrapes configured company career pages using Playwright."""
    from playwright.async_api import async_playwright

    all_jobs: List[Job] = []
    seen_urls = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        for config in COMPANY_CAREER_PAGES:
            for term in search_terms[:2]:       # limit to first 2 terms per company
                for location in locations[:2]:  # limit to first 2 locations

                    url = config["url"].format(
                        term=term.replace(" ", "+"),
                        location=location.replace(" ", "+"),
                    )
                    logger.info(f"Scraping {config['company']} careers: {term} in {location}")

                    try:
                        await page.goto(url, wait_until="networkidle", timeout=30000)
                        await asyncio.sleep(random.uniform(2.0, 3.5))

                        cards = await page.query_selector_all(config["job_card_selector"])
                        logger.info(f"{config['company']}: found {len(cards)} cards")

                        for card in cards[:10]:  # max 10 per company/term/location
                            try:
                                title_el    = await card.query_selector(config["title_selector"])
                                location_el = await card.query_selector(config["location_selector"])
                                link_el     = await card.query_selector(config["link_selector"])

                                title    = await title_el.inner_text() if title_el else ""
                                loc_text = await location_el.inner_text() if location_el else location
                                job_url  = await link_el.get_attribute("href") if link_el else ""

                                if not job_url or job_url in seen_urls:
                                    continue
                                seen_urls.add(job_url)

                                # Make relative URLs absolute
                                # e.g. "/jobs/123" → "https://careers.google.com/jobs/123"
                                if job_url.startswith("/"):
                                    from urllib.parse import urlparse
                                    parsed = urlparse(url)
                                    job_url = f"{parsed.scheme}://{parsed.netloc}{job_url}"

                                job = Job(
                                    title=title.strip(),
                                    company=config["company"],
                                    location=loc_text.strip(),
                                    description=f"{title} position at {config['company']} in {loc_text}",
                                    url=job_url,
                                    platform="company_page",
                                    posted_date=datetime.now().strftime("%Y-%m-%d"),
                                    job_id=job_url,
                                )
                                all_jobs.append(job)

                            except Exception as e:
                                logger.warning(f"Card parse error at {config['company']}: {e}")
                                continue

                        await asyncio.sleep(random.uniform(2.0, 4.0))

                    except Exception as e:
                        logger.error(f"Failed to scrape {config['company']}: {e}")
                        continue

        await browser.close()

    logger.info(f"Company pages total: {len(all_jobs)} jobs")
    return all_jobs


def scrape_company_pages(search_terms: List[str], locations: List[str]) -> List[Job]:
    """Synchronous wrapper for company page scraping."""
    return asyncio.run(scrape_company_pages_async(search_terms, locations))


# ---------------------------------------------------------------------------
# SECTION 4: Master scraper — calls all sources and merges results
# ---------------------------------------------------------------------------

def scrape_all_jobs(
    search_terms: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    hours_old: int = 24,
    enable_naukri: bool = True,
    enable_company_pages: bool = True,
) -> List[Job]:
    """
    The single function your Scout Agent will call.
    Runs all scrapers and returns one merged, deduplicated list of jobs.

    Optional[List[str]] means the parameter can be either a List[str]
    or None. If None, we fall back to the preferences in config.py.
    """
    from config import JOB_PREFERENCES

    # Use config defaults if not overridden
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

    # 1. LinkedIn + Indeed via jobspy
    logger.info("\n--- Source 1: LinkedIn + Indeed (jobspy) ---")
    jobspy_jobs = scrape_with_jobspy(search_terms, locations, hours_old)
    all_jobs.extend(jobspy_jobs)
    logger.info(f"jobspy contributed {len(jobspy_jobs)} jobs")

    # 2. Naukri via Playwright
    if enable_naukri:
        logger.info("\n--- Source 2: Naukri (Playwright) ---")
        naukri_jobs = scrape_naukri(search_terms, locations)
        all_jobs.extend(naukri_jobs)
        logger.info(f"Naukri contributed {len(naukri_jobs)} jobs")

    # 3. Company career pages
    if enable_company_pages:
        logger.info("\n--- Source 3: Company career pages ---")
        company_jobs = scrape_company_pages(search_terms, locations)
        all_jobs.extend(company_jobs)
        logger.info(f"Company pages contributed {len(company_jobs)} jobs")

    # Final deduplication by URL across all sources
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        if job.url not in seen and job.url:
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
        hours_old=48,           # widen to 48h for testing
        enable_naukri=True,
        enable_company_pages=False,  # skip company pages for quick test
    )

    print(f"\n=== SCRAPE RESULTS: {len(jobs)} jobs ===")
    for job in jobs[:5]:       # print first 5
        print(f"\n{job.title} — {job.company}")
        print(f"  Platform : {job.platform}")
        print(f"  Location : {job.location}")
        print(f"  Posted   : {job.posted_date}")
        print(f"  URL      : {job.url[:80]}...")