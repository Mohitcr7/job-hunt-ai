# agents/applier_agent.py
#
# WHAT THIS AGENT DOES:
# Takes every application kit from the Tailor Agent and:
#   1. Saves it as a readable Markdown file in output/applications/
#   2. Records it in the SQLite tracker (status: "prepared")
#   3. Sends a notification (console/Slack/email) so you know it's ready
#
# WHY IT DOESN'T AUTO-SUBMIT APPLICATIONS:
# Auto-filling application forms on LinkedIn/Naukri/Indeed violates their
# Terms of Service and gets accounts banned — the worst possible outcome
# mid job-hunt. Recruiters also filter out obviously-botted applications.
# This project's philosophy: automate the RESEARCH and WRITING (the slow
# part), keep the human on the final click (the risky part).

import re
from pathlib import Path
from typing import List

from loguru import logger

from graph.state import ApplicationKit, PipelineState
from review import tracker
from tools.notifier_tool import notify_new_kits

OUTPUT_DIR = Path("output/applications")


def _slugify(text: str) -> str:
    """'ML Engineer @ Acme!' → 'ml-engineer-acme' (safe for filenames)."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")[:60]


def save_kit_markdown(kit: ApplicationKit) -> Path:
    """Writes one application kit as a Markdown file you can read/copy from."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(kit['company'])}--{_slugify(kit['job_title'])}.md"
    path = OUTPUT_DIR / filename

    bullets = "\n".join(f"- {b}" for b in kit["tailored_bullets"])
    keywords = ", ".join(kit["keywords_to_add"]) or "(none — resume already covers the JD)"

    path.write_text(f"""# {kit['job_title']} — {kit['company']}

**Match score:** {kit['match_score']}%
**Apply here:** {kit['job_url']}

## Tailored summary

{kit['tailored_summary']}

## Tailored resume bullets

{bullets}

## Keywords to add for ATS

{keywords}

## Cover letter

{kit['cover_letter']}
""")
    return path


def applier_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node. Reads: kits, jobs (for location/platform metadata).
    Writes: applied, status, errors.
    """
    errors = list(state.get("errors", []))
    kits = state.get("kits", [])

    if not kits:
        status = "Applier: no kits to process"
        logger.info(status)
        return {"applied": [], "status": status, "errors": errors}

    # Look up job metadata (location/platform) by URL for the tracker
    job_meta = {job.url: job for job in state.get("jobs", [])}

    applied: List[str] = []
    new_kits: List[ApplicationKit] = []

    for kit in kits:
        try:
            path = save_kit_markdown(kit)
            job = job_meta.get(kit["job_url"])
            is_new = tracker.record_application(
                dict(kit),
                location=job.location if job else "",
                platform=job.platform if job else "",
            )
            if is_new:
                applied.append(kit["job_url"])
                new_kits.append(kit)
                logger.info(f"Prepared: {kit['job_title']} @ {kit['company']} → {path}")
            else:
                logger.info(f"Already tracked, skipping: {kit['job_url']}")
        except Exception as e:
            logger.error(f"Failed to record kit for {kit['job_url']}: {e}")
            errors.append(f"Applier failed for {kit['job_title']}: {e}")

    # Notify about genuinely new kits only
    if new_kits:
        try:
            notify_new_kits(new_kits)
        except Exception as e:
            logger.warning(f"Notification failed (non-fatal): {e}")
            errors.append(f"Notification failed: {e}")

    status = f"Applier: {len(new_kits)} new application kits ready in {OUTPUT_DIR}/"
    logger.info(status)
    return {"applied": applied, "status": status, "errors": errors}
