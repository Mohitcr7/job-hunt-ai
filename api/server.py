# api/server.py
#
# WHAT THIS FILE DOES:
# The FastAPI backend behind the web dashboard. It exposes:
#   - pipeline control  : start a run, poll its progress
#   - application data  : list/update tracked applications, stats
#   - resume management : view the parsed resume, upload a new PDF
#   - settings          : view current search preferences
#
# The pipeline runs in a background thread (it takes minutes — scraping,
# embeddings, LLM calls) while the UI polls /api/pipeline/status.
#
# Run locally:  uvicorn api.server:app --reload
# Or simply :   python main.py serve

import dataclasses
import shutil
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from config import JOB_PREFERENCES, MATCH_THRESHOLD, LLM_PROVIDER
from review import tracker

app = FastAPI(
    title="Job Hunt AI",
    description="Agentic AI job search assistant — scout, match, tailor, track.",
    version="1.0.0",
)

# CORS: allow the dashboard to be served from a different origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pipeline run management (one run at a time, tracked in memory)
# ---------------------------------------------------------------------------

_run_lock = threading.Lock()
_run_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "status": "idle",
    "phase": "idle",        # scraping | matching | reviewing | done | error
    "preliminary": [],      # jobs above 50% from the fast FAISS pass (Phase 1)
    "kits_total": 0,        # how many applications Phase 2 will prepare
    "kits_done": [],        # applications finished so far — streamed in live
    "result": None,         # final summary once Phase 2 finishes
    "error": None,
    "log": deque(maxlen=80),  # rolling progress lines streamed to the dashboard
}


class RunRequest(BaseModel):
    search_terms: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    hours_old: int = 24
    min_score: Optional[int] = None


def _job_summary(match) -> dict:
    """Flatten a JobMatch into the small dict the dashboard renders."""
    return {
        "title": match.job.title,
        "company": match.job.company,
        "score": match.score_percent,
        "url": match.job.url,
        "location": match.job.location,
        "platform": match.job.platform,
    }


def _prepare_kit(resume, match, job_meta: dict):
    """
    Tailor → review → save → track ONE match, end to end. This is the unit the
    thread pool parallelises and streams. Returns (kit_dict, is_new).
    """
    from agents.tailor_agent import build_kit
    from review.autogen_crew import review_cover_letter, ENABLE_REVIEW_CREW
    from agents.applier_agent import save_kit_markdown

    kit = build_kit(resume, match)
    if ENABLE_REVIEW_CREW:
        try:
            kit["cover_letter"] = review_cover_letter(kit)
        except Exception as e:
            logger.warning(f"Review failed for {kit['job_title']}: {e}")  # keep original letter

    save_kit_markdown(kit)
    job = job_meta.get(kit["job_url"])
    is_new = tracker.record_application(
        dict(kit),
        location=job.location if job else "",
        platform=job.platform if job else "",
    )
    return kit, is_new


def _pipeline_worker(req: RunRequest):
    """
    Runs the pipeline in a background thread as TWO phases, so the UI feels fast:

      Phase 1 (seconds): scout (scrape + parse) + FAISS semantic match, then
                         publish every job above 50% for the user to browse NOW.
      Phase 2 (minutes): LLM-rerank the top candidates → tailor → review → track,
                         all while the user reads the Phase-1 list.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from agents.scout_agent import scout_node
    from agents.matcher_agent import faiss_prefilter, rerank_candidates, llm_rerank
    from agents.tailor_agent import MAX_KITS_PER_RUN
    from config import MATCH_THRESHOLD, JOB_PREFERENCES, LLM_CONCURRENCY

    _run_state["log"].clear()

    # Stream ONLY genuine progress (INFO level) to the dashboard. Warnings and
    # errors — rate-limit dumps, skipped-source notices — stay in the server
    # log, out of the user's view. Skip pure separator lines too.
    def _capture(message):
        text = message.record["message"].strip()
        if text and any(c.isalnum() for c in text):
            _run_state["log"].append(text)

    sink_id = logger.add(_capture, filter=lambda r: r["level"].name == "INFO")

    try:
        min_score = req.min_score if req.min_score is not None else MATCH_THRESHOLD
        state = {
            "resume_path": "data/resume.pdf",
            "search_terms": req.search_terms,
            "locations": req.locations,
            "hours_old": req.hours_old,
            "min_score": min_score,
            "errors": [],
        }

        # ---- Phase 1: scrape + parse + fast FAISS prefilter ----
        _run_state["phase"] = "scraping"
        state.update(scout_node(state))
        resume, jobs = state.get("resume"), state.get("jobs", [])

        if not resume or not jobs:
            _run_state["result"] = {
                "jobs_scraped": len(jobs), "matches": 0, "kits_created": 0,
                "new_applications": 0, "errors": state.get("errors", []), "top_matches": [],
            }
            _run_state["status"] = state.get("status", "No jobs found for this search")
            _run_state["phase"] = "done"
            return

        _run_state["phase"] = "matching"
        prelim = faiss_prefilter(resume, jobs, min_score=50)
        _run_state["preliminary"] = [_job_summary(m) for m in prelim]
        _run_state["status"] = (
            f"Found {len(prelim)} matches above 50% — reviewing the top "
            f"{min(len(prelim), 15)} with the LLM in the background…"
        )

        # ---- Phase 2a: rerank → parallel LLM fit-scoring ----
        _run_state["phase"] = "reviewing"
        exp_years = JOB_PREFERENCES.get("experience_years", 0)
        # Free cross-encoder narrows the ≥50% list to the most relevant few
        # BEFORE the (now parallel) LLM fit-scoring runs on them.
        shortlist = rerank_candidates(resume, prelim)
        matches = llm_rerank(resume, shortlist, min_score, exp_years, state["errors"])

        # ---- Phase 2b: tailor + review + save PER KIT, in parallel, STREAMED ----
        to_prepare = matches[:MAX_KITS_PER_RUN]
        job_meta = {j.url: j for j in jobs}
        _run_state["kits_total"] = len(to_prepare)
        _run_state["kits_done"] = []

        kits, new_kits = [], []
        with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
            futures = {pool.submit(_prepare_kit, resume, m, job_meta): m for m in to_prepare}
            for future in as_completed(futures):
                m = futures[future]
                try:
                    kit, is_new = future.result()
                    kits.append(kit)
                    if is_new:
                        new_kits.append(kit)
                    # Stream this finished application to the UI immediately.
                    _run_state["kits_done"].append({
                        "title": kit["job_title"], "company": kit["company"],
                        "score": kit["match_score"], "url": kit["job_url"],
                    })
                except Exception as e:
                    logger.error(f"Kit prep failed for {m.job.url}: {e}")
                    state["errors"].append(f"Kit prep failed for {m.job.title}: {e}")

        # One notification for the genuinely new kits.
        if new_kits:
            try:
                from tools.notifier_tool import notify_new_kits
                notify_new_kits(new_kits)
            except Exception as e:
                logger.warning(f"Notification failed: {e}")

        _run_state["result"] = {
            "jobs_scraped": len(jobs),
            "matches": len(matches),
            "kits_created": len(kits),
            "new_applications": len(new_kits),
            "errors": state.get("errors", []),
            "top_matches": [_job_summary(m) for m in matches[:20]],
        }
        _run_state["status"] = f"Done — {len(kits)} application kits ready"
        _run_state["phase"] = "done"
    except Exception as e:
        _run_state["error"] = str(e)
        _run_state["status"] = f"failed: {e}"
        _run_state["phase"] = "error"
    finally:
        logger.remove(sink_id)  # detach the buffer sink so it can't leak
        _run_state["running"] = False
        _run_state["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/pipeline/run")
def start_pipeline(req: RunRequest):
    with _run_lock:
        if _run_state["running"]:
            raise HTTPException(409, "A pipeline run is already in progress")
        _run_state.update({
            "running": True,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "status": "running — scraping jobs…",
            "phase": "scraping",
            "preliminary": [],
            "kits_total": 0,
            "kits_done": [],
            "result": None,
            "error": None,
        })
    threading.Thread(target=_pipeline_worker, args=(req,), daemon=True).start()
    return {"message": "Pipeline started", "started_at": _run_state["started_at"]}


@app.get("/api/pipeline/status")
def pipeline_status():
    # Snapshot the mutable collections (the worker thread mutates them live) so
    # JSON serialisation can't hit a "changed size during iteration" race.
    state = dict(_run_state)
    state["log"] = list(_run_state["log"])
    state["preliminary"] = list(_run_state["preliminary"])
    state["kits_done"] = list(_run_state["kits_done"])
    return state


# ---------------------------------------------------------------------------
# Applications tracker
# ---------------------------------------------------------------------------

class StatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None


@app.get("/api/applications")
def get_applications(status: Optional[str] = None):
    return tracker.list_applications(status=status)


@app.patch("/api/applications/{app_id}")
def patch_application(app_id: int, update: StatusUpdate):
    try:
        ok = tracker.update_status(app_id, update.status, update.notes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, f"Application {app_id} not found")
    return {"message": "updated"}


@app.get("/api/stats")
def get_stats():
    return tracker.get_stats()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

RESUME_PATH = Path("data/resume.pdf")


@app.get("/api/resume")
def get_parsed_resume():
    cache = Path("data/parsed_resume.json")
    if not cache.exists():
        raise HTTPException(404, "No parsed resume yet — upload a PDF and run the pipeline")
    import json
    return json.loads(cache.read_text())


@app.post("/api/resume/upload")
def upload_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file")
    RESUME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESUME_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse immediately so the new profile shows up right away. Parsing is one
    # LLM call (a few seconds) — fine for an explicit upload, and it refreshes
    # the cached parsed_resume.json the Scout agent reuses on the next run.
    # If parsing fails, we keep the saved PDF and fall back to lazy parsing.
    try:
        from agents.scout_agent import get_resume

        resume = get_resume(str(RESUME_PATH), force_reparse=True)
        return {
            "message": f"Resume saved and parsed — found {len(resume.skills)} skills.",
            "parsed": True,
            "name": resume.name,
            "skills_count": len(resume.skills),
        }
    except Exception as e:
        # Drop any stale cache from the previous resume so the next run reparses.
        cache = Path("data/parsed_resume.json")
        if cache.exists():
            cache.unlink()
        return {
            "message": f"Resume saved, but parsing failed ({e}). It will be parsed on the next run.",
            "parsed": False,
        }


# ---------------------------------------------------------------------------
# Settings / health
# ---------------------------------------------------------------------------

@app.get("/api/settings")
def get_settings():
    return {
        "llm_provider": LLM_PROVIDER,
        "match_threshold": MATCH_THRESHOLD,
        "preferences": JOB_PREFERENCES,
        "resume_uploaded": RESUME_PATH.exists(),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Serve the dashboard (frontend/) at the root URL
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
