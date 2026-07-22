# review/tracker.py
#
# WHAT THIS FILE DOES:
# A small SQLite-backed tracker for every application kit the pipeline
# produces. This is the "memory" of your job hunt — which jobs you found,
# which you prepared materials for, which you actually submitted, and
# what happened next (interview, rejection, offer).
#
# WHY SQLITE (stdlib) INSTEAD OF A BIG ORM:
# - zero extra dependencies, works everywhere Python works
# - one file on disk (data/applications.db) — trivial to back up
# - the dashboard reads it over the API

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import DATABASE_URL

# Application lifecycle statuses shown in the dashboard
STATUSES = [
    "prepared",      # kit generated, not yet submitted
    "applied",       # you submitted the application
    "interviewing",  # heard back, in process
    "offer",         # got an offer 🎉
    "rejected",      # didn't work out
    "archived",      # decided not to apply
]


def _db_path() -> str:
    """DATABASE_URL looks like sqlite:///./data/applications.db — strip the scheme."""
    path = DATABASE_URL.replace("sqlite:///", "", 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_conn():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Creates the applications table if it doesn't exist. Safe to call always."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_url TEXT UNIQUE NOT NULL,
                job_title TEXT NOT NULL,
                company TEXT NOT NULL,
                location TEXT DEFAULT '',
                platform TEXT DEFAULT '',
                match_score INTEGER DEFAULT 0,
                status TEXT DEFAULT 'prepared',
                tailored_summary TEXT DEFAULT '',
                tailored_bullets TEXT DEFAULT '[]',   -- JSON list
                cover_letter TEXT DEFAULT '',
                keywords_to_add TEXT DEFAULT '[]',    -- JSON list
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_application(
    kit: dict,
    location: str = "",
    platform: str = "",
) -> bool:
    """
    Inserts a new application kit. Returns False if this job_url is
    already tracked (so re-running the pipeline never creates duplicates).
    """
    init_db()
    with get_conn() as conn:
        try:
            conn.execute(
                """INSERT INTO applications
                   (job_url, job_title, company, location, platform, match_score,
                    status, tailored_summary, tailored_bullets, cover_letter,
                    keywords_to_add, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?)""",
                (
                    kit["job_url"], kit["job_title"], kit["company"],
                    location, platform, kit["match_score"],
                    kit["tailored_summary"],
                    json.dumps(kit["tailored_bullets"]),
                    kit["cover_letter"],
                    json.dumps(kit["keywords_to_add"]),
                    _now(), _now(),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False  # already tracked


def list_applications(status: Optional[str] = None) -> List[dict]:
    """Returns all tracked applications, newest first."""
    init_db()
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM applications WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM applications ORDER BY created_at DESC"
            ).fetchall()

    apps = []
    for row in rows:
        app = dict(row)
        app["tailored_bullets"] = json.loads(app["tailored_bullets"])
        app["keywords_to_add"] = json.loads(app["keywords_to_add"])
        apps.append(app)
    return apps


def update_status(app_id: int, status: str, notes: Optional[str] = None) -> bool:
    """Moves an application through the lifecycle (prepared → applied → ...)."""
    if status not in STATUSES:
        raise ValueError(f"Unknown status '{status}'. Valid: {STATUSES}")
    init_db()
    with get_conn() as conn:
        if notes is not None:
            cur = conn.execute(
                "UPDATE applications SET status = ?, notes = ?, updated_at = ? WHERE id = ?",
                (status, notes, _now(), app_id),
            )
        else:
            cur = conn.execute(
                "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), app_id),
            )
        return cur.rowcount > 0


def is_tracked(job_url: str) -> bool:
    """True if we already prepared materials for this job URL."""
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM applications WHERE job_url = ?", (job_url,)
        ).fetchone()
        return row is not None


def get_stats() -> dict:
    """Aggregate counts for the dashboard."""
    init_db()
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM applications").fetchone()["c"]
        by_status = {
            row["status"]: row["c"]
            for row in conn.execute(
                "SELECT status, COUNT(*) c FROM applications GROUP BY status"
            ).fetchall()
        }
        avg_score = conn.execute(
            "SELECT AVG(match_score) avg FROM applications"
        ).fetchone()["avg"]

    return {
        "total": total,
        "by_status": by_status,
        "avg_match_score": round(avg_score or 0, 1),
    }
