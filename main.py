# main.py
#
# Job Hunt AI — command line entry point.
#
#   python main.py check   → verify config + LLM connection
#   python main.py run     → run the full agent pipeline once (headless)
#   python main.py sheet   → scrape today's jobs into a dated .xlsx
#   python main.py serve   → start the web dashboard at http://localhost:8000
#
# Running with no arguments defaults to `serve`.

import sys

from config import validate_config, get_llm, JOB_PREFERENCES, MATCH_THRESHOLD


def check_setup():
    print("=== Job Hunt AI — Setup Check ===\n")

    ok = validate_config()
    if not ok:
        print("Fix the errors above before continuing.")
        return False

    print(f"Match threshold : {MATCH_THRESHOLD}%")
    print(f"Target roles    : {JOB_PREFERENCES['roles']}")
    print(f"Target locations: {JOB_PREFERENCES['locations']}")

    print("\nTesting LLM connection...")
    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke("Reply with exactly three words: setup is working")
        print(f"LLM replied: {response.content}")
        print("\nAll good — you're ready to run the pipeline!")
        return True
    except Exception as e:
        print(f"LLM test failed: {e}")
        print("Double-check your API key in .env")
        return False


def run_once():
    """Headless pipeline run — ideal for cron jobs."""
    from graph.pipeline import run_pipeline

    final = run_pipeline()
    kits = final.get("kits", [])
    print(f"\nDone. {len(kits)} application kit(s) ready in output/applications/")
    print("Open the dashboard with:  python main.py serve")


def make_sheet():
    """
    Scrape today's postings into a dated spreadsheet — what the daily schedule runs.

      python main.py sheet                        # → SHEET_OUTPUT_DIR, or ./output
      python main.py sheet ~/Desktop              # → explicit destination
      python main.py sheet ~/Desktop --hours 48   # widen the window
      python main.py sheet ~/Desktop --llm-top 15 # real LLM scores for the top 15
    """
    import os
    from exports.daily_sheet import build_daily_sheet, EmptyScrapeError

    args = sys.argv[2:]

    def flag(name, default):
        if name in args:
            position = args.index(name) + 1
            if position < len(args):
                return int(args[position])
        return default

    positional = [a for a in args if not a.startswith("--") and not a.isdigit()]
    output_dir = positional[0] if positional else os.getenv("SHEET_OUTPUT_DIR", "output")
    hours_old = flag("--hours", 24)
    llm_top_n = flag("--llm-top", 0)

    try:
        path = build_daily_sheet(
            output_dir=output_dir,
            hours_old=hours_old,
            llm_top_n=llm_top_n,
        )
    except EmptyScrapeError as e:
        # Exit non-zero so the scheduled wrapper logs a failure and notifies,
        # instead of a silent green run that produced nothing.
        print(f"\nNo sheet written: {e}")
        sys.exit(1)

    print(f"\nSpreadsheet ready: {path}")


def serve():
    """
    Start the API + web dashboard.

    Port resolution order (first one wins):
      1. CLI argument   →  python main.py serve 8001
      2. PORT env var   →  PORT=8001 python main.py serve
      3. default 8000
    """
    import os
    import uvicorn

    port = 8000
    if os.getenv("PORT"):
        port = int(os.environ["PORT"])
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        port = int(sys.argv[2])

    # Bind to localhost only — the dashboard has no auth and holds your
    # resume. (Docker overrides host via the uvicorn CLI in the Dockerfile.)
    print(f"Dashboard: http://localhost:{port}")
    print("(Ctrl+C to stop)")
    uvicorn.run("api.server:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if command == "check":
        check_setup()
    elif command == "run":
        run_once()
    elif command == "sheet":
        make_sheet()
    elif command == "serve":
        serve()
    else:
        print(f"Unknown command: {command}")
        print(__doc__ or "Usage: python main.py [check|run|sheet|serve]")
        sys.exit(1)
