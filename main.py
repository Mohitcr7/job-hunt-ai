# main.py
#
# Job Hunt AI — command line entry point.
#
#   python main.py check   → verify config + LLM connection
#   python main.py run     → run the full agent pipeline once (headless)
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
    elif command == "serve":
        serve()
    else:
        print(f"Unknown command: {command}")
        print(__doc__ or "Usage: python main.py [check|run|serve]")
        sys.exit(1)
