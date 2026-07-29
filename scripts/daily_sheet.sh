#!/bin/bash
#
# Daily job-sheet runner — what the launchd agent (macOS) or cron (Linux) calls.
#
# Kept as a wrapper rather than calling python directly from the scheduler
# because a scheduled job starts with almost no environment: no PATH to your
# virtualenv, no working directory, and therefore no .env. This fixes all three
# and leaves a log behind so a silent failure at noon is still diagnosable.
#
# Machine-specific values come from the environment so this file stays generic:
#   JOB_HUNT_PYTHON   interpreter to use          (default: python3 on PATH)
#   SHEET_OUTPUT_DIR  where the .xlsx is written  (default: ~/Desktop)
#   SHEET_HOURS       how far back to scrape      (default: 24)
#   SHEET_LLM_TOP     LLM-score this many rows    (default: 0 = free run)

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || { echo "Cannot enter $PROJECT_DIR"; exit 1; }

PYTHON_BIN="${JOB_HUNT_PYTHON:-python3}"
OUTPUT_DIR="${SHEET_OUTPUT_DIR:-$HOME/Desktop}"
HOURS="${SHEET_HOURS:-24}"
LLM_TOP="${SHEET_LLM_TOP:-0}"

LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/daily-sheet.log"
mkdir -p "$LOG_DIR"

# Keep the log from growing without bound — this runs every day forever.
if [[ -f "$LOG_FILE" ]] && [[ $(wc -c < "$LOG_FILE") -gt 5242880 ]]; then
    mv "$LOG_FILE" "$LOG_FILE.1"
fi

{
    echo ""
    echo "=============================================================="
    echo "Run started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Python: $PYTHON_BIN | Output: $OUTPUT_DIR | Window: ${HOURS}h"
    echo "=============================================================="
} >> "$LOG_FILE"

"$PYTHON_BIN" main.py sheet "$OUTPUT_DIR" --hours "$HOURS" --llm-top "$LLM_TOP" >> "$LOG_FILE" 2>&1
STATUS=$?

if [[ $STATUS -eq 0 ]]; then
    echo "Run finished OK: $(date '+%H:%M:%S')" >> "$LOG_FILE"
else
    echo "Run FAILED (exit $STATUS): $(date '+%H:%M:%S')" >> "$LOG_FILE"
    # Surface the failure where it will actually be noticed, not just in a log.
    /usr/bin/osascript -e 'display notification "Job scrape failed — see logs/daily-sheet.log" with title "Job Hunt AI"' 2>/dev/null
fi

exit $STATUS
