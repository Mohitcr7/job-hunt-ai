# tools/notifier_tool.py
#
# WHAT THIS FILE DOES:
# Sends "your application kits are ready" notifications through whichever
# channels you've configured in .env:
#   - Console  : always on (pretty terminal output)
#   - Slack    : if SLACK_BOT_TOKEN + SLACK_CHANNEL are set
#   - Email    : if SENDGRID_API_KEY + NOTIFY_EMAIL are set
#
# Every channel fails soft — a missing token or network error logs a
# warning and moves on. Notifications must never break the pipeline.

import os
from typing import List

from loguru import logger

from config import SLACK_BOT_TOKEN, SENDGRID_API_KEY

SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#job-hunt")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "jobhunt-ai@noreply.local")


def _format_summary(kits: List[dict]) -> str:
    lines = [f"🎯 Job Hunt AI — {len(kits)} new application kit(s) ready!\n"]
    for kit in kits:
        lines.append(
            f"  • [{kit['match_score']}%] {kit['job_title']} @ {kit['company']}\n"
            f"    {kit['job_url']}"
        )
    lines.append("\nMaterials saved in output/applications/ — review and apply!")
    return "\n".join(lines)


def notify_console(kits: List[dict]):
    print("\n" + "=" * 60)
    print(_format_summary(kits))
    print("=" * 60 + "\n")


def notify_slack(kits: List[dict]) -> bool:
    if not SLACK_BOT_TOKEN:
        return False
    try:
        from slack_sdk import WebClient

        client = WebClient(token=SLACK_BOT_TOKEN)
        client.chat_postMessage(channel=SLACK_CHANNEL, text=_format_summary(kits))
        logger.info(f"Slack notification sent to {SLACK_CHANNEL}")
        return True
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")
        return False


def notify_email(kits: List[dict]) -> bool:
    if not (SENDGRID_API_KEY and NOTIFY_EMAIL):
        return False
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        body = _format_summary(kits).replace("\n", "<br>")
        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=NOTIFY_EMAIL,
            subject=f"Job Hunt AI: {len(kits)} new application kit(s) ready",
            html_content=f"<pre style='font-family:sans-serif'>{body}</pre>",
        )
        SendGridAPIClient(SENDGRID_API_KEY).send(message)
        logger.info(f"Email notification sent to {NOTIFY_EMAIL}")
        return True
    except Exception as e:
        logger.warning(f"Email notification failed: {e}")
        return False


def notify_new_kits(kits: List[dict]):
    """The one function the Applier Agent calls. Tries every channel."""
    if not kits:
        return
    notify_console(kits)
    notify_slack(kits)
    notify_email(kits)
