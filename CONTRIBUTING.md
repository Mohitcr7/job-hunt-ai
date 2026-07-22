# Contributing to Job Hunt AI

Thanks for helping make job hunting less painful for students everywhere! 🎉

## Development setup

```bash
git clone https://github.com/<your-username>/job-hunt-ai.git
cd job-hunt-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
scrapling install       # headless browser for Scrapling-based scrapers
cp .env.example .env      # add a Gemini key (free at aistudio.google.com)
python main.py check      # verify everything works
```

## Where to contribute

| Area | File(s) | Difficulty |
|---|---|---|
| New job-board scraper | `tools/scraper_tool.py` | Medium |
| Company career-page configs | `COMPANY_CAREER_PAGES` in `tools/scraper_tool.py` | Easy |
| Better fit-scoring prompts | `agents/matcher_agent.py` | Easy |
| Better tailoring prompts | `agents/tailor_agent.py` | Easy |
| Dashboard improvements | `frontend/` (vanilla JS, no build step) | Easy–Medium |
| Tailored resume PDF export | new tool | Medium |
| Tests | `tests/` | Easy |

## Ground rules

1. **No auto-submission features.** Automating the final "Apply" click violates platform ToS and gets users banned. PRs adding this will be closed. Automating research, matching, and writing is the project's scope.
2. **Keep scrapers polite.** Preserve rate-limiting delays. Don't add proxy-rotation / CAPTCHA-bypass code.
3. **The tailor must never fabricate.** Any prompt change to `tailor_agent.py` must keep the "never invent experience" constraint.
4. **Fail soft.** Optional integrations (Slack, email, a new scraper) must degrade gracefully when unconfigured — never break the pipeline.
5. **Keep it free-tier friendly.** New features shouldn't require paid APIs by default.

## Pull request process

1. Fork, create a branch: `git checkout -b feat/my-feature`
2. Make your change. Match the existing code style — this project is intentionally heavily commented so beginners can learn from it.
3. Run the smoke test: `python -m pytest tests/ -q` and `python main.py check`
4. Open a PR with a clear description of what and why.

## Reporting bugs

Open an issue with:
- what you ran and what happened (paste the log)
- your OS and Python version
- which LLM provider you're using

## Code style

- Python: readable > clever. Type hints on public functions. `loguru` for logging.
- Comments should explain *why*, and teach — many users are students reading agentic-AI code for the first time.
