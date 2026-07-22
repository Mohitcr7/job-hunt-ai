# 🎯 Job Hunt AI

**An open-source, agentic AI job-search copilot for students and early-career engineers.**

Job hunting is a numbers game where the slow part isn't clicking "Apply" — it's *finding* fresh relevant postings every day, figuring out which ones actually fit you, and tailoring your resume + cover letter for each one. Job Hunt AI automates exactly that slow part with a team of AI agents, and leaves the final click to you.

```
 ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
 │  SCOUT   │───►│ MATCHER  │───►│ TAILOR  │───►│ REVIEW  │───►│ APPLIER  │
 │ scrapes  │    │ FAISS +  │    │ resume  │    │ critic  │    │ saves kit │
 │ jobs     │    │ LLM fit  │    │ + cover │    │ + editor│    │ + tracks  │
 │          │    │ scoring  │    │ letter  │    │ polish  │    │ + notifies│
 └─────────┘    └──────────┘    └─────────┘    └─────────┘    └──────────┘
```

Built with **LangGraph**, **LangChain**, **FAISS**, **Scrapling**, and **FastAPI** — with a clean web dashboard to run the pipeline and track every application from *prepared* → *applied* → *interviewing* → *offer*.

## ✨ What it does

- **Scout Agent** — scrapes fresh postings from LinkedIn + Indeed (via [jobspy](https://github.com/Bunsly/JobSpy)) and Naukri (via [Scrapling](https://github.com/D4Vinci/Scrapling), whose self-healing selectors survive site markup changes), deduplicated across sources. Parses your resume PDF into structured data with an LLM (cached, so it only costs one call).
- **Matcher Agent** — two-stage matching: free local FAISS embeddings filter out the noise, then an LLM reads the top candidates against your resume and scores fit 0–100 (catching hard requirements embeddings miss, like "8+ years required").
- **Tailor Agent** — for each match above your threshold, writes a tailored resume summary, re-worded bullet points, missing ATS keywords, and a full cover letter. **It never invents experience** — it only re-words and emphasises what's genuinely on your resume.
- **Review Crew** — a skeptical "hiring manager" persona critiques each cover letter, then an "editor" persona rewrites it. Kills the generic AI-slop tone.
- **Applier Agent** — saves each application kit as Markdown in `output/applications/`, records it in a SQLite tracker, and notifies you (console, Slack, or email).
- **Web dashboard** — run the pipeline, browse matches with fit scores, read/copy your tailored materials, and move applications through the funnel.

## 🖱️ Why it doesn't auto-submit applications

Auto-filling forms on LinkedIn/Naukri/Indeed violates their Terms of Service and gets accounts banned — the worst possible outcome mid job-hunt. Recruiters also increasingly filter obviously-botted applications. This project's philosophy: **automate the research and writing (the slow part), keep the human on the final click (the risky part).**

## 🚀 Quickstart

**Prerequisites:** Python 3.10+, a free [Gemini API key](https://aistudio.google.com)

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/job-hunt-ai.git
cd job-hunt-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
scrapling install       # downloads the headless browser Scrapling uses for Naukri

# 2. Configure
cp .env.example .env
#    → open .env and paste your GOOGLE_API_KEY

# 3. Add your resume
cp /path/to/your/resume.pdf data/resume.pdf

# 4. Verify setup
python main.py check

# 5. Launch the dashboard
python main.py serve          # → http://localhost:8000
```

Set your target roles and locations in `config.py` (`JOB_PREFERENCES`), or override them per-run from the dashboard.

You can also run headless (great for a daily cron job):

```bash
python main.py run
```

## 🗂️ Project structure

```
job_hunt_ai/
├── agents/                  # the four pipeline agents
│   ├── scout_agent.py       #   parse resume + scrape jobs
│   ├── matcher_agent.py     #   FAISS pre-filter + LLM fit scoring
│   ├── tailor_agent.py      #   tailored summary/bullets/cover letter
│   └── applier_agent.py     #   save kits, track, notify
├── graph/
│   ├── state.py             # shared pipeline state (TypedDict)
│   └── pipeline.py          # LangGraph wiring: scout→match→tailor→review→apply
├── tools/
│   ├── resume_parser_tool.py  # PDF → structured ParsedResume (LLM)
│   ├── scraper_tool.py        # jobspy + Scrapling scrapers (self-healing)
│   ├── vector_store_tool.py   # FAISS index over job embeddings
│   └── notifier_tool.py       # console / Slack / email
├── review/
│   ├── autogen_crew.py      # reviewer+editor cover-letter polish
│   └── tracker.py           # SQLite application tracker
├── api/server.py            # FastAPI backend
├── frontend/                # dashboard (vanilla JS — no build step)
├── config.py                # settings + LLM factory
└── main.py                  # CLI: check | run | serve
```

## 💸 Cost

Designed to run on **free tiers**:

- Embeddings are local ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) — $0, private, offline.
- LLM calls per run are capped: 1 resume parse (cached) + ~15 fit scores + ~8 tailoring calls + ~16 review calls. On Gemini Flash's free tier this costs **nothing**; on paid tiers it's pennies per run.

## 🐳 Docker

```bash
cp .env.example .env   # fill in your key
docker compose up --build
# → http://localhost:8000
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for cloud options (Railway, Render, Hugging Face Spaces) and daily scheduling.

## ☁️ Enterprise deploy — Amazon Bedrock AgentCore

The same crew runs serverless on AWS with one flag. Because every agent calls a single `get_llm()` factory, setting `LLM_PROVIDER=bedrock` moves the whole pipeline to **Claude on Amazon Bedrock** — then AgentCore Runtime adds serverless scaling, session isolation, memory, and observability, with a **Managed Knowledge Base** replacing the FAISS layer and platform **Guardrails** enforcing the no-fabrication rule.

```bash
pip install -r requirements.txt -r requirements-bedrock.txt
python provision_bedrock.py                       # create the Guardrail
agentcore configure --entrypoint job_hunt_agent.py
agentcore launch --env LLM_PROVIDER=bedrock --env GUARDRAIL_ID=... --env GUARDRAIL_VERSION=...
```

Full walkthrough: [BEDROCK.md](BEDROCK.md).

## 🤝 Contributing

This project exists to help students land jobs — contributions are very welcome! Good first issues:

- Add a scraper for your region's job board (Internshala, Wellfound, Instahyre…)
- Add company career-page configs in `tools/scraper_tool.py`
- Improve the fit-scoring prompt
- Add resume export (tailored PDF generation)

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

## ⚖️ Responsible use

- Scrapers include polite rate-limiting — keep it that way.
- Respect each platform's Terms of Service; don't add auto-submission features.
- The tailor agent is prompt-constrained to never fabricate experience. A resume that lies gets you blacklisted; keep it honest.

## 📄 License

[MIT](LICENSE) — free to use, modify, and share.

---

*Built by a student, for students. If it helped you land an interview, star the repo so others find it.* ⭐
