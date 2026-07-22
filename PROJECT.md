# Job Hunt AI — Project Summary

A single source of truth for what this project is, how it's built, and every
decision made along the way. Read this before touching the code or resuming
work in a new session.

---

## 1. What this is

**Job Hunt AI** is an agentic AI system that automates the slow, repetitive
parts of a job search — finding fresh postings, judging fit, and writing
tailored application materials — while deliberately leaving the final
"Submit" click to the human. It is built to be:

- **Open source** — MIT-licensed, documented for contributors, no personal
  data ever committed.
- **Resume-worthy** — architecture and language chosen to read well for
  Data Scientist / AI Engineer job applications (RAG, multi-agent
  orchestration, two-stage retrieval, MLOps).
- **Free-tier friendly** — designed to run on $0 (Gemini free tier +
  free OpenRouter models + local embeddings), with a paid path available.

---

## 2. Architecture overview

### 2.1 The agent pipeline (LangGraph)

One shared state dict (`PipelineState`, `graph/state.py`) flows through five
nodes wired as a LangGraph state machine (`graph/pipeline.py`):

```
                    PipelineState (one dict, enriched at each step)
                          │
  ┌────────┐   ┌─────────┐   ┌────────┐   ┌────────┐   ┌─────────┐
  │ SCOUT  │──►│ MATCHER │──►│ TAILOR │──►│ REVIEW │──►│ APPLIER │──► END
  └────────┘   └─────────┘   └────────┘   └────────┘   └─────────┘
   resume        matches       kits         kits         applied
   + jobs       (3-stage)    (parallel)   (parallel)   (saved+tracked)
       │
       └─ if no resume OR no jobs ──────────────────────────────► END
```

**Scout** ([agents/scout_agent.py](agents/scout_agent.py)) — parses the resume
PDF into structured JSON via one cached LLM call, and scrapes fresh postings.

**Matcher** ([agents/matcher_agent.py](agents/matcher_agent.py)) — a
**three-stage funnel**, not a single pass:

1. `faiss_prefilter()` — free, local, ~1–3s. Embeds every job and the resume
   with `all-MiniLM-L6-v2` and keeps everything above a low floor (removes
   obvious noise).
2. `rerank_candidates()` — one **batched** call to a free NVIDIA cross-encoder
   reranker (via OpenRouter's `/rerank` endpoint) that sharpens relevance and
   narrows the shortlist to ~8. This runs on a *separate* free budget, so it
   doesn't compete with the generation LLM's rate limit.
3. `llm_rerank()` — the generation LLM reads each of the ~8 shortlisted JDs
   against the resume and returns a real 0–100 fit score + reasoning. This is
   the stage that catches hard requirements embeddings/rerankers can't
   ("8+ years required") — it can *penalize* a topically-perfect match the
   candidate isn't eligible for. Runs the 8 calls **concurrently**
   (`ThreadPoolExecutor`, bounded by `LLM_CONCURRENCY`).

**Tailor** ([agents/tailor_agent.py](agents/tailor_agent.py)) — for each
match above threshold (capped at `MAX_KITS_PER_RUN=8`), one LLM call produces
a tailored summary, reworded bullets, ATS keywords, and a cover letter. Built
**concurrently** across kits. The prompt hard-forbids inventing experience.

**Review** ([review/autogen_crew.py](review/autogen_crew.py)) — a two-persona
"crew" (critic → editor) polishes each cover letter. Runs **concurrently**
across kits (the two calls *within* one kit stay ordered, since the editor
needs the critique). Toggle with `ENABLE_REVIEW_CREW`.

**Applier** ([agents/applier_agent.py](agents/applier_agent.py)) — saves each
kit as Markdown in `output/applications/`, records it in the SQLite tracker
(deduplicated by URL), and notifies (console/Slack/email). Deliberately does
**not** auto-submit — see §5.

### 2.2 Two-phase web experience

The FastAPI backend ([api/server.py](api/server.py)) runs the pipeline in a
background thread as **two phases** so the UI never feels like a black box:

- **Phase 1 (seconds):** scrape → FAISS prefilter → publish every job ≥50%
  match immediately. The user has something to read while Phase 2 runs.
- **Phase 2 (minutes, background):** reranker → LLM fit-scoring → tailor →
  review → track. Kits **stream into the UI one at a time** as each finishes
  (`kits_done` in `/api/pipeline/status`), not as one big wait-then-dump.

The pipeline's `loguru` log is tee'd into a rolling in-memory buffer
(`_run_state["log"]`, capped at 80 lines, INFO-level only) that the frontend
polls every second and renders as a live stream — separate from the raw
log detail view.

### 2.3 LLM provider abstraction — the key design decision

Every agent calls a single factory, `get_llm()` in [config.py](config.py).
Nothing downstream knows or cares which provider answers. This one
abstraction is what made every later integration a small, additive change:

```
LLM_PROVIDER=gemini|openai|bedrock|openrouter   → picks the primary
                         │
                         ▼
        get_llm() returns primary.with_fallbacks([nemotron])
                         │            (only if OPENROUTER_API_KEY is set)
                         ▼
        Any call that raises (rate limit, quota, transient error)
        transparently retries on the fallback — invisible to the agent.
```

- **Primary providers:** Gemini (default, free tier), OpenAI, AWS Bedrock
  (Claude via `ChatBedrockConverse`), or OpenRouter itself.
- **Fallback:** NVIDIA **Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b:free`)
  via OpenRouter. Chosen over **Nemotron 3 Ultra** deliberately — both are
  free with equivalent reasoning, but Super is ~46% lower latency and ~2.5×
  higher throughput (1.37s/59 tok/s vs 2.53s/24 tok/s p50). A fallback exists
  to keep the pipeline moving fast when the primary fails; Ultra's larger
  context (1M vs 262K) buys nothing since a resume + one JD is a few
  thousand tokens.
- **Verified live:** with `GOOGLE_API_KEY` deliberately broken, `python main.py
  check` still returned a correct reply — proving the fallback actually fires
  and Nemotron answers, not just that the code compiles.

### 2.4 Reranker — a distinct stage from LLM fit-scoring

Added because Gemini's free tier caps at ~20 requests/day, and burning that
budget on matcher scoring starved the tailoring stage. Key insight preserved
throughout: **a reranker and an LLM judge are not interchangeable.**

- A reranker (cross-encoder) scores *topical relevance* — same class of
  judgment as FAISS, just sharper (joint attention over both texts).
- It **cannot** reason about eligibility. It would happily rank "Senior DS,
  10 yrs required" as a top match for a 2-year-experience resume, because
  it's relevant, not because the candidate qualifies.
- So the LLM fit-scoring stage stays — just on a narrower, better-ranked
  shortlist (`RERANK_TOP_N=8` instead of the old `LLM_RERANK_TOP_K=15`),
  which is both faster and doesn't lose the hard-requirement filtering.

Implementation: [tools/reranker_tool.py](tools/reranker_tool.py) calls
OpenRouter's `POST /api/v1/rerank` with `{model, query, documents}` and
parses the Cohere-style `results: [{index, relevance_score}]` response.
**Fails soft** — any error (missing key, network, schema mismatch) falls back
to the FAISS order untouched, so the pipeline never breaks on this stage.
Covered by mocked unit tests ([tests/test_reranker.py](tests/test_reranker.py))
for reorder, no-key, and API-error paths.

### 2.5 Scraping — jobspy + Scrapling (self-healing)

- **LinkedIn + Indeed** via `jobspy` — the reliable source. Runs every
  (role × city) search **concurrently** (`ThreadPoolExecutor`, bounded by
  `SCRAPE_CONCURRENCY=4`) instead of one at a time, cutting scrape wall-clock
  roughly by the concurrency factor.
- **Naukri + company pages** via **Scrapling** (`DynamicFetcher`/
  `DynamicSession`), replacing a hand-rolled async Playwright scraper.
  Chosen specifically for its **self-healing selectors** (`adaptive=True`,
  `auto_save=True`) — when a site's markup changes, Scrapling relocates
  elements by similarity instead of silently returning zero jobs, which was
  the single biggest reliability complaint before the swap. Kept in "polite
  mode" deliberately: rate-limit delays preserved, and the aggressive
  anti-bot `StealthyFetcher` was **not** adopted, to stay inside the
  project's responsible-scraping stance (see §5).
- All fields pass through a `_clean()` helper that turns pandas'
  stringified `NaN` into an empty string — fixes the "nan" company-name
  bug seen in early UI testing.

---

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** + **LangChain** | stateful multi-agent graph, provider-agnostic LLM calls |
| LLM (primary) | **Gemini 2.5 Flash** (default), OpenAI, AWS Bedrock (Claude), OpenRouter | free-tier friendly, swappable via one env var |
| LLM (fallback) | **NVIDIA Nemotron 3 Super** via OpenRouter | free, fast, automatic on primary failure |
| Reranker | **NVIDIA Llama-Nemotron Rerank VL 1B v2** via OpenRouter `/rerank` | free, batched, off the generation budget |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | local, free, private |
| Vector search | **FAISS** (`IndexFlatIP`) | fast local cosine similarity |
| Resume parsing | **pdfplumber** + LLM + **Pydantic** schema | structured, validated extraction |
| Scraping | **jobspy** (LinkedIn/Indeed), **Scrapling** (Naukri, self-healing) | reliability where it matters, resilience where it's flaky |
| Backend | **FastAPI** + background threads | two-phase run, live status polling |
| Persistence | **SQLite** (`sqlite3`, no ORM) | zero-dependency application tracker |
| Frontend | Vanilla **HTML/CSS/JS**, no build step | dark, glassy, premium SaaS aesthetic; inline SVG icon set |
| Deployment | **Docker** / docker-compose, **AWS Bedrock AgentCore** guide | local-first, with an enterprise-serverless path |
| CI | **GitHub Actions** | compile check + pytest smoke tests |
| Testing | **pytest** | tracker roundtrip, notifier formatting, reranker mocked tests |

---

## 4. AWS Bedrock AgentCore integration

A parallel deployment path that moves the *entire* crew onto AWS with no
per-agent rewrite, because everything already goes through `get_llm()`:

| File | Role |
|---|---|
| [job_hunt_agent.py](job_hunt_agent.py) | `BedrockAgentCoreApp` + `@app.entrypoint` wrapper — the serverless entrypoint. Sources jobs from a Bedrock Managed Knowledge Base (or payload, or live scrape fallback), then runs matcher → tailor → review. |
| [tools/bedrock_kb_tool.py](tools/bedrock_kb_tool.py) | Managed KB retrieval — the deployed-on-AWS replacement for the local FAISS layer. |
| [provision_bedrock.py](provision_bedrock.py) | One-time script creating a Bedrock **Guardrail** (contextual grounding + a `FabricatedExperience` denied-topic) that enforces "never invent experience" at the platform level, not just via prompt wording. |
| [BEDROCK.md](BEDROCK.md) | Full deploy guide: install → provision → `agentcore configure/launch` → invoke → observe. |

All AWS imports are **lazy** so the default local/Gemini path has zero new
dependencies; `requirements-bedrock.txt` holds the AWS-only extras
(`boto3`, `langchain-aws`, `bedrock-agentcore`).

---

## 5. Deliberate design decisions (the "why", not just the "what")

- **No auto-submission, ever.** The Applier prepares kits; the human clicks
  Submit. Auto-filling LinkedIn/Naukri/Indeed forms violates their ToS and
  risks account bans — the worst outcome mid job-hunt. This is documented as
  a hard rule in `CONTRIBUTING.md`: PRs adding auto-submit are out of scope.
- **Scraping stays polite.** Rate-limit delays preserved everywhere;
  Scrapling's aggressive stealth fetcher was explicitly rejected in favor of
  its self-healing-selector feature only.
- **Tailoring never fabricates.** The tailor prompt hard-forbids inventing
  skills/employers/metrics; on Bedrock this is *additionally* enforced by a
  platform Guardrail, not just prompt wording.
- **Reranker narrows, LLM judges — never the same thing.** Repeatedly
  re-confirmed during the build: swapping the LLM fit-scoring stage out for
  reranker-only would silently reintroduce false positives on hard
  requirements. Kept as two distinct stages on purpose.
- **Everything fails soft.** Reranker errors → FAISS order. Notifier errors
  → logged, pipeline continues. Tailor/Review errors on one kit → that kit
  is skipped, others proceed. A single provider or stage failing should
  never take down a whole run.
- **We investigated and rejected "OmniRoute"-style free-Claude gateways.**
  A user question raised using a multi-provider proxy to route Claude Code
  traffic through "free Claude" tiers (Kiro AI, OpenCode Free). Declined:
  those free tiers are subsidized access from other products, used outside
  their licensed context — a ToS violation risk for the user's account, not
  a legitimate optimization. Documented here so the reasoning isn't lost:
  the project's OpenRouter/Nemotron fallback is legitimate (NVIDIA's own free
  model, offered directly on OpenRouter, no piggybacking); that proposal was
  a different and rejected pattern.
- **Privacy note on the free OpenRouter tier.** OpenRouter's `:free` endpoint
  logs requests and asks users not to send personal data — and resumes are
  personal data. This is called out in `config.py` and `.env.example`; users
  who want stricter terms can drop `:free` from `FALLBACK_MODEL_ID` /
  `RERANK_MODEL_ID` for the paid endpoint.

---

## 6. Frontend

Vanilla HTML/CSS/JS (no framework, no build step) styled as a premium dark
SaaS product (Linear/Vercel/Perplexity-inspired): deep near-black canvas,
glassy cards, indigo→violet gradient accents, an inline SVG icon system
(`ICONS` map in `app.js`, no external icon library dependency).

**Three top-level tabs:** Dashboard (funnel + stats), **Resume** (its own
screen — drag-and-drop upload, LinkedIn-style parsed profile card with
skills auto-categorized into Languages/ML/Deep Learning/LLM/Databases/
Deployment/Cloud with subtle per-category color variants), **Search**
(control panel with a gradient slider for match threshold, a large glowing
animated Launch button, a live per-source search stream, and an "Agent
status" timeline showing Resume/Scout/Matcher/Tailor agents as animated
cards with progress bars, elapsed time, and jobs-found counts — with raw
logs demoted to an expandable `<details>` panel), and **Applications**
(filterable tracker with a kit detail modal).

The Resume and Search screens were split from an earlier single merged
"Search jobs" tab specifically so a first-time user gets one clear focus per
screen instead of one long scrolling form, with a "Continue to search" CTA
bridging them.

---

## 7. Testing & CI

- [tests/test_smoke.py](tests/test_smoke.py) — tracker insert/dedupe/status
  lifecycle, stats, notifier formatting, config sanity.
- [tests/test_reranker.py](tests/test_reranker.py) — mocked reorder, no-key
  fallback, API-error fallback (no network/API key needed to run).
- `.github/workflows/ci.yml` — compileall over every module + `pytest -q` on
  every push/PR.
- Manually verified end-to-end at least twice with real API keys: a full
  scrape → match → tailor → review → track run producing genuinely tailored,
  non-fabricated kits (verified against the candidate's actual listed
  projects), and the live fallback proof described in §2.3.

---

## 8. Configuration reference

All settings live in `config.py`, sourced from `.env` (see `.env.example`
for the full annotated list). Key knobs:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `bedrock` \| `openrouter` |
| `MATCH_THRESHOLD` | `70` | Minimum LLM fit % to generate a kit |
| `OPENROUTER_API_KEY` | unset | Enables both the LLM fallback and the reranker |
| `FALLBACK_MODEL_ID` | `nvidia/nemotron-3-super-120b-a12b:free` | Fallback model on primary failure |
| `ENABLE_LLM_FALLBACK` | `true` | Master switch for the fallback wrapper |
| `ENABLE_RERANK` | `true` | Master switch for the reranker narrowing stage |
| `RERANK_MODEL_ID` | `nvidia/llama-nemotron-rerank-vl-1b-v2:free` | Reranker model |
| `RERANK_TOP_N` | `8` | How many candidates survive reranking into LLM scoring |
| `LLM_CONCURRENCY` | `4` | Thread-pool width for matcher/tailor/review LLM calls |
| `SCRAPE_CONCURRENCY` | `4` | Thread-pool width for jobspy (role × city) searches |
| `ENABLE_REVIEW_CREW` | `true` | Toggle the critic→editor cover-letter polish pass |
| `GUARDRAIL_ID` / `GUARDRAIL_VERSION` | unset | Bedrock Guardrail, set by `provision_bedrock.py` |
| `MANAGED_KB_ID` | unset | Bedrock Managed Knowledge Base id (AgentCore deploy only) |

---

## 9. Known limitations (be upfront about these)

- Scraping reliability depends on IP reputation — datacenter IPs (cloud
  hosting) get rate-limited/blocked more than a residential IP; this is why
  local/cron execution is the recommended deployment, not a public server.
- Gemini's free tier caps around 20 requests/day, which is why the
  OpenRouter fallback and the reranker's separate free budget both exist —
  but a heavy user will still eventually want a paid primary tier.
- The dashboard has **no authentication** — it is a personal, single-user
  tool by design; do not expose it on the open internet without adding auth.
- Company career-page scraping (`enable_company_pages`) is off by default —
  selectors are per-company and go stale; only two example companies are
  configured as a starting point for contributors.

---

## 10. File map

```
job_hunt_ai/
├── agents/                    # the pipeline's LLM-driven nodes
│   ├── scout_agent.py         #   resume parse (cached) + job scraping
│   ├── matcher_agent.py       #   FAISS → reranker → LLM fit-scoring (3 stages)
│   ├── tailor_agent.py        #   concurrent kit generation (summary/bullets/letter)
│   └── applier_agent.py       #   save kits, track in SQLite, notify
├── graph/
│   ├── state.py               # shared PipelineState (TypedDict)
│   └── pipeline.py            # LangGraph wiring: scout→matcher→tailor→review→applier
├── tools/
│   ├── resume_parser_tool.py  # PDF → structured ParsedResume (LLM + Pydantic)
│   ├── scraper_tool.py        # jobspy (concurrent) + Scrapling (self-healing)
│   ├── vector_store_tool.py   # FAISS index + Job/JobMatch dataclasses
│   ├── reranker_tool.py       # OpenRouter /rerank client, fails soft
│   ├── notifier_tool.py       # console / Slack / email, fails soft
│   └── bedrock_kb_tool.py     # Bedrock Managed KB retrieval (AgentCore path)
├── review/
│   ├── autogen_crew.py        # concurrent critic→editor cover-letter polish
│   └── tracker.py             # SQLite application tracker (no ORM)
├── api/server.py               # FastAPI backend, two-phase background run, live log stream
├── frontend/                   # premium dark dashboard — vanilla HTML/CSS/JS
│   ├── index.html              #   Dashboard · Resume · Search · Applications
│   ├── style.css                #   dark glassy SaaS design system
│   └── app.js                   #   inline SVG icons, drag-drop, live agent timeline
├── job_hunt_agent.py            # AWS Bedrock AgentCore entrypoint
├── provision_bedrock.py         # one-time Guardrail provisioning script
├── config.py                    # settings + get_llm() factory (provider + fallback + rerank)
├── main.py                      # CLI: check | run | serve
├── tests/                       # pytest — tracker, notifier, reranker (mocked)
├── .github/workflows/ci.yml     # compileall + pytest on push/PR
├── README.md · BEDROCK.md · DEPLOYMENT.md · CONTRIBUTING.md · LICENSE
└── requirements.txt · requirements-bedrock.txt · .env.example
```

---

## 11. Status

Functionally complete and manually verified end-to-end (scrape → match →
tailor → review → track), with a premium redesigned frontend in progress
(Resume/Search split into separate screens, agent-status timeline, live
search stream — see recent commits for the latest UI state). Not yet
pushed to a public GitHub remote; open-source readiness checklist
(security banner, `SECURITY.md`, screenshots, CORS hardening) was scoped
but not yet fully executed — treat that as the next milestone before
publishing.
