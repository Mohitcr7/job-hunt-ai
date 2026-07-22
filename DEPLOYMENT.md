# Deployment Guide

Job Hunt AI is a **personal** tool — each user runs their own instance with their own resume, API key, and tracker database. That shapes the deployment options below.

> **Heads up on hosting scrapers:** LinkedIn/Indeed/Naukri aggressively rate-limit datacenter IPs. Scraping is most reliable from a **residential IP** — i.e., your own laptop. The recommended setup is therefore local or Docker-on-your-machine, with cloud hosting as a convenience for the dashboard/tracker.

## Option 1 — Local (recommended)

```bash
python main.py serve        # dashboard at http://localhost:8000
```

### Daily automated runs (macOS/Linux)

Add a cron job that runs the pipeline every morning:

```bash
crontab -e
# run at 8:30 AM daily, log to file
30 8 * * * cd /path/to/job_hunt_ai && ./venv/bin/python main.py run >> cron.log 2>&1
```

On macOS you can also use a LaunchAgent; on Windows, Task Scheduler with `python main.py run`.

## Option 2 — Docker (local or any VPS)

```bash
cp .env.example .env        # fill in GOOGLE_API_KEY
docker compose up --build -d
```

- Dashboard: `http://localhost:8000`
- Your resume, tracker DB, and application kits persist in `./data` and `./output` on the host.
- Daily runs inside the container: `docker compose exec jobhunt python main.py run` (cron that from the host).

## Option 3 — Free cloud hosting

### Hugging Face Spaces (easiest free option)

1. Create a new **Docker** Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Push this repo to the Space (it picks up the `Dockerfile` automatically).
3. In Space settings → *Variables and secrets*, add `GOOGLE_API_KEY` and set the Space port to `8000` (add `app_port: 8000` to the Space README front-matter).
4. **Make the Space private** — the dashboard has no authentication and holds your resume.

Note: free Spaces sleep when idle and have ephemeral storage — your tracker DB resets on restart. Add a persistent-storage upgrade or export your data regularly.

### Railway / Render

Both detect the `Dockerfile` automatically:

1. New project → deploy from your GitHub repo.
2. Set env vars from `.env.example` (at minimum `GOOGLE_API_KEY`).
3. Attach a persistent volume mounted at `/app/data` so the tracker survives redeploys.
4. Railway's free trial / Render's free tier both work, but the image is ~2GB (Playwright + PyTorch) — expect slow cold builds.

### Scheduling in the cloud

Use the platform's cron feature (Railway cron jobs, Render cron jobs, or GitHub Actions `schedule:`) to hit the pipeline endpoint daily:

```bash
curl -X POST https://your-app.example.com/api/pipeline/run \
  -H "Content-Type: application/json" -d '{}'
```

## Security checklist before exposing to the internet

- [ ] The dashboard has **no built-in auth** — put it behind a login (Cloudflare Access, Tailscale, or HTTP basic auth on a reverse proxy) or keep it private.
- [ ] Never commit `.env`, `data/`, or `output/` (the `.gitignore` already covers this).
- [ ] Restrict CORS in `api/server.py` (`allow_origins`) to your actual domain if hosting publicly.

## Resource requirements

| Component | Requirement |
|---|---|
| RAM | ~2 GB (sentence-transformers + Chromium) |
| Disk | ~3 GB (PyTorch, model weights, Chromium) |
| CPU | Anything — embeddings on CPU take seconds for hundreds of jobs |
| GPU | Not needed |
