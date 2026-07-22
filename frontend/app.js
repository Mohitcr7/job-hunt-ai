// Job Hunt AI — dashboard logic (vanilla JS, no build step)
// Talks to the FastAPI backend under /api/*

const API = "";
const STATUSES = ["prepared", "applied", "interviewing", "offer", "rejected", "archived"];

const $ = (sel) => document.querySelector(sel);

async function api(path, options = {}) {
  const res = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Icons (inline SVG — one consistent stroke set, no external dependency)
// ---------------------------------------------------------------------------
const ICONS = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
  search: '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
  layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="14" y2="17"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
  sliders: '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
  briefcase: '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
  pin: '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  zap: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
  spinner: '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
  cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
  terminal: '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  refresh: '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
  mail: '<rect x="2" y="4" width="20" height="16" rx="2"/><polyline points="22 6 12 13 2 6"/>',
  phone: '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.8 19.8 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.12 4.18 2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>',
  edit: '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>',
  award: '<circle cx="12" cy="8" r="6"/><path d="M15.5 13.5 17 22l-5-3-5 3 1.5-8.5"/>',
  "arrow-right": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
};
function svg(name, cls) { return `<svg class="icon ${cls || ""}" viewBox="0 0 24 24">${ICONS[name] || ""}</svg>`; }
function injectIcons(root) {
  (root || document).querySelectorAll("[data-icon]").forEach((el) => {
    if (el.dataset.filled) return;
    el.innerHTML = svg(el.getAttribute("data-icon"));
    el.dataset.filled = "1";
  });
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
    // the merged Search Job screen uses the full-width split layout
    document.querySelector(".main").classList.toggle("wide", btn.dataset.tab === "search");
    if (btn.dataset.tab === "dashboard") loadDashboard();
    if (btn.dataset.tab === "applications") loadApplications();
    if (btn.dataset.tab === "search") { loadResume(); loadSettings(); }
    window.scrollTo(0, 0);
  });
});

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
async function loadDashboard() {
  try {
    const stats = await api("/api/stats");
    const by = stats.by_status || {};
    $("#stat-total").textContent = stats.total;
    $("#stat-prepared").textContent = by.prepared || 0;
    $("#stat-applied").textContent = by.applied || 0;
    $("#stat-interviewing").textContent = by.interviewing || 0;
    $("#stat-offers").textContent = by.offer || 0;
    $("#stat-avg").textContent = stats.avg_match_score ? stats.avg_match_score + "%" : "—";

    const max = Math.max(1, ...STATUSES.map((s) => by[s] || 0));
    $("#funnel").innerHTML = STATUSES.filter((s) => s !== "archived").map((s) => {
      const count = by[s] || 0;
      return `<div class="funnel-row"><div class="funnel-label">${s}</div>
        <div class="funnel-track"><div class="funnel-bar" style="width:${(count / max) * 100}%"></div></div>
        <div class="funnel-count">${count}</div></div>`;
    }).join("");

    const apps = await api("/api/applications");
    $("#recent-apps").innerHTML = apps.length
      ? apps.slice(0, 6).map(appCardSmall).join("")
      : `<p class="muted">Nothing yet — run the pipeline to find your first matches.</p>`;
  } catch (e) { console.error(e); }
}

function appCardSmall(app) {
  return `<div class="job-card">${scoreRing(app.match_score)}
    <div class="job-info"><div class="job-title">${esc(app.job_title)}</div>
      <div class="job-meta"><span>${esc(app.company)}</span><span>${esc(app.location)}</span></div></div>
    <span class="badge ${esc(app.status)}">${esc(app.status)}</span></div>`;
}

function scoreRing(score) {
  return `<div class="score-ring ${score >= 80 ? "high" : ""}" style="--pct:${score}">${score}%</div>`;
}

// ---------------------------------------------------------------------------
// Run pipeline + live agent status
// ---------------------------------------------------------------------------
let pollTimer = null;
let prelimKey = "";
let lastKitCount = -1;

const PHASE_LABELS = {
  scraping: "Scraping job boards…",
  matching: "Matching to your resume…",
  reviewing: "Tailoring top candidates…",
  done: "Complete",
};

async function loadSettings() {
  try {
    const s = await api("/api/settings");
    $("#input-roles").value = s.preferences.roles.join(", ");
    $("#input-locations").value = s.preferences.locations.join(", ");
    $("#input-threshold").value = s.match_threshold;
    $("#input-threshold").dispatchEvent(new Event("input"));
  } catch (e) { /* backend not up yet */ }
}

$("#btn-run").addEventListener("click", async () => {
  const parse = (v) => v.split(",").map((x) => x.trim()).filter(Boolean);
  const btn = $("#btn-run");
  try {
    btn.classList.add("loading");
    btn.disabled = true;
    await api("/api/pipeline/run", {
      method: "POST",
      body: JSON.stringify({
        search_terms: parse($("#input-roles").value),
        locations: parse($("#input-locations").value),
        hours_old: parseInt($("#input-hours").value) || 24,
        min_score: parseInt($("#input-threshold").value) || 70,
      }),
    });
    $("#run-status").textContent = "";
    $("#run-log").textContent = "";
    $("#run-results-panel").classList.add("hidden");
    $("#agent-empty").classList.add("hidden");
    $("#agent-panel").classList.remove("hidden");
    prelimKey = "";
    lastKitCount = -1;
    startPolling();
  } catch (e) {
    btn.classList.remove("loading");
    btn.disabled = false;
    $("#run-status").textContent = "⚠ " + e.message;
  }
});

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(checkPipeline, 1000);
  checkPipeline();
}

function renderLog(lines) {
  const el = $("#run-log");
  if (!el) return;
  if (lines && lines.length) {
    el.textContent = lines.slice(-40).join("\n");
    el.scrollTop = el.scrollHeight;
  }
}

async function checkPipeline() {
  try {
    const st = await api("/api/pipeline/status");
    renderLog(st.log);
    updateLiveUI(st);
    const dot = $("#pipeline-dot"), pill = $("#pipeline-pill-text"), btn = $("#btn-run");

    if (!st.result) {
      if (st.kits_done && st.kits_done.length) {
        if (st.kits_done.length !== lastKitCount) { renderStreamingKits(st.kits_done, st); lastKitCount = st.kits_done.length; }
      } else if (st.preliminary && st.preliminary.length) {
        const key = st.preliminary.length + ":" + st.phase;
        if (key !== prelimKey) { renderPreliminary(st.preliminary, st); prelimKey = key; }
      }
    }

    if (st.running) {
      dot.className = "dot running";
      pill.textContent = PHASE_LABELS[st.phase] || "Running…";
      btn.disabled = true; btn.classList.add("loading");
    } else {
      btn.disabled = false; btn.classList.remove("loading");
      clearInterval(pollTimer);
      if (st.error) {
        dot.className = "dot err"; pill.textContent = "Last run failed";
        $("#run-status").textContent = "⚠ " + st.error;
      } else if (st.result) {
        dot.className = "dot ok"; pill.textContent = "Last run OK";
        renderRunResults(st.result); loadDashboard();
      } else {
        dot.className = "dot idle"; pill.textContent = "Pipeline idle";
      }
    }
  } catch (e) { /* server restarting */ }
}

// ---- live UI: overall progress + agent timeline + search stream + stats ----
function jobsFound(st) {
  const log = (st.log || []).join("\n");
  const m = log.match(/Total unique jobs collected:\s*(\d+)/i) || log.match(/found\s+(\d+)\s+jobs/i);
  if (m) return parseInt(m[1]);
  if (st.result) return st.result.jobs_scraped;
  if (st.preliminary && st.preliminary.length) return st.preliminary.length;
  return 0;
}

function overallPct(st) {
  if (st.result || st.phase === "done") return 100;
  if (st.phase === "reviewing" && st.kits_total) {
    return Math.min(98, 80 + Math.round(((st.kits_done || []).length / st.kits_total) * 18));
  }
  return { scraping: 22, matching: 58, reviewing: 80 }[st.phase] || 6;
}

function agentCardHTML(a) {
  const badge = a.status === "done" ? "✓ Done" : a.status === "running" ? "Running…" : "Waiting";
  return `<div class="agent-card ${a.status}">
    <div class="agent-ic">${svg(a.icon)}</div>
    <div class="agent-main">
      <div class="agent-name">${a.name}</div>
      <div class="agent-sub">${a.sub}</div>
      <div class="agent-bar-track"><div class="agent-bar"></div></div>
    </div>
    <div class="agent-badge ${a.status}">${badge}</div></div>`;
}

function renderAgentTimeline(st) {
  const order = ["scraping", "matching", "reviewing", "done"];
  const cur = order.indexOf(st.phase);
  const done = st.result || st.phase === "done";
  const stat = (i) => done ? "done" : (cur > i ? "done" : (cur === i ? "running" : "waiting"));
  const kitsDone = (st.kits_done || []).length, kitsTotal = st.kits_total || 0;
  const jf = jobsFound(st);
  const stages = [
    { icon: "file", name: "Resume agent", sub: "Parsed & indexed your profile", status: (st.phase && st.phase !== "idle") ? "done" : "waiting" },
    { icon: "search", name: "Scout agent", sub: jf ? `Scraped ${jf} jobs · LinkedIn · Indeed · Naukri` : "Scraping LinkedIn, Indeed & Naukri", status: stat(0) },
    { icon: "target", name: "Matcher agent", sub: "FAISS → reranker → LLM fit-scoring", status: stat(1) },
    { icon: "edit", name: "Tailor agent", sub: kitsTotal ? `${kitsDone} of ${kitsTotal} applications tailored` : "Writing tailored applications", status: stat(2) },
  ];
  $("#agent-timeline").innerHTML = stages.map(agentCardHTML).join("");
}

function renderStream(st) {
  const log = (st.log || []).join("\n");
  const past = ["matching", "reviewing", "done"].includes(st.phase) || st.result;
  const scraping = st.phase === "scraping";
  const jobspyDone = past || /jobspy total/i.test(log);
  const naukriDone = past || /Naukri total/i.test(log);
  const stt = (d) => d ? "done" : (scraping ? "running" : "waiting");
  const sources = [["LinkedIn", stt(jobspyDone)], ["Indeed", stt(jobspyDone)], ["Naukri", stt(naukriDone)]];
  $("#stream-sources").innerHTML = sources.map(([n, s]) => `
    <div class="src ${s}">
      <div class="src-head"><span class="src-name">${n}</span>
        <span class="src-state ${s}">${s === "done" ? "done" : s === "running" ? "searching…" : "queued"}</span></div>
      <div class="src-track"><div class="src-fill"></div></div>
    </div>`).join("");
  $("#stream-ticker").textContent = (st.log || []).slice(-1)[0] || "Working…";
}

function renderRunStats(st) {
  const el = $("#run-stats");
  if (!el) return;
  if ((!st.phase || st.phase === "idle") && !st.result) { el.innerHTML = ""; return; }
  let elapsed = "";
  if (st.started_at) {
    const secs = Math.max(0, Math.floor((Date.now() - new Date(st.started_at).getTime()) / 1000));
    elapsed = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
  }
  const jf = jobsFound(st);
  el.innerHTML = `${elapsed ? `<span class="stat-pill">${svg("clock")} ${elapsed}</span>` : ""}` +
    `${jf ? `<span class="stat-pill">${svg("briefcase")} ${jf} jobs</span>` : ""}`;
}

function updateLiveUI(st) {
  const idle = (!st.phase || st.phase === "idle") && !st.result && !st.running;
  $("#agent-empty").classList.toggle("hidden", !idle);
  $("#agent-panel").classList.toggle("hidden", idle);
  $("#log-details").classList.toggle("hidden", !(st.log && st.log.length));
  $("#search-stream").classList.toggle("hidden", !st.running);
  if (st.running) renderStream(st);
  if (!idle) {
    renderAgentTimeline(st);
    renderRunStats(st);
    const pct = overallPct(st);
    $("#op-bar").style.width = pct + "%";
    $("#op-pct").textContent = pct + "%";
    $("#op-label").textContent = st.error ? "Run failed" : (PHASE_LABELS[st.phase] || (st.result ? "Complete" : "Starting…"));
  }
}

// ---- results rendering ----
function jobCardHTML(m) {
  return `<div class="job-card">${scoreRing(m.score)}
    <div class="job-info"><div class="job-title">${esc(m.title)}</div>
      <div class="job-meta"><span>${esc(m.company)}</span><span>${esc(m.location)}</span><span>${esc(m.platform)}</span></div></div>
    <div class="job-actions"><a class="btn-ghost" href="${esc(m.url)}" target="_blank" rel="noopener">View job ↗</a></div></div>`;
}

function renderPreliminary(list, st) {
  $("#run-results-panel").classList.remove("hidden");
  $("#run-results-title").textContent = "Scraped jobs — browse while the agents work";
  const reviewing = st.phase === "reviewing";
  $("#run-summary").innerHTML = `<span><b>${list.length}</b> jobs above 50% match</span>
    <span class="reviewing-note">${reviewing ? "⟳ tailoring the top candidates in the background…" : "matching…"}</span>`;
  $("#top-matches").innerHTML = list.map(jobCardHTML).join("");
}

function renderStreamingKits(kits, st) {
  $("#run-results-panel").classList.remove("hidden");
  $("#run-results-title").textContent = "Preparing your applications…";
  const total = st.kits_total || kits.length;
  $("#run-summary").innerHTML = `<span class="reviewing-note">⟳ ${kits.length} of ${total} applications tailored &amp; saved…</span>`;
  $("#top-matches").innerHTML = kits.map(jobCardHTML).join("");
}

function renderRunResults(r) {
  $("#run-results-panel").classList.remove("hidden");
  $("#run-results-title").textContent = "Top matches — tailored & tracked";
  $("#run-summary").innerHTML = `<span><b>${r.jobs_scraped}</b> jobs scraped</span>
    <span><b>${r.matches}</b> matches</span><span><b>${r.kits_created}</b> kits created</span>
    <span><b>${r.new_applications}</b> new in tracker</span>`;
  $("#top-matches").innerHTML = (r.top_matches || []).map(jobCardHTML).join("");
}

// ---------------------------------------------------------------------------
// Applications
// ---------------------------------------------------------------------------
let currentFilter = "";
let appsCache = [];

function renderFilters() {
  $("#status-filters").innerHTML = ["all", ...STATUSES].map((s) =>
    `<button class="filter-chip ${((s === "all" && !currentFilter) || s === currentFilter) ? "active" : ""}" data-status="${s}">${s}</button>`
  ).join("");
  document.querySelectorAll(".filter-chip").forEach((chip) =>
    chip.addEventListener("click", () => { currentFilter = chip.dataset.status === "all" ? "" : chip.dataset.status; loadApplications(); })
  );
}

async function loadApplications() {
  renderFilters();
  try {
    const qs = currentFilter ? `?status=${currentFilter}` : "";
    appsCache = await api("/api/applications" + qs);
    $("#applications-list").innerHTML = appsCache.length
      ? appsCache.map(appCardFull).join("")
      : `<p class="muted">No applications${currentFilter ? ` with status "${currentFilter}"` : ""} yet.</p>`;

    document.querySelectorAll(".status-select").forEach((sel) =>
      sel.addEventListener("change", async (e) => {
        await api(`/api/applications/${e.target.dataset.id}`, { method: "PATCH", body: JSON.stringify({ status: e.target.value }) });
        loadApplications(); loadDashboard();
      })
    );
    document.querySelectorAll("[data-kit]").forEach((btn) =>
      btn.addEventListener("click", () => openKitModal(parseInt(btn.dataset.kit)))
    );
  } catch (e) { console.error(e); }
}

function appCardFull(app) {
  const options = STATUSES.map((s) => `<option value="${s}" ${s === app.status ? "selected" : ""}>${s}</option>`).join("");
  return `<div class="job-card">${scoreRing(app.match_score)}
    <div class="job-info"><div class="job-title">${esc(app.job_title)}</div>
      <div class="job-meta"><span>${esc(app.company)}</span><span>${esc(app.location)}</span>
        <span>${esc(app.platform)}</span><span>${new Date(app.created_at).toLocaleDateString()}</span></div></div>
    <div class="job-actions">
      <button class="btn-ghost" data-kit="${app.id}">📄 Kit</button>
      <a class="btn-ghost" href="${esc(app.job_url)}" target="_blank" rel="noopener">Apply ↗</a>
      <select class="status-select" data-id="${app.id}">${options}</select></div></div>`;
}

function openKitModal(id) {
  const app = appsCache.find((a) => a.id === id);
  if (!app) return;
  $("#modal-content").innerHTML = `
    <h2>${esc(app.job_title)}</h2>
    <div class="job-meta"><span>${esc(app.company)}</span>
      <span class="badge ${esc(app.status)}">${esc(app.status)}</span><span>${app.match_score}% match</span></div>
    <h3>Tailored summary</h3><p>${esc(app.tailored_summary)}</p>
    <h3>Tailored resume bullets</h3><ul>${app.tailored_bullets.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
    <h3>ATS keywords to add</h3>
    <div class="chips">${app.keywords_to_add.length
      ? app.keywords_to_add.map((k) => `<span class="chip">${esc(k)}</span>`).join("")
      : '<span class="muted">None — resume already covers the JD</span>'}</div>
    <h3>Cover letter</h3><div class="cover-letter">${esc(app.cover_letter)}</div>
    <button class="btn-ghost copy-btn" id="copy-letter">Copy cover letter</button>`;
  $("#modal").classList.remove("hidden");
  $("#copy-letter").addEventListener("click", () => {
    navigator.clipboard.writeText(app.cover_letter);
    $("#copy-letter").textContent = "Copied ✓";
  });
}

$("#modal-close").addEventListener("click", () => $("#modal").classList.add("hidden"));
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); });

// ---------------------------------------------------------------------------
// Resume — upload (drag & drop) + parsed profile card
// ---------------------------------------------------------------------------
function initDropzone() {
  const dz = $("#dropzone"), input = $("#resume-file");
  if (!dz || !input) return;
  const showFile = () => {
    const f = input.files[0];
    if (f) { const el = $("#dz-file"); el.textContent = f.name; el.classList.remove("hidden"); dz.classList.add("has-file"); }
  };
  dz.addEventListener("click", () => input.click());
  $("#dz-browse").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
  input.addEventListener("change", showFile);
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  dz.addEventListener("dragleave", (e) => { if (!dz.contains(e.relatedTarget)) dz.classList.remove("dragover"); });
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; showFile(); }
  });
}

$("#btn-upload").addEventListener("click", async () => {
  const file = $("#resume-file").files[0];
  const status = $("#upload-status");
  if (!file) { status.className = "upload-status err"; status.textContent = "Choose a PDF first."; return; }
  const btn = $("#btn-upload");
  const form = new FormData();
  form.append("file", file);
  btn.disabled = true;
  status.className = "upload-status";
  status.textContent = "Uploading and parsing… (a few seconds)";
  try {
    const res = await fetch(API + "/api/resume/upload", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    status.className = "upload-status ok";
    status.textContent = "✓ " + body.message;
    loadResume();
  } catch (e) {
    status.className = "upload-status err";
    status.textContent = "⚠ " + e.message;
  } finally { btn.disabled = false; }
});

// group skills into named categories with subtle per-category hues
const SKILL_GROUPS = [
  { key: "llm", label: "LLM & GenAI", kw: ["llm", "rag", "langchain", "llama", "hugging", "prompt", "genai", "agent", "fine-tun", "embedding", "retrieval", "hallucinat", "openai", "gemini", "claude", "vector"] },
  { key: "dl", label: "Deep Learning", kw: ["pytorch", "tensorflow", "keras", "cnn", "rnn", "lstm", "deep learning", "computer vision", "neural", "attention", "transformer", "gan", "nlp"] },
  { key: "ml", label: "Machine Learning", kw: ["scikit", "sklearn", "xgboost", "lightgbm", "machine learning", "shap", "random forest", "regression", "classification", "feature", "smote", "clustering", "statistic", "pandas", "numpy", "matplotlib", "seaborn"] },
  { key: "databases", label: "Databases", kw: ["postgres", "mysql", "mongo", "redis", "faiss", "chroma", "neo4j", "pinecone", "sqlite", "database", "elasticsearch", "sql"] },
  { key: "deployment", label: "Deployment", kw: ["docker", "kubernetes", "fastapi", "flask", "rest", "mlflow", "ci/cd", "deploy", "airflow", "git", "api", "microservice", "celery", "mlops"] },
  { key: "cloud", label: "Cloud", kw: ["aws", "gcp", "azure", "google cloud", "s3", "lambda", "bedrock", "sagemaker", "vertex", "cloud"] },
  { key: "languages", label: "Languages", kw: ["python", "java", "c++", "javascript", "typescript", "scala", "bash", "shell"] },
];
function categorize(skill) {
  const s = skill.toLowerCase();
  for (const g of SKILL_GROUPS) { if (g.kw.some((k) => s.includes(k))) return g.key; }
  return "core";
}
function renderSkills(skills) {
  const groups = {};
  skills.forEach((s) => { const k = categorize(s); (groups[k] = groups[k] || []).push(s); });
  const order = [...SKILL_GROUPS.map((g) => g.key), "core"];
  const labelOf = (k) => (SKILL_GROUPS.find((g) => g.key === k) || {}).label || "Core";
  return order.filter((k) => groups[k] && groups[k].length).map((k) =>
    `<div class="skill-cat"><div class="skill-cat-name">${labelOf(k)}</div>
      <div class="chips">${groups[k].map((s) => `<span class="chip cat-${k}">${esc(s)}</span>`).join("")}</div></div>`
  ).join("");
}

async function loadResume() {
  try {
    const r = await api("/api/resume");
    const skills = r.skills || [];
    const initials = (r.name || "?").trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
    const xp = (r.work_experience || []).map((w) =>
      `<div class="xp-item"><div class="xp-role">${esc(w.role)} — ${esc(w.company)}</div><div class="xp-meta">${esc(w.duration)}</div></div>`
    ).join("") || '<p class="muted">None parsed</p>';
    const edu = (r.education || []).map((e2) =>
      `<div class="xp-item"><div class="xp-role">${esc(e2.degree)}</div><div class="xp-meta">${esc(e2.institution)} · ${esc(e2.year)}</div></div>`
    ).join("") || '<p class="muted">None parsed</p>';

    $("#resume-view").innerHTML = `
      <div class="profile-card">
        <div class="profile-banner"></div>
        <div class="profile-top">
          <div class="avatar">${esc(initials)}</div>
          <div class="profile-id">
            <div class="profile-name">${esc(r.name)}</div>
            <div class="profile-contact">
              ${r.email ? `<span>${svg("mail")} ${esc(r.email)}</span>` : ""}
              ${r.phone ? `<span>${svg("phone")} ${esc(r.phone)}</span>` : ""}
              ${r.location ? `<span>${svg("pin")} ${esc(r.location)}</span>` : ""}
            </div>
          </div>
          <div class="parsed-badge">${svg("check")} Parsed successfully</div>
        </div>
        <div class="profile-body">
          <p class="profile-summary">${esc(r.summary)}</p>
          <div class="profile-block">
            <div class="block-label">${svg("zap")} Skills · ${skills.length}</div>
            <div class="skill-cats">${renderSkills(skills)}</div>
          </div>
          <div class="profile-block"><div class="block-label">${svg("briefcase")} Experience</div>${xp}</div>
          <div class="profile-block"><div class="block-label">${svg("award")} Education</div>${edu}</div>
          <button class="btn-replace" id="replace-resume">${svg("refresh")} Replace resume</button>
        </div>
      </div>`;
  } catch (e) {
    $("#resume-view").innerHTML = `<div class="empty-state" style="padding:38px 20px">
      <div class="empty-icon">${svg("file")}</div>
      <div class="empty-title">No resume yet</div>
      <div class="empty-sub">Drop a PDF above and we'll parse it into a profile.</div></div>`;
  }
}

// replace-resume (rendered dynamically → delegate)
document.addEventListener("click", (e) => {
  if (e.target.closest("#replace-resume")) {
    $("#dropzone").scrollIntoView({ behavior: "smooth", block: "center" });
    $("#resume-file").click();
  }
});

function initSlider() {
  const s = $("#input-threshold");
  if (!s) return;
  const upd = () => { $("#threshold-val").textContent = s.value + "%"; s.style.backgroundSize = s.value + "% 100%"; };
  s.addEventListener("input", upd);
  upd();
}

// draggable divider between the resume pane and the search pane
function initSplitResizer() {
  const split = $("#split"), resizer = $("#split-resizer"), left = $("#pane-resume");
  if (!split || !resizer || !left) return;
  const MIN = 24, MAX = 68; // clamp the left pane between 24% and 68% of the split
  let dragging = false;

  const onMove = (clientX) => {
    const rect = split.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(MIN, Math.min(MAX, pct));
    left.style.flexBasis = pct + "%";
  };
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    split.classList.remove("resizing");
    window.removeEventListener("mousemove", mouseMove);
    window.removeEventListener("touchmove", touchMove);
  };
  const mouseMove = (e) => onMove(e.clientX);
  const touchMove = (e) => { if (e.touches[0]) onMove(e.touches[0].clientX); };

  resizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    split.classList.add("resizing");
    window.addEventListener("mousemove", mouseMove);
    window.addEventListener("mouseup", stop, { once: true });
  });
  resizer.addEventListener("touchstart", (e) => {
    dragging = true;
    split.classList.add("resizing");
    window.addEventListener("touchmove", touchMove, { passive: true });
    window.addEventListener("touchend", stop, { once: true });
  }, { passive: true });
  // double-click resets to the default balance
  resizer.addEventListener("dblclick", () => { left.style.flexBasis = ""; });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
injectIcons();
initDropzone();
initSlider();
initSplitResizer();
loadDashboard();
loadSettings();
loadResume();
checkPipeline();
