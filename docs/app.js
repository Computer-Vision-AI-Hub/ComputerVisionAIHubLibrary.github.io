/* ============================================================
   ComputerVisionAIHub — catalog logic (vanilla JS, no build step)
   Flow:  fetch models.json -> render cards -> filter on input
   The page never hardcodes models; it all comes from the manifest.
   ============================================================ */

// Change this to your GHCR/Docker Hub image.
const DOCKER_IMAGE = "docker.io/lukasiktar/computervisionaihub";

// Module-level state
let ALL_MODELS = [];
let activeTask = "all";
let query = "";

const els = {
  grid:    document.getElementById("grid"),
  count:   document.getElementById("count"),
  state:   document.getElementById("state"),
  search:  document.getElementById("search"),
  filters: document.getElementById("task-filters"),
};

// ---- small helper: escape text before inserting into HTML ----
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---- 1. LOAD ----
async function load() {
  try {
    // cache:no-store so contributors see new models without a hard refresh
    const res = await fetch("models.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    ALL_MODELS = Array.isArray(data.models) ? data.models : [];
    buildFilters();
    render();
  } catch (err) {
    showState(`Could not load models.json (${esc(err.message)}). ` +
              `If you opened this file directly, serve it instead: ` +
              `run "python -m http.server" in the docs/ folder.`);
  }
}

// ---- 2. BUILD TASK FILTER PILLS from the data itself ----
function buildFilters() {
  const tasks = ["all", ...new Set(ALL_MODELS.map((m) => m.task).filter(Boolean))];
  els.filters.innerHTML = tasks
    .map((t) => `<button class="pill" data-task="${esc(t)}"
                  aria-pressed="${t === activeTask}">${esc(t)}</button>`)
    .join("");

  els.filters.querySelectorAll(".pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeTask = btn.dataset.task;
      els.filters.querySelectorAll(".pill")
        .forEach((b) => b.setAttribute("aria-pressed", b === btn));
      render();
    });
  });
}

// ---- 3. FILTER + RENDER ----
function render() {
  const q = query.trim().toLowerCase();
  const list = ALL_MODELS.filter((m) => {
    const taskOk = activeTask === "all" || m.task === activeTask;
    const hay = [m.name, m.summary, m.dataset, ...(m.classes || [])]
      .join(" ").toLowerCase();
    const searchOk = !q || hay.includes(q);
    return taskOk && searchOk;
  });

  els.count.textContent =
    `${list.length} model${list.length === 1 ? "" : "s"}` +
    (activeTask === "all" ? "" : ` · ${activeTask}`);

  if (!list.length) {
    showState("No models match. Try clearing the search or filter.");
    return;
  }
  els.state.hidden = true;
  els.grid.innerHTML = list.map(card).join("");
  wireCopyButtons();
}

// ---- 4. ONE CARD (template per model) ----
function card(m) {
  const classes = m.classes || [];
  const shown = classes.slice(0, 6);
  const extra = classes.length - shown.length;

  const chips =
    shown.map((c) => `<span class="chip">${esc(c)}</span>`).join("") +
    (extra > 0 ? `<span class="chip more">+${extra}</span>` : "");

  const metric = (label, val) => {
    const pct = Math.round((val ?? 0) * 100);
    return `<div class="metric">
      <span class="metric-label">${label}</span>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <span class="metric-val">${(val ?? 0).toFixed(2)}</span>
    </div>`;
  };

  const runCmd =
    `docker run -v $(pwd):/data ${DOCKER_IMAGE} ${m.download} /data/test.jpg`;

  const imageBlock = m.image
    ? `<div class="card-image">
         <img src="${esc(m.image)}" alt="Example detections from ${esc(m.name)}" loading="lazy"
              onerror="this.closest('.card-image').classList.add('placeholder'); this.remove();">
       </div>`
    : `<div class="card-image placeholder" role="img" aria-label="No preview image yet"></div>`;

  return `<article class="model-card">
    <span class="card-tab">${esc(m.id)}</span>

    ${imageBlock}

    <div class="card-head">
      <h2 class="card-name">${esc(m.name)}</h2>
      <p class="card-summary">${esc(m.summary || "")}</p>
    </div>

    <div class="badges">
      <span class="badge base">${esc(m.base_model || "yolo")}</span>
      <span class="badge task-${esc(m.task || "detection")}">${esc(m.task || "detection")}</span>
      <span class="badge">${esc(m.image_size || 640)}px</span>
    </div>

    <div class="chips">${chips}</div>

    <div class="metrics">
      ${m.metrics ? metric("mAP@50", m.metrics.mAP50) : ""}
      ${m.metrics ? metric("mAP@50-95", m.metrics.mAP50_95) : ""}
      ${m.metrics && m.metrics.precision != null ? metric("precision", m.metrics.precision) : ""}
      ${m.metrics && m.metrics.recall != null ? metric("recall", m.metrics.recall) : ""}
    </div>

    <div class="card-meta">
      <span>${m.size_mb ?? "?"} MB</span>
      <span>license: ${esc(m.dataset_license || "see dataset")}</span>
    </div>

    <div class="actions">
      <a class="btn primary" href="${esc(m.download)}" download>Download .pt</a>
      ${m.onnx ? `<a class="btn secondary" href="${esc(m.onnx)}" download>.onnx</a>` : ""}
      ${m.dataset ? `<a class="btn secondary" href="${esc(m.dataset)}" target="_blank" rel="noopener">Dataset</a>` : ""}
    </div>

    <div class="run-group">
      <span class="run-label">Model URL — paste into localhost:7860</span>
      <div class="run">
        <button class="copy-btn" data-cmd="${esc(m.download)}">copy</button>
        <code>${esc(m.download)}</code>
      </div>
    </div>

    <div class="run-group">
      <span class="run-label">Run via CLI</span>
      <div class="run">
        <button class="copy-btn" data-cmd="${esc(runCmd)}">copy</button>
        <code>${esc(runCmd)}</code>
      </div>
    </div>
  </article>`;
}

// ---- 5. COPY BUTTONS ----
function wireCopyButton(btn) {
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(btn.dataset.cmd);
      const old = btn.textContent;
      btn.textContent = "copied";
      setTimeout(() => (btn.textContent = old), 1200);
    } catch {
      btn.textContent = "press ⌘C";
    }
  });
}

function wireCopyButtons() {
  els.grid.querySelectorAll(".copy-btn").forEach(wireCopyButton);
}

function showState(msg) {
  els.grid.innerHTML = "";
  els.state.hidden = false;
  els.state.textContent = msg;
  els.count.textContent = "";
}

// ---- search input (debounced lightly via input event) ----
els.search.addEventListener("input", (e) => { query = e.target.value; render(); });

// ---- static "run locally" panel copy button (outside the grid, wired once) ----
const launchCopyBtn = document.getElementById("launch-copy");
if (launchCopyBtn) wireCopyButton(launchCopyBtn);

load();