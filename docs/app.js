/* ============================================================
   ComputerVisionAIHub — catalog logic (vanilla JS, no build step)
   Flow:  fetch models.json -> render cards -> filter on input
   The page never hardcodes models; it all comes from the manifest.
   ============================================================ */

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

  // Try-in-browser modal (shared across all cards, filled per model on open)
  tryModal:       document.getElementById("try-modal"),
  tryModalTitle:  document.getElementById("try-modal-title"),
  tryModalClose:  document.getElementById("try-modal-close"),
  tryUploadLabel: document.getElementById("try-upload-label"),
  tryDrop:        document.getElementById("try-drop"),
  tryDropEmpty:   document.getElementById("try-drop-empty"),
  tryDropActions: document.getElementById("try-drop-actions"),
  tryUploadBtn:   document.getElementById("try-upload-btn"),
  tryPasteBtn:    document.getElementById("try-paste-btn"),
  tryFile:        document.getElementById("try-file"),
  tryConf:        document.getElementById("try-conf"),
  tryConfVal:     document.getElementById("try-conf-val"),
  tryShowLabels:  document.getElementById("try-show-labels"),
  tryStatus:      document.getElementById("try-status"),
  tryCanvas:      document.getElementById("try-canvas"),
  tryDetList:     document.getElementById("try-detections-list"),
};

// ---- Try-in-browser state ----
// Sessions are keyed by onnx URL and loaded lazily on first run, never on page load.
const sessionCache = new Map();
const tryState = { model: null, imgEl: null, scale: 1, lastDetections: [] };
const BOX_COLORS = ["#185FA5", "#0C8567", "#378ADD", "#5DCAA5", "#C4432B", "#B98A1E"];
const colorForClass = (cls) => BOX_COLORS[cls % BOX_COLORS.length];
// yolo-web.js decodes plain detection, obb (rotated boxes), and segmentation (masks).
const SUPPORTED_TRY_TASKS = new Set(["detection", "obb", "segmentation"]);

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
  wireTryButtons();
}

// ---- 4. ONE CARD (template per model) ----
function card(m) {
  const classes = m.classes || [];
  const canTryInBrowser = Boolean(m.onnx) && SUPPORTED_TRY_TASKS.has(m.task || "detection");
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

    ${canTryInBrowser ? `<div class="actions">
      <button type="button" class="btn try-btn" data-model-id="${esc(m.id)}">Try in browser ▶</button>
    </div>` : ""}

    <div class="run-group">
      <span class="run-label">Model URL — paste into localhost:7860</span>
      <div class="run">
        <button class="copy-btn" data-cmd="${esc(m.download)}">copy</button>
        <code>${esc(m.download)}</code>
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

// ---- 6. TRY IN BROWSER (client-side inference via yolo-web.js) ----
function wireTryButtons() {
  els.grid.querySelectorAll(".try-btn").forEach((btn) => {
    btn.addEventListener("click", () => openTryModal(btn.dataset.modelId));
  });
}

function openTryModal(id) {
  const m = ALL_MODELS.find((x) => x.id === id);
  if (!m || !els.tryModal) return;

  tryState.model = m;
  tryState.imgEl = null;
  tryState.scale = 1;
  tryState.lastDetections = [];

  els.tryModalTitle.textContent = `Try in browser — ${m.name}`;
  els.tryFile.value = "";
  els.tryConf.value = "0.25";
  els.tryConfVal.textContent = "0.25";
  els.tryShowLabels.checked = true;
  els.tryDetList.innerHTML = "";
  setTryStatus("");
  clearTryCanvas();
  setUploadState(false);

  els.tryModal.hidden = false;
  document.body.classList.add("modal-open");
}

function closeTryModal() {
  els.tryModal.hidden = true;
  document.body.classList.remove("modal-open");
}

function setTryStatus(msg, isError) {
  els.tryStatus.textContent = msg;
  els.tryStatus.classList.toggle("error", !!isError);
}

function clearTryCanvas() {
  els.tryCanvas.width = 0;
  els.tryCanvas.height = 0;
}

// Before an image is set: big placeholder, click-anywhere-to-choose, with the
// upload/paste icons already available. After: the canvas (image + detections)
// fills the same box in its place, with an "Upload a new image" label above —
// mirrors the local Docker UI's image input.
function setUploadState(hasImage) {
  els.tryDrop.classList.toggle("has-image", hasImage);
  els.tryDropEmpty.hidden = hasImage;
  els.tryCanvas.hidden = !hasImage;
  els.tryDropActions.hidden = false;
  els.tryUploadLabel.hidden = !hasImage;
}

function loadImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setTryStatus("Please choose an image file.", true);
    return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    tryState.imgEl = img;
    setUploadState(true);
    drawBaseImage();
    runTry();
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    setTryStatus("Could not read that image file.", true);
  };
  img.src = url;
}

function drawBaseImage() {
  const img = tryState.imgEl;
  if (!img) return;
  const maxW = 640;
  const scale = Math.min(1, maxW / img.naturalWidth);
  tryState.scale = scale;

  els.tryCanvas.width = Math.round(img.naturalWidth * scale);
  els.tryCanvas.height = Math.round(img.naturalHeight * scale);
  const ctx = els.tryCanvas.getContext("2d");
  ctx.drawImage(img, 0, 0, els.tryCanvas.width, els.tryCanvas.height);
}

// Sessions are cached per onnx URL so switching models or re-running never re-downloads.
function getSession(onnxUrl) {
  if (!sessionCache.has(onnxUrl)) {
    sessionCache.set(
      onnxUrl,
      YOLOWeb.loadModel(onnxUrl).catch((err) => {
        sessionCache.delete(onnxUrl); // allow retry on the next run
        throw err;
      })
    );
  }
  return sessionCache.get(onnxUrl);
}

async function runTry() {
  const m = tryState.model;
  const img = tryState.imgEl;
  if (!m || !img) return;

  try {
    setTryStatus(sessionCache.has(m.onnx) ? "Running detection…" : "Loading model…");
    const session = await getSession(m.onnx);

    setTryStatus("Running detection…");
    const confThreshold = parseFloat(els.tryConf.value);
    const detections = await YOLOWeb.detect(session, img, {
      classes: m.classes || [],
      imgSize: m.image_size || 640,
      confThreshold,
      colors: BOX_COLORS,
    });

    tryState.lastDetections = detections;
    drawBaseImage();
    drawDetections(detections);
    fillDetectionsList(detections);
    setTryStatus(
      `${detections.length} detection${detections.length === 1 ? "" : "s"} at ${Math.round(confThreshold * 100)}% confidence.`
    );
  } catch (err) {
    setTryStatus(`Error: ${err && err.message ? err.message : err}`, true);
  }
}

// Toggling "show labels" only needs to re-paint the last result, not re-run inference.
function redrawDetections() {
  if (!tryState.imgEl) return;
  drawBaseImage();
  drawDetections(tryState.lastDetections);
}

function drawDetections(detections) {
  const ctx = els.tryCanvas.getContext("2d");
  const scale = tryState.scale || 1;
  const showLabels = els.tryShowLabels.checked;

  // segmentation masks go down first, so outlines/tags sit on top of them
  detections.forEach((d) => {
    if (d.mask) {
      ctx.drawImage(d.mask, 0, 0, d.mask.width, d.mask.height, 0, 0, els.tryCanvas.width, els.tryCanvas.height);
    }
  });

  ctx.lineWidth = 2;
  ctx.font = "600 12px 'JetBrains Mono', monospace";
  ctx.textBaseline = "top";

  detections.forEach((d) => {
    const color = colorForClass(d.cls);
    let tagX, tagY;

    if (d.corners) {
      // obb: rotated polygon instead of an axis-aligned rect
      const pts = d.corners.map(([x, y]) => [x * scale, y * scale]);
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.stroke();
      const top = pts.reduce((a, b) => (b[1] < a[1] ? b : a));
      tagX = top[0];
      tagY = Math.max(0, top[1] - 16);
    } else {
      const x1 = d.x1 * scale, y1 = d.y1 * scale, x2 = d.x2 * scale, y2 = d.y2 * scale;
      if (!d.mask) {
        // plain detection: the mask fill already reads as the shape for segmentation,
        // so only draw the box outline when there isn't one
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      tagX = x1;
      tagY = Math.max(0, y1 - 16);
    }

    if (!showLabels) return;

    const tag = `${d.label} ${Math.round(d.score * 100)}%`;
    const tagW = ctx.measureText(tag).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(tagX, tagY, tagW, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, tagX + 4, tagY + 2);
  });
}

function fillDetectionsList(detections) {
  if (!detections.length) {
    els.tryDetList.innerHTML = `<li class="try-det-empty">No detections above this confidence.</li>`;
    return;
  }
  els.tryDetList.innerHTML = detections
    .map(
      (d) => `<li>
        <span class="try-det-swatch" style="background:${colorForClass(d.cls)}"></span>
        <span class="try-det-label">${esc(d.label)}</span>
        <span class="try-det-score">${(d.score * 100).toFixed(1)}%</span>
      </li>`
    )
    .join("");
}

// ---- search input (debounced lightly via input event) ----
els.search.addEventListener("input", (e) => { query = e.target.value; render(); });

// ---- static copy buttons outside the grid (launch panel + install guide), wired once ----
const launchCopyBtn = document.getElementById("launch-copy");
if (launchCopyBtn) wireCopyButton(launchCopyBtn);
document.querySelectorAll("#install-guide .copy-btn").forEach(wireCopyButton);

// ---- try-in-browser modal chrome, wired once (the modal itself is shared/reused per model) ----
if (els.tryModal) {
  els.tryModalClose.addEventListener("click", closeTryModal);
  els.tryModal.addEventListener("click", (e) => {
    if (e.target === els.tryModal) closeTryModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.tryModal.hidden) closeTryModal();
  });

  document.addEventListener("paste", (e) => {
    if (els.tryModal.hidden) return;
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          e.preventDefault();
          loadImageFile(file);
        }
        break;
      }
    }
  });

  els.tryFile.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) loadImageFile(file);
  });

  // empty-state placeholder: click anywhere to choose a file
  els.tryDropEmpty.addEventListener("click", () => els.tryFile.click());

  // once an image is loaded, the same actions move into these two icon buttons
  els.tryUploadBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    els.tryFile.click();
  });
  els.tryPasteBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!navigator.clipboard || !navigator.clipboard.read) {
      setTryStatus("Clipboard access isn't supported in this browser — try Ctrl/Cmd+V instead.", true);
      return;
    }
    try {
      const items = await navigator.clipboard.read();
      for (const item of items) {
        const type = item.types.find((t) => t.startsWith("image/"));
        if (type) {
          loadImageFile(new File([await item.getType(type)], "clipboard-image", { type }));
          return;
        }
      }
      setTryStatus("No image found on the clipboard.", true);
    } catch (err) {
      setTryStatus("Couldn't read the clipboard — try Ctrl/Cmd+V instead.", true);
    }
  });

  els.tryDrop.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.tryDrop.classList.add("dragover");
  });
  els.tryDrop.addEventListener("dragleave", () => els.tryDrop.classList.remove("dragover"));
  els.tryDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    els.tryDrop.classList.remove("dragover");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) loadImageFile(file);
  });

  els.tryConf.addEventListener("input", () => {
    els.tryConfVal.textContent = parseFloat(els.tryConf.value).toFixed(2);
  });
  els.tryConf.addEventListener("change", () => {
    if (tryState.imgEl) runTry();
  });

  els.tryShowLabels.addEventListener("change", redrawDetections);
}

load();