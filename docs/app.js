/* ============================================================
   ComputerVisionAIHub — catalog logic
   Flow:  fetch models.json -> render cards -> filter on input
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

  // Try-in-browser modal elements
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
  tryShowBoxes:   document.getElementById("try-show-boxes"),
  tryShowLabels:  document.getElementById("try-show-labels"),
  tryStatus:      document.getElementById("try-status"),
  tryCanvas:      document.getElementById("try-canvas"),
  tryDetList:     document.getElementById("try-detections-list"),

  tryProgressWrap: document.getElementById("try-progress-wrap"),
  tryProgressText: document.getElementById("try-progress-text"),
  tryProgressPct:  document.getElementById("try-progress-pct"),
  tryProgressFill: document.getElementById("try-progress-fill"),
};

// Try-in-browser state 
// Sessions are keyed by onnx URL and loaded  on first run.
const sessionCache = new Map();
const tryState = { model: null, imgEl: null, scale: 1, lastDetections: [], triggerEl: null };
const BOX_COLORS = ["#185FA5", "#0C8567", "#378ADD", "#5DCAA5", "#C4432B", "#B98A1E"];
const colorForClass = (cls) => BOX_COLORS[cls % BOX_COLORS.length];
// yolo-web.js decodes detection, obb (rotated boxes), segmentation (masks),
// classification, and pose.
const SUPPORTED_TRY_TASKS = new Set(["detection", "obb", "segmentation", "classification", "pose"]);

// COCO-17 skeleton (0-indexed) for pose visualization — every pose detection
// shares class "person", so distinguishing color comes from detection index
// instead of colorForClass (see drawDetections).
const POSE_SKELETON = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
  [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];
const KPT_VISIBLE_THRESHOLD = 0.5;

function drawKeypoints(ctx, keypoints, scale, color) {
  if (!keypoints || !keypoints.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  POSE_SKELETON.forEach(([a, b]) => {
    const pa = keypoints[a], pb = keypoints[b];
    if (!pa || !pb || pa.score < KPT_VISIBLE_THRESHOLD || pb.score < KPT_VISIBLE_THRESHOLD) return;
    ctx.beginPath();
    ctx.moveTo(pa.x * scale, pa.y * scale);
    ctx.lineTo(pb.x * scale, pb.y * scale);
    ctx.stroke();
  });
  ctx.fillStyle = color;
  keypoints.forEach((kp) => {
    if (kp.score < KPT_VISIBLE_THRESHOLD) return;
    ctx.beginPath();
    ctx.arc(kp.x * scale, kp.y * scale, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

// small helper: escape text before inserting into HTML 
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// 1. LOAD 
function renderSkeletons(count = 6) {
  els.grid.innerHTML = Array.from({ length: count }, () => `
    <div class="model-card skeleton" aria-hidden="true">
      <div class="skeleton-block skeleton-image"></div>
      <div class="skeleton-block skeleton-line" style="width:65%"></div>
      <div class="skeleton-block skeleton-line" style="width:40%"></div>
      <div class="skeleton-block skeleton-line" style="width:90%"></div>
      <div class="skeleton-block skeleton-line" style="width:80%"></div>
    </div>`).join("");
}

async function load() {
  renderSkeletons();
  try {
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

//  2. BUILD TASK FILTER PILLS from the data itself 
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

//  3. FILTER + RENDER 
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
  animateMetricBars();
}


function animateMetricBars() {
  const bars = els.grid.querySelectorAll(".bar > span");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bars.forEach((el) => { el.style.transform = `scaleX(${el.dataset.pct / 100})`; });
    });
  });
}

// 4. ONE CARD (template per model) 
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
      <div class="bar"><span data-pct="${pct}"></span></div>
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

    <span class="card-hint">Hover for specs, metrics &amp; download ↓</span>

    <div class="card-extra">
      <div class="card-extra-inner">
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
      </div>
    </div>
  </article>`;
}

// 5. COPY BUTTONS 
const COPY_ICON = `<svg class="copy-icon icon-copy" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
const CHECK_ICON = `<svg class="copy-icon icon-check" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>`;

function wireCopyButton(btn) {
  if (!btn.querySelector(".copy-label")) {
    const label = btn.textContent.trim() || "copy";
    btn.innerHTML = `${COPY_ICON}${CHECK_ICON}<span class="copy-label">${esc(label)}</span>`;
  }
  const label = btn.querySelector(".copy-label");
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(btn.dataset.cmd);
      const old = label.textContent;
      btn.classList.add("copied");
      label.textContent = "copied";
      setTimeout(() => {
        btn.classList.remove("copied");
        label.textContent = old;
      }, 1200);
    } catch {
      label.textContent = "press ⌘C";
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

// 6. TRY IN BROWSER (client-side inference via yolo-web.js)
function wireTryButtons() {
  els.grid.querySelectorAll(".try-btn").forEach((btn) => {
    btn.addEventListener("click", () => openTryModal(btn.dataset.modelId, btn));
  });
}

function openTryModal(id, triggerEl) {
  const m = ALL_MODELS.find((x) => x.id === id);
  if (!m || !els.tryModal) return;

  tryState.model = m;
  tryState.imgEl = null;
  tryState.scale = 1;
  tryState.lastDetections = [];
  tryState.triggerEl = triggerEl || document.activeElement;

  els.tryModalTitle.textContent = `Try in browser — ${m.name}`;
  els.tryFile.value = "";
  els.tryConf.value = "0.25";
  els.tryConfVal.textContent = "0.25";
  els.tryShowBoxes.checked = true;
  els.tryShowLabels.checked = true;
  els.tryDetList.innerHTML = "";
  setTryStatus("");
  clearTryCanvas();
  setUploadState(false);

  els.tryModal.hidden = false;
  document.body.classList.add("modal-open");
  els.tryModalClose.focus();

  prefetchModel(m);
}

function closeTryModal() {
  els.tryModal.hidden = true;
  document.body.classList.remove("modal-open");
  if (tryState.triggerEl && document.body.contains(tryState.triggerEl)) {
    tryState.triggerEl.focus();
  }
}

// Focusable, currently-visible controls inside the modal.
function getModalFocusable() {
  const selector = 'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])';
  return Array.from(els.tryModal.querySelectorAll(selector)).filter((el) => el.offsetParent !== null);
}

function setTryStatus(msg, isError) {
  els.tryStatus.textContent = msg;
  els.tryStatus.classList.toggle("error", !!isError);
}

function clearTryCanvas() {
  els.tryCanvas.width = 0;
  els.tryCanvas.height = 0;
}


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
const wasmOnlyModels = new Set();
const sessionCacheKey = (onnxUrl) => (wasmOnlyModels.has(onnxUrl) ? `${onnxUrl}#wasm` : onnxUrl);

// Download progress / fullyLoaded signs
const downloadProgress = new Map();
const fullyLoaded = new Set();

function reportProgress(onnxUrl, loaded, total) {
  downloadProgress.set(onnxUrl, { loaded, total });
  if (tryState.model && tryState.model.onnx === onnxUrl) {
    updateProgressUI(loaded, total);
  }
}

function getSession(onnxUrl) {
  const forceWasm = wasmOnlyModels.has(onnxUrl);
  const cacheKey = sessionCacheKey(onnxUrl);
  if (!sessionCache.has(cacheKey)) {
    sessionCache.set(
      cacheKey,
      YOLOWeb.loadModel(onnxUrl, {
        forceWasm,
        onProgress: (loaded, total) => reportProgress(onnxUrl, loaded, total),
      })
        .then((session) => {
          fullyLoaded.add(onnxUrl);
          return session;
        })
        .catch((err) => {
          sessionCache.delete(cacheKey); // allow retry on the next run
          throw err;
        })
    );
  }
  return sessionCache.get(cacheKey);
}

// Kicked off the instant "Try in browser" is clicked — the model starts
// downloading before the user has picked an image.
function prefetchModel(m) {
  if (!m.onnx) return;
  if (fullyLoaded.has(m.onnx)) {
    hideProgress();
    return;
  }
  const known = downloadProgress.get(m.onnx) || { loaded: 0, total: 0 };
  showProgress(known.loaded, known.total);
  getSession(m.onnx).then(hideProgress, hideProgress); // real errors surface when the user actually runs detection
}

function showProgress(loaded, total) {
  if (!els.tryProgressWrap) return;
  els.tryProgressWrap.hidden = false;
  updateProgressUI(loaded, total);
}

function hideProgress() {
  if (els.tryProgressWrap) els.tryProgressWrap.hidden = true;
}

function updateProgressUI(loaded, total) {
  if (!els.tryProgressWrap || els.tryProgressWrap.hidden) return;
  const mb = (n) => (n / (1024 * 1024)).toFixed(1);
  if (total > 0) {
    const pct = Math.min(100, Math.round((loaded / total) * 100));
    els.tryProgressFill.style.transform = `scaleX(${pct / 100})`;
    els.tryProgressPct.textContent = `${pct}%`;
    els.tryProgressText.textContent = `Downloading model… ${mb(loaded)} / ${mb(total)} MB`;
  } else if (loaded > 0) {
    // no Content-Length header to compute a percentage from — show bytes so far instead
    els.tryProgressFill.style.transform = "scaleX(1)";
    els.tryProgressPct.textContent = "";
    els.tryProgressText.textContent = `Downloading model… ${mb(loaded)} MB`;
  } else {
    els.tryProgressFill.style.transform = "scaleX(0)";
    els.tryProgressPct.textContent = "";
    els.tryProgressText.textContent = "Downloading model…";
  }
}

// The actual load+infer+draw attempt, isolated so runTry can retry it once on
// the CPU backend if the GPU one fails partway through (see runTry).
async function attemptRun(m, img, confThreshold) {
  if (!fullyLoaded.has(m.onnx)) {
    setTryStatus("Waiting for the model download to finish…");
  }
  const session = await getSession(m.onnx);
  hideProgress();

  setTryStatus("Running detection…");
  const detections = await YOLOWeb.detect(session, img, {
    classes: m.classes || [],
    imgSize: m.image_size || 640,
    confThreshold,
    colors: BOX_COLORS,
    task: m.task || "detection",
  });

  tryState.lastDetections = detections;
  drawBaseImage();
  drawDetections(detections);
  fillDetectionsList(detections);
  const unit = m.task === "classification" ? "prediction" : "detection";
  setTryStatus(
    `${detections.length} ${unit}${detections.length === 1 ? "" : "s"} at ${Math.round(confThreshold * 100)}% confidence.`
  );
}

async function runTry() {
  const m = tryState.model;
  const img = tryState.imgEl;
  if (!m || !img) return;

  const confThreshold = parseFloat(els.tryConf.value);
  try {
    await attemptRun(m, img, confThreshold);
  } catch (err) {
    if (!wasmOnlyModels.has(m.onnx)) {
      wasmOnlyModels.add(m.onnx);
      try {
        setTryStatus("That failed on the GPU backend — retrying on CPU…");
        await attemptRun(m, img, confThreshold);
        return;
      } catch (err2) {
        setTryStatus(`Error: ${err2 && err2.message ? err2.message : err2}`, true);
        return;
      }
    }
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
  if (!els.tryShowBoxes.checked) return; // boxes off -> just the plain image underneath
  // classification displays its results only live in the side list
  if (tryState.model && tryState.model.task === "classification") return;

  const ctx = els.tryCanvas.getContext("2d");
  const scale = tryState.scale || 1;
  const showLabels = els.tryShowLabels.checked;
  const isPose = tryState.model && tryState.model.task === "pose";

  // segmentation masks go down first, so outlines/tags sit on top of them
  detections.forEach((d) => {
    if (d.mask) {
      ctx.drawImage(d.mask, 0, 0, d.mask.width, d.mask.height, 0, 0, els.tryCanvas.width, els.tryCanvas.height);
    }
  });

  ctx.lineWidth = 2;
  ctx.font = "600 12px 'JetBrains Mono', monospace";
  ctx.textBaseline = "top";

  detections.forEach((d, idx) => {
    // pose detections are all class "person" — color by instance instead of class
    const color = isPose ? BOX_COLORS[idx % BOX_COLORS.length] : colorForClass(d.cls);
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
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      tagX = x1;
      tagY = Math.max(0, y1 - 16);
    }

    if (isPose) drawKeypoints(ctx, d.keypoints, scale, color);

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

// search input
els.search.addEventListener("input", (e) => { query = e.target.value; render(); });

// static copy buttons outside the grid (launch panel + install guide) 
const launchCopyBtn = document.getElementById("launch-copy");
if (launchCopyBtn) wireCopyButton(launchCopyBtn);
document.querySelectorAll("#install-guide .copy-btn").forEach(wireCopyButton);

// try-in-browser modal chrome, wired once (the modal itself is shared/reused per model) 
if (els.tryModal) {
  els.tryModalClose.addEventListener("click", closeTryModal);
  els.tryModal.addEventListener("click", (e) => {
    if (e.target === els.tryModal) closeTryModal();
  });
  document.addEventListener("keydown", (e) => {
    if (els.tryModal.hidden) return;
    if (e.key === "Escape") {
      closeTryModal();
      return;
    }
    // trap Tab inside the modal so background cards aren't reachable while it's open
    if (e.key === "Tab") {
      const focusable = getModalFocusable();
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
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
  els.tryDropEmpty.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.tryFile.click();
    }
  });

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

  els.tryShowBoxes.addEventListener("change", () => {
    if (!els.tryShowBoxes.checked) els.tryShowLabels.checked = false;
    redrawDetections();
  });
  els.tryShowLabels.addEventListener("change", () => {
    if (els.tryShowLabels.checked) els.tryShowBoxes.checked = true;
    redrawDetections();
  });
}

load();