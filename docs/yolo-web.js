/* ============================================================
   ComputerVisionAIHub — YOLOWeb
   Zero-install, client-side inference for the "Try in browser"
   feature. Loads an .onnx model with onnxruntime-web (wasm) and
   runs the full Ultralytics YOLOv8/v11/YOLO26 pipeline (letterbox
   preprocess -> session.run -> decode -> NMS) entirely in the tab.
   No server, no build step.

   Supports plain detection, OBB (rotated boxes), and segmentation
   (instance masks) — the export shape on session output0 tells
   decode() which one it's looking at (see decode() below).

   Pin note: the onnxruntime-web version below MUST match the CDN
   <script> tag loaded in index.html — mismatched JS/wasm builds
   fail to initialize.
   ============================================================ */

(function () {
  const ORT_VERSION = "1.27.0";

  if (typeof ort !== "undefined") {
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // ---- letterbox preprocess: fit into imgSize x imgSize, pad with 114,114,114 ----
  function letterbox(imgEl, imgSize) {
    const srcW = imgEl.naturalWidth || imgEl.width;
    const srcH = imgEl.naturalHeight || imgEl.height;
    const r = Math.min(imgSize / srcW, imgSize / srcH);
    const newW = Math.round(srcW * r);
    const newH = Math.round(srcH * r);
    const dw = (imgSize - newW) / 2;
    const dh = (imgSize - newH) / 2;

    const canvas = document.createElement("canvas");
    canvas.width = imgSize;
    canvas.height = imgSize;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, imgSize, imgSize);
    ctx.drawImage(imgEl, 0, 0, srcW, srcH, dw, dh, newW, newH);

    const { data } = ctx.getImageData(0, 0, imgSize, imgSize);
    const chw = new Float32Array(3 * imgSize * imgSize);
    const plane = imgSize * imgSize;
    for (let i = 0; i < plane; i++) {
      const o = i * 4;
      chw[i] = data[o] / 255; // R
      chw[plane + i] = data[o + 1] / 255; // G
      chw[2 * plane + i] = data[o + 2] / 255; // B
    }

    return {
      tensor: new ort.Tensor("float32", chw, [1, 3, imgSize, imgSize]),
      r,
      dw,
      dh,
    };
  }

  // undo letterbox -> original-image pixel space
  function unletterboxPoint(x, y, r, dw, dh) {
    return [(x - dw) / r, (y - dh) / r];
  }

  // ---- IoU between two axis-aligned [x1,y1,x2,y2] boxes ----
  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    const union = areaA + areaB - inter;
    return union <= 0 ? 0 : inter / union;
  }

  // ---- class-aware NMS (axis-aligned only — OBB boxes skip this, see detect()) ----
  function nms(boxes, iouThreshold) {
    const sorted = boxes.slice().sort((a, b) => b.score - a.score);
    const keep = [];
    for (const box of sorted) {
      let suppressed = false;
      for (const k of keep) {
        if (k.cls === box.cls && iou(k, box) > iouThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) keep.push(box);
    }
    return keep;
  }

  // ==== decoders — one per Ultralytics ONNX export shape ====
  // Every export in this catalog is end-to-end (NMS baked into the graph) with
  // boxes in input-pixel/imgSize letterboxed space. Which decoder applies is
  // determined by output shape in detect(), not by the model's declared task,
  // so a contributor's raw (non-end2end) export still works via decodeRawHead.

  // [1, numDets, 6] = x1,y1,x2,y2,score,cls
  function decodeDetection(data, dims, classes, r, dw, dh, confThreshold) {
    const numDets = dims[1];
    const boxes = [];
    for (let i = 0; i < numDets; i++) {
      const base = i * 6;
      const score = data[base + 4];
      if (score < confThreshold) continue;
      const cls = Math.round(data[base + 5]);
      const [x1, y1] = unletterboxPoint(data[base], data[base + 1], r, dw, dh);
      const [x2, y2] = unletterboxPoint(data[base + 2], data[base + 3], r, dw, dh);
      boxes.push({ x1, y1, x2, y2, score, cls, label: (classes && classes[cls]) || `class ${cls}` });
    }
    return boxes;
  }

  // raw detection head (no NMS baked in): [1, 4+numClasses, numAnchors], plane-major
  // i.e. value(attr, i) = data[attr*numAnchors + i]
  function decodeRawHead(data, dims, classes, r, dw, dh, confThreshold) {
    const numClasses = classes && classes.length ? classes.length : dims[1] - 4;
    const numAnchors = dims[2];
    const value = (attr, i) => data[attr * numAnchors + i];
    const boxes = [];
    for (let i = 0; i < numAnchors; i++) {
      let bestScore = -Infinity;
      let bestCls = -1;
      for (let c = 0; c < numClasses; c++) {
        const s = value(4 + c, i);
        if (s > bestScore) {
          bestScore = s;
          bestCls = c;
        }
      }
      if (bestScore < confThreshold) continue;
      const cx = value(0, i), cy = value(1, i), w = value(2, i), h = value(3, i);
      const [x1, y1] = unletterboxPoint(cx - w / 2, cy - h / 2, r, dw, dh);
      const [x2, y2] = unletterboxPoint(cx + w / 2, cy + h / 2, r, dw, dh);
      boxes.push({
        x1, y1, x2, y2,
        score: bestScore,
        cls: bestCls,
        label: (classes && classes[bestCls]) || `class ${bestCls}`,
      });
    }
    return boxes;
  }

  // OBB end-to-end: [1, numDets, 7] = cx,cy,w,h,score,cls,angle (radians).
  // Corner math matches Ultralytics' xywhr2xyxyxyxy. Rotation is preserved as-is
  // through the letterbox undo since letterbox is a pure uniform scale + translate.
  function decodeObb(data, dims, classes, r, dw, dh, confThreshold) {
    const numDets = dims[1];
    const boxes = [];
    for (let i = 0; i < numDets; i++) {
      const base = i * 7;
      const score = data[base + 4];
      if (score < confThreshold) continue;
      const cls = Math.round(data[base + 5]);
      const cx = data[base], cy = data[base + 1], w = data[base + 2], h = data[base + 3], angle = data[base + 6];

      const cos = Math.cos(angle), sin = Math.sin(angle);
      const v1x = (w / 2) * cos, v1y = (w / 2) * sin;
      const v2x = -(h / 2) * sin, v2y = (h / 2) * cos;
      const rawCorners = [
        [cx + v1x + v2x, cy + v1y + v2y],
        [cx + v1x - v2x, cy + v1y - v2y],
        [cx - v1x - v2x, cy - v1y - v2y],
        [cx - v1x + v2x, cy - v1y + v2y],
      ];
      const corners = rawCorners.map(([x, y]) => unletterboxPoint(x, y, r, dw, dh));
      const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);

      boxes.push({
        corners,
        x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys),
        score, cls,
        label: (classes && classes[cls]) || `class ${cls}`,
      });
    }
    return boxes;
  }

  // Segmentation end-to-end detections: [1, numDets, 6+numCoeffs] =
  // x1,y1,x2,y2,score,cls,coeff0..N (paired with a separate proto tensor).
  // Cheap pass only — box + score + cls + raw coeffs. The actual mask
  // (coeffs . protos, one 32x160x160 dot product per box) is deferred until
  // after NMS so suppressed candidates never pay for it.
  function decodeSegmentation(data, dims, classes, r, dw, dh, confThreshold) {
    const numDets = dims[1];
    const rowLen = dims[2];
    const numCoeffs = rowLen - 6;
    const boxes = [];
    for (let i = 0; i < numDets; i++) {
      const base = i * rowLen;
      const score = data[base + 4];
      if (score < confThreshold) continue;
      const cls = Math.round(data[base + 5]);
      const [x1, y1] = unletterboxPoint(data[base], data[base + 1], r, dw, dh);
      const [x2, y2] = unletterboxPoint(data[base + 2], data[base + 3], r, dw, dh);
      boxes.push({
        x1, y1, x2, y2,
        score, cls,
        label: (classes && classes[cls]) || `class ${cls}`,
        _coeffs: data.slice(base + 6, base + 6 + numCoeffs),
        _rawX1: data[base], _rawY1: data[base + 1], _rawX2: data[base + 2], _rawY2: data[base + 3],
      });
    }
    return boxes;
  }

  // mask = sigmoid(coeffs . protos), evaluated only inside the box's own proto-space
  // region (mirrors Ultralytics' crop_mask) so it never bleeds into other objects.
  function computeMaskGrid(coeffs, protoData, protoH, protoW, imgSize, rawX1, rawY1, rawX2, rawY2) {
    const planeSize = protoH * protoW;
    const downscale = imgSize / protoW;
    const px1 = Math.max(0, Math.floor(rawX1 / downscale));
    const py1 = Math.max(0, Math.floor(rawY1 / downscale));
    const px2 = Math.min(protoW, Math.ceil(rawX2 / downscale));
    const py2 = Math.min(protoH, Math.ceil(rawY2 / downscale));

    const grid = new Uint8Array(planeSize);
    for (let y = py1; y < py2; y++) {
      for (let x = px1; x < px2; x++) {
        let sum = 0;
        const pIdx = y * protoW + x;
        for (let k = 0; k < coeffs.length; k++) sum += coeffs[k] * protoData[k * planeSize + pIdx];
        grid[pIdx] = sigmoid(sum) > 0.5 ? 1 : 0;
      }
    }
    return grid;
  }

  function hexToRgb(hex) {
    const v = parseInt(String(hex).replace("#", ""), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }

  // Renders a mask grid (proto-resolution) into a canvas covering the FULL
  // original image extent (not just the box) — same letterbox-undo crop the
  // boxes themselves go through — so callers can just stretch-draw it over
  // the display canvas with a single drawImage, no extra geometry needed.
  function buildMaskCanvas(grid, protoH, protoW, imgSize, r, dw, dh, colorHex) {
    const [cr, cg, cb] = hexToRgb(colorHex);
    const protoCanvas = document.createElement("canvas");
    protoCanvas.width = protoW;
    protoCanvas.height = protoH;
    const pctx = protoCanvas.getContext("2d");
    const imgData = pctx.createImageData(protoW, protoH);
    for (let i = 0; i < grid.length; i++) {
      if (!grid[i]) continue;
      const o = i * 4;
      imgData.data[o] = cr;
      imgData.data[o + 1] = cg;
      imgData.data[o + 2] = cb;
      imgData.data[o + 3] = 150;
    }
    pctx.putImageData(imgData, 0, 0);

    const midCanvas = document.createElement("canvas");
    midCanvas.width = imgSize;
    midCanvas.height = imgSize;
    const mctx = midCanvas.getContext("2d");
    mctx.drawImage(protoCanvas, 0, 0, protoW, protoH, 0, 0, imgSize, imgSize);

    const newW = Math.max(1, Math.round(imgSize - 2 * dw));
    const newH = Math.max(1, Math.round(imgSize - 2 * dh));
    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = newW;
    finalCanvas.height = newH;
    finalCanvas.getContext("2d").drawImage(midCanvas, dw, dh, newW, newH, 0, 0, newW, newH);

    return finalCanvas;
  }

  async function loadModel(url) {
    return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  }

  async function detect(session, imgEl, opts) {
    const {
      classes = [],
      imgSize = 640,
      confThreshold = 0.25,
      iouThreshold = 0.45,
      colors = null,
    } = opts || {};

    const { tensor, r, dw, dh } = letterbox(imgEl, imgSize);
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    const results = await session.run(feeds);

    const outNames = session.outputNames;
    const output0 = results[outNames[0]];
    const dims0 = output0.dims;
    const lastDim = dims0[dims0.length - 1];
    const isSegmentation = outNames.length > 1;

    let boxes;
    let applyNms = true;

    if (isSegmentation) {
      boxes = decodeSegmentation(output0.data, dims0, classes, r, dw, dh, confThreshold);
    } else if (lastDim === 7) {
      boxes = decodeObb(output0.data, dims0, classes, r, dw, dh, confThreshold);
      applyNms = false; // rotated boxes: axis-aligned NMS would be wrong here, and the export already NMS'd
    } else if (lastDim === 6) {
      boxes = decodeDetection(output0.data, dims0, classes, r, dw, dh, confThreshold);
    } else {
      boxes = decodeRawHead(output0.data, dims0, classes, r, dw, dh, confThreshold);
    }

    let kept = applyNms ? nms(boxes, iouThreshold) : boxes;

    if (isSegmentation) {
      const proto = results[outNames[1]];
      const protoH = proto.dims[2], protoW = proto.dims[3];
      kept = kept.map((b) => {
        const grid = computeMaskGrid(b._coeffs, proto.data, protoH, protoW, imgSize, b._rawX1, b._rawY1, b._rawX2, b._rawY2);
        const color = (colors && colors[b.cls % colors.length]) || "#185FA5";
        const mask = buildMaskCanvas(grid, protoH, protoW, imgSize, r, dw, dh, color);
        const { _coeffs, _rawX1, _rawY1, _rawX2, _rawY2, ...rest } = b;
        return { ...rest, mask };
      });
    }

    return kept;
  }

  window.YOLOWeb = { loadModel, detect };
})();
