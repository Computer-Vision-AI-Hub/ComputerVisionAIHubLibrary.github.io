#!/usr/bin/env python3
"""
ComputerVisionAIHub local runner — a small web UI for running any catalog model.

Paste a model URL from the catalog (or set DEFAULT_MODEL), drop an image,
and see annotated detections. Runs fully on the user's machine.

Launched automatically by the Docker image on port 7860.
"""
import hashlib
import os
import urllib.request
import gradio as gr
from ultralytics import YOLO

# Optionally bake a default model into the image via build arg / env var.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")

_cache = {}


def get_model(model_ref: str):
    """Load (and cache) a model from a local path or URL."""
    if not model_ref:
        raise gr.Error("Provide a model URL or path.")
    if model_ref in _cache:
        return _cache[model_ref]

    path = model_ref
    if model_ref.startswith("http"):
        # Every catalog repo serves its weights as "best.pt", so basename alone
        # collides across models — hash the full URL to keep each download distinct.
        clean_url = model_ref.split("?")[0]
        ext = os.path.splitext(clean_url)[1] or ".pt"
        digest = hashlib.sha256(model_ref.encode()).hexdigest()[:16]
        path = os.path.join("/tmp", f"{digest}{ext}")
        if not os.path.exists(path):
            urllib.request.urlretrieve(model_ref, path)

    model = YOLO(path)
    _cache[model_ref] = model
    return model


def run(model_ref, image, conf):
    model = get_model(model_ref)
    results = model(image, conf=conf)
    annotated = results[0].plot()[..., ::-1]   # BGR -> RGB

    names = model.names

    # Works for both detection and OBB models
    detections = results[0].obb if results[0].obb is not None else results[0].boxes

    rows = [
        [names[int(det.cls)], round(float(det.conf), 3)]
        for det in detections
    ]

    return annotated, rows


with gr.Blocks(title="ComputerVisionAIHub runner") as demo:
    gr.Markdown("# ComputerVisionAIHub · local model runner\nPaste a model URL from the catalog, drop an image, and run it locally.")
    with gr.Row():
        with gr.Column():
            model_ref = gr.Textbox(
                label="Model URL or path",
                value=DEFAULT_MODEL,
                placeholder="https://huggingface.co/youruser/model/resolve/main/best.pt",
            )
            image = gr.Image(label="Image", type="numpy")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
            btn = gr.Button("Run detection", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Result")
            out_tbl = gr.Dataframe(headers=["class", "confidence"], label="Detections")

    btn.click(run, inputs=[model_ref, image, conf], outputs=[out_img, out_tbl])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)