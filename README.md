# ComputerVisionAIHub

An open catalog of custom-trained **YOLO** models. Browse models, download the weights from the HuggingFace Hub, and run them locally with one Docker command.

## How it works

```
Roboflow dataset ──► Google Colab (train YOLO model) ──► Hugging Face Hub (weights)
                                                            │
   GitHub repo (code) ──► GitHub Pages (catalog) ───────────┘
                                   │
                          user downloads + runs via Docker
```


## Repo layout

```
docs/         
  index.html  structure
  styles.css  design tokens + the bounding-box card styling
  app.js      fetches models.json and renders the catalog
  models.json THE CATALOG — add a model by appending
notebooks/
  train_yolo26.ipynb   end-to-end training + publishing in HuggingFace
scripts/
  publish_to_hf.py     push existing weights trained locally to HuggingFace from your terminal
inference/
  Dockerfile  detect.py  app.py  requirements.txt   local model runner
LICENSE       AGPL-3.0
```

## For users: run a model

*Prerequisites*:

- Install Docker Engine: [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)

- Install Docker GPU support: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)

Every catalog card has a copy-paste command. The model argument is the Hugging
Face URL — the container downloads it on first run.

```bash
# build once (make sure you build from computervisionaihub directory)
docker build -t computervisionaihub:1.0 ./inference

# CLI: annotate one image
docker run -v $(pwd):/data computervisionaihub:1.0 \
  https://huggingface.co/youruser/yourmodel/resolve/main/best.pt \
  /data/test.jpg

# web UI: drag-and-drop at http://localhost:7860
docker run -p 7860:7860 --entrypoint python computervisionaihub:1.0 app.py
```

## For maintainers: add a model

1. Open `notebooks/train_yolo26.ipynb` in Google Colab, set runtime to a **T4 GPU** (free resource).
2. Fill in your Roboflow + Hugging Face details and run all cells. It trains
   YOLO26, exports `.pt` + `.onnx`, pushes them to the Hub, and **prints a
   ready-to-paste `models.json` entry**.
3. Paste that entry into the `models` array in `docs/models.json`, edit the
   human-readable fields (name, summary, dataset link, license), and commit.

That's it — no website code changes. (Trained locally instead - Use
`python scripts/publish_to_hf.py --repo youruser/model-id --pt best.pt --onnx best.onnx`.)


## Local preview

The site fetches `models.json`, so open it through a server (not `file://`):

```bash
cd docs && python -m http.server
# visit http://localhost:8000
```

## A note on licensing

Ultralytics YOLO is **AGPL-3.0**, a strong copyleft license. Because the training
notebook and Docker image use the Ultralytics library, this repo is AGPL-3.0 too.
Two practical consequences:

- Anyone redistributing or network-serving Ultralytics-based code inherits AGPL
  obligations. That's fine for an open project, but worth telling your users.
- The **ONNX** export can be run with `onnxruntime` *without* Ultralytics, letting
  downstream users avoid AGPL in their own apps. That's why every model ships both
  formats.

Each model's underlying **dataset** has its own license (shown on each card).
Check it before any commercial use.