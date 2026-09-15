# ComputerVisionAIHub

An open catalog of custom-trained **YOLO** and **RT-DETR** models. Browse models, try them online, download the weights from the HuggingFace Hub, and run them locally with one Docker command.

## How training and publishing works (developer side)

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
  index.html  (structure)
  styles.css  (design tokens + the bounding-box card styling)
  app.js      (fetches models.json and renders the catalog)
  models.json (the catalog)
notebooks/
  train_yolo26.ipynb   (end-to-end training + publishing in HuggingFace)
scripts/
  publish_to_hf.py     (push existing weights trained locally to HuggingFace from your terminal)
inference/
  Dockerfile      (Dockerfile)
  detect.py       (inference testing script)
  app.py          (docker inference)
  requirements.txt   (requirements)
LICENSE       AGPL-3.0
```

## For users: run a model immediately

The fastest way to run a model is to "Try in Browser" our trained models.
"Try in Browser" fetches the model, and runs it on your local resources, so the model and images that you use never leaves your machine.


## For users: run a model locally

*Prerequisites*:

- Install Docker Engine: [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)

- Install Docker GPU support: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)

Every catalog card has a copy-paste command. The model argument is the Hugging
Face URL — the container downloads it on first run.

```bash
# CLI: annotate one image (pulls the image from Docker Hub on first run)
docker run -v $(pwd):/data docker.io/lukasiktar/computervisionaihub:latest \
  https://huggingface.co/youruser/yourmodel/resolve/main/best.pt \
  /data/test.jpg

# web UI: drag-and-drop at http://localhost:7860
docker run -p 7860:7860 --entrypoint python docker.io/lukasiktar/computervisionaihub:latest app.py
```

Add `--gpus all` to either command to use the GPU (requires the NVIDIA Container Toolkit — see prerequisites above).

Building from source instead of pulling from Docker Hub:

```bash
# build once (make sure you build from the computervisionaihub directory)
docker build -t computervisionaihub:local ./inference
# then swap docker.io/lukasiktar/computervisionaihub:latest for computervisionaihub:local in the commands above
```

## For maintainers: train a model

1. Open `notebooks/train_yolo26.ipynb` in Google Colab, set runtime to a **T4 GPU** (free resource).
2. Fill in your Roboflow + Hugging Face details and run all cells. It trains
   YOLO26, exports `.pt` + `.onnx`, pushes them to the Hub, and **prints a
   ready-to-paste `models.json` entry**.
3. Paste that entry into the `models` array in `docs/models.json`, edit the
   human-readable fields (name, summary, dataset link, license), and commit.

That's it — no website code changes. (Trained locally instead - Use
`python scripts/publish_to_hf.py --repo youruser/model-id --pt best.pt --onnx best.onnx`.)



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