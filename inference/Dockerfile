# ComputerVisionAIHub local inference image.
#
# Build:
#   docker build -t computervisionaihub ./inference
#
# Run the CLI (replace <MODEL_URL> and test.jpg with your own):
#   docker run -v $(pwd):/data computervisionaihub <MODEL_URL> /data/test.jpg
#
# Run the web UI instead (drag-and-drop on http://localhost:7860):
#   docker run -p 7860:7860 -v $(pwd)/inference:/app  --entrypoint python docker.io/lukasiktar/computervisionaihub app.py


FROM python:3.11-slim

# System libs that OpenCV (pulled in by ultralytics) needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY detect.py app.py ./app

# Default: CLI inference. Override --entrypoint for the web UI (see header).
ENTRYPOINT ["python", "detect.py"]