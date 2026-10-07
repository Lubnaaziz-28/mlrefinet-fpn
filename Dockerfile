# MLRefineFPN — Production Dockerfile
#
# Build (CPU):
#   docker build -t mlrefinet-fpn:cpu .
#
# Build (GPU / CUDA):
#   docker build --buildarg BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu22.04 \
#                  -t mlrefinet-fpn:gpu .
#
# Run (CPU):
#   docker run --rm -p 7860:7860 mlrefinet-fpn:cpu
#
# Run (GPU):
#   docker run --rm --gpus all -p 7860:7860 mlrefinet-fpn:gpu

ARG BASE_IMAGE=python:3.10-slim
FROM $BASE_IMAGE AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies required by OpenCV and Gradio
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gradio fastapi uvicorn

COPY . /app

RUN mkdir -p weights && \
    (wget -q -O weights/mlrefinet.pth https://github.com/Lubnaaziz-28/mlrefinet-fpn/releases/download/v1.0.0/mlrefinet.pth || true)

EXPOSE 7860

CMD ["python", "app.py", "--config", "configs/voc.yaml", "--checkpoint", "weights/mlrefinet.pth", "--server-name", "0.0.0.0", "--server-port", "7860"]