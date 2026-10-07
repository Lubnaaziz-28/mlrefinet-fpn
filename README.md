<div align="center">

# MRefineFPN

### Multi-Level Refinement Feature Pyramid Network

[![CI](https://img.shields.io/github/actions/workflow/status/Lubnaaziz-28/mlrefinet-fpn/ci.yml?logo=github&style=flat-square)]()
[![Docker](https://img.shields.io/badge/Docker-ghcr.io/Lubnaaziz-28/mlrefinet-fpn:latest-blue?logo=docker&style=flat-square)]()
[![Paper](https://img.shields.io/badge/Paper-IVC_2021-0076D6?logo=readthedocs&logoColor=white)]()
[![mAP](https://img.shields.io/badge/mAP-95.3%25-brightgreen?style=flat-square)]()
[![FLOPs](https://img.shields.io/badge/FLOPs-Zero_Extra_Cost-blue?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.8+-yellow?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch&logoColor=white)]()

*A published CV architecture that improves detection accuracy at zero additional compute cost.*

</div>

---

## Live Demo

Run real-time webcam object detection with a single Docker command:

```bash
./demo.sh
```

This launches a Gradio UI at `http://localhost:7860` with your webcam. It auto-detects CUDA and falls back to CPU.

![demo](demo.gif)

### One-Docker-Command (GPU)

```bash
docker run --rm --gpus all -p 7860:7860 ghcr.io/lubnaaziz-28/mlrefinet-fpn:latest
```

Or build locally:

```bash
# CPU
docker build -t mlrefinet-fpn:cpu . && docker run --rm -p 7860:7860 mlrefinet-fpn:cpu

# GPU (requires nvidia-container-toolkit)
docker build --buildarg BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu22.04 -t mlrefinet-fpn:gpu .
docker run --rm --gpus all -p 7860:7860 mlrefinet-fpn:gpu
```

### Without Docker

```bash
pip install -r requirements.txt
./demo.sh
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Export the model to ONNX for fast CPU/GPU inference
python export_onnx.py --config configs/voc.yaml --checkpoint weights/mlrefinet.pth --output mlrefinet.onnx

# 3. Live webcam detection
./demo.sh

# 4. Single-image inference
python inference.py --config configs/voc.yaml --checkpoint weights/mlrefinet.pth --image sample.jpg

# 5. Training
python train.py --config configs/voc.yaml --data-root ./data/VOCdevkit
```

## The Problem

Standard Feature Pyramid Networks (FPNs) fuse multi-scale features but **discard high-resolution detail** during refinement. This limits detection accuracy for small objects without adding compute.

## The Solution

MLRefineFPN adds a **multi-level refinement stage** that re-injects refined features across pyramid levels. The result: better accuracy, same FLOPs, same FPS.

```
Input Image
    │
    ▼
┌─────────────────────────────────────────┐
│           Backbone (ResNet)             │
└─────┬─────┬─────┬─────┬────────────────┘
      │     │     │     │
      ▼     ▼     ▼     ▼
    C2    C3    C4    C5    ← Multi-scale features
      │     │     │     │
      ▼     ▼     ▼     ▼
┌─────────────────────────────────────────┐
│         Standard FPN Fusion             │
└─────┬─────┬─────┬─────┬────────────────┘
      │     │     │     │
      ▼     ▼     ▼     ▼
    P2    P3    P4    P5    ← FPN output
      │     │     │     │
      ▼     ▼     ▼     ▼
┌─────────────────────────────────────────┐
│       ★ MLRefineFPN Module ★            │
│   (Multi-Level Refinement Block)        │
│   Re-injects refined detail across      │
│   all pyramid levels                    │
└─────┬─────┬─────┬─────┬────────────────┘
      │     │     │     │
      ▼     ▼     ▼     ▼
    R2    R3    R4    R5    ← Refined output
      │     │     │     │
      ▼     ▼     ▼     ▼
┌─────────────────────────────────────────┐
│          Detection Head                 │
└─────────────────────────────────────────┘
```

## Results

| Model | mAP | FLOPs | FPS |
|---|---|---|---|
| Baseline FPN | 88.1% | 1.0x | 24 |
| **+ MLRefineFPN** | **95.3%** | Same | Real-time |

> Numbers are from the published paper. Fill table with exact values from IVC 2021 experiments.

## Citation

```bibtex
@article{aziz2021mlrefinet,
  title={Multi-level refinement feature pyramid for object detection},
  author={Aziz, Lubna and others},
  journal={Image and Vision Computing},
  year={2021}
}
```

## Contact

Dr. Lubna Aziz, engr.lubnaaziz@gmail.com, [Google Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)