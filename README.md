<div align="center">

# MRefineFPN

### Multi-Level Refinement Feature Pyramid Network

[![Paper](https://img.shields.io/badge/Paper-IVC_2021-0076D6?logo=readthedocs&logoColor=white)]()
[![mAP](https://img.shields.io/badge/mAP-+7%25-brightgreen?style=flat-square)]()
[![FLOPs](https://img.shields.io/badge/FLOPs-Zero_Extra_Cost-blue?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.8+-yellow?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch&logoColor=white)]()

*A published CV architecture that improves detection accuracy at zero additional compute cost.*

</div>

---

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
| Baseline FPN | TBD | TBD | TBD |
| **+ MLRefineFPN** | **+7%** | Same | Real-time |

> Numbers are from the published paper. Fill table with exact values from IVC 2021 experiments.

## Quickstart

```bash
pip install -r requirements.txt

# Inference
python inference.py --config configs/voc.yaml --checkpoint weights/mlrefinet.pth --image sample.jpg

# Training
python train.py --config configs/voc.yaml --data-root ./data/VOCdevkit
```

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
