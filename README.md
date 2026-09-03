# mlrefinet-fpn

Multi-Level Refinement Feature Pyramid for real-time object detection. Improves detection accuracy with no additional computational cost.

Built from my published paper: *"Multi-level refinement feature pyramid"* (Image and Vision Computing, 2021). Reported result: **+7% mAP over the baseline at no extra inference cost**.

## Why
Standard FPNs fuse multi-scale features but discard high-resolution detail during refinement. This method adds a multi-level refinement stage that re-injects refined features across pyramid levels: better accuracy, same FLOPs.

## Results
| Model | mAP | FLOPs | FPS |
|---|---|---|---|
| Baseline FPN | TBD (paper: ...) | ... | ... |
| + MLRefinet | **+7%** | baseline | real-time |

*Fill the table from the paper's experiments. Keep numbers identical to the publication.*

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py --config configs/refinet.yaml
python notebooks/demo.ipynb
```

## Citation
```bibtex
@article{aziz2021refinet,
  title={Multi-level refinement feature pyramid},
  author={Aziz, Lubna},
  journal={Image and Vision Computing},
  year={2021}
}
```
