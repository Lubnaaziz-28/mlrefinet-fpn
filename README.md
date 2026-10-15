# mlrefinet-fpn

Multi-Level Refinement Feature Pyramid Network for Object Detection.

## Features

- **Multi-Level Refinement**: Progressive feature refinement across FPN levels
- **Modern Architecture**: ResNet50 backbone with FPN + refinement blocks
- **Flexible Training**: COCO-format dataset support, configurable via YAML
- **Export Ready**: ONNX export for deployment
- **Colab Kit**: One-click training on Google Colab with Gradio demo

## Quick Start

### Installation

```bash
git clone https://github.com/lubna-github/mlrefinet-fpn.git
cd mlrefinet-fpn
pip install -r requirements.txt
pip install albumentations pycocotools
```

### Train on Custom Data

Prepare your data in COCO format:
```
data/
├── train.json
├── val.json
└── images/
    ├── img1.jpg
    └── ...
```

Run training:
```bash
python train_custom.py --data-path ./data --epochs 50 --batch-size 8
```

Or use the config file:
```bash
python train.py --config configs/voc.yaml --data ./data
```

### Export to ONNX

```bash
python export_onnx.py --checkpoint outputs/best_model.pth --output model.onnx
```

## 📓 Colab Kit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lubna-github/mlrefinet-fpn/blob/main/mlrefinet_fpn_colab.ipynb)

Train a custom object detector on your own data in Google Colab with zero setup.

### What the Colab Notebook Does

1. **Installs dependencies** automatically
2. **Accepts your COCO-format data** via upload or Google Drive mount
3. **Trains mlrefinet-fpn** with configurable hyperparameters
4. **Tracks metrics** with Weights & Biases (optional, graceful fallback)
5. **Exports to ONNX** automatically after training
6. **Launches Gradio demo** with public link for immediate testing

### Usage

1. Click the "Open in Colab" badge above
2. Run the setup cell
3. Upload your `train.json`, `val.json`, and `images.zip` (or mount Google Drive)
4. Configure training parameters (epochs, batch size, learning rate)
5. Enable Weights & Biases if desired (enter API key when prompted)
6. Run the training cell
7. Get your trained model (`.pth`), ONNX export (`.onnx`), and a live Gradio demo link

### Requirements for Your Data

- **COCO format** JSON annotations (`train.json`, `val.json`)
- **Images folder** with corresponding image files
- **Categories** defined in the JSON (auto-detected for class count)

Example COCO structure:
```json
{
  "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "my_object", "supercategory": "object"}]
}
```

## CLI Script: `train_custom.py`

Standalone training script with all options exposed as CLI arguments:

```bash
python train_custom.py \
  --data-path /path/to/coco/data \
  --output-dir ./my_outputs \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.001 \
  --backbone resnet50 \
  --fpn-channels 256 \
  --wandb \
  --wandb-project my-project \
  --export-onnx
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | *required* | Path to COCO dataset (train.json, val.json, images/) |
| `--output-dir` | `outputs` | Output directory for checkpoints |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--lr-scheduler` | `cosine` | LR scheduler (`cosine` or `step`) |
| `--backbone` | `resnet50` | Backbone architecture |
| `--fpn-channels` | `256` | FPN channel count |
| `--input-size` | `640 640` | Input resolution (H W) |
| `--wandb` | `False` | Enable Weights & Biases |
| `--export-onnx` | `False` | Export ONNX after training |

## GitHub Actions CI

The repository includes a CI workflow that:
1. Generates a synthetic COCO dataset
2. Runs training for 2 epochs
3. Exports to ONNX
4. Validates ONNX inference

Run locally:
```bash
python scripts/generate_synthetic_data.py --output data/synthetic --num-images 20 --num-classes 3
python train_custom.py --data-path data/synthetic --epochs 2 --batch-size 2 --no-pretrained --fpn-channels 128
```

## Project Structure

```
mlrefinet-fpn/
├── mlrefinet_fpn/
│   ├── __init__.py
│   ├── model.py          # MLRefineFPN architecture
│   ├── dataset.py        # COCO dataset loader
│   ├── loss.py           # Focal + IoU losses
│   └── export_onnx.py    # ONNX export utility
├── train.py              # Config-based training
├── train_custom.py       # CLI training script
├── export_onnx.py        # Standalone ONNX export
├── mlrefinet_fpn_colab.ipynb  # Colab notebook
├── configs/
│   └── voc.yaml          # Example config
├── scripts/
│   └── generate_synthetic_data.py
├── .github/workflows/
│   └── ci.yml            # CI pipeline
├── requirements.txt
└── README.md
```

## Model Architecture

```
Input (3, 640, 640)
    │
    ▼
ResNet50 Backbone (C2, C3, C4, C5)
    │
    ▼
FPN (P2, P3, P4, P5) ──► Multi-Level Refinement (2 blocks per level)
    │
    ├──► P6 (stride 2)
    │    └──► P7 (stride 2)
    │
    ▼
Detection Head (per level)
    ├──► Classification (num_classes × 9 anchors)
    └──► Regression (4 × 9 anchors)
```

## Citation

If you use this code, please cite:
```bibtex
@misc{mlrefinet-fpn,
  title={mlrefinet-fpn: Multi-Level Refinement FPN for Object Detection},
  author={mlrefinet-fpn team},
  year={2024}
}
```

## License

MIT License