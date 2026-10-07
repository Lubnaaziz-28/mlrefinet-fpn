"""Export the MLRefineFPN detector to ONNX for fast CPU/GPU inference.

Usage:
    python export_onnx.py --config configs/voc.yaml --checkpoint weights/mlrefinet.pth \
        --output mlrefinet.onnx --opset 14
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlrefinet_fpn import MLRefineFPNDetector


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class ONNXWrapper(torch.nn.Module):
    """Wraps the detector so ONNX export sees two single-tensor outputs.

    Each pyramid level is zero-padded to the largest spatial size, then all
    levels are concatenated along the channel dimension. The consumer can
    slice the channel dimension back into per-level predictions.
    """

    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, images):
        cls_scores, bboxes = self.det(images)
        ref_h = max(c.shape[2] for c in cls_scores)
        ref_w = max(c.shape[3] for c in cls_scores)

        def pad(t):
            _, _, h, w = t.shape
            pad_h = ref_h - h
            pad_w = ref_w - w
            return F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0)

        cls = torch.cat([pad(c) for c in cls_scores], dim=1)
        bbox = torch.cat([pad(b) for b in bboxes], dim=1)
        return cls, bbox


def main():
    parser = argparse.ArgumentParser(description="Export MLRefineFPN to ONNX")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--checkpoint", default="weights/mlrefinet.pth")
    parser.add_argument("--output", default="mlrefinet.onnx")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", help="export with dynamic batch/height/width")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")
    model = MLRefineFPNDetector(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_refinements=cfg["model"]["num_refinements"],
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()

    wrapped = ONNXWrapper(model)

    dummy = torch.randn(args.batch, 3, 640, 640, device=device)
    dynamic = {"input": {0: "batch", 2: "height", 3: "width"}} if args.dynamic else None
    input_names = ["input"]
    output_names = ["cls_scores", "bboxes"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        args.output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        opset_version=args.opset,
    )
    print(f"Exported ONNX model to {args.output}")

    # Verify with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})
    print(f"ONNX verification OK — cls={out[0].shape} bbox={out[1].shape}")


if __name__ == "__main__":
    main()