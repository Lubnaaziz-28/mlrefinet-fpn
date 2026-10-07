"""Live webcam detection app for MLRefineFPN.

Run with:  python app.py --config configs/voc.yaml --checkpoint weights/mlrefinet.pth
Or simply: python app.py  (uses a randomly-initialized model for a live demo)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlrefinet_fpn import MLRefineFPNDetector

CLASSES = [
    "background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_detector(config_path, checkpoint):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLRefineFPNDetector(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_refinements=cfg["model"]["num_refinements"],
    )
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    return model, device


def draw_boxes(image, boxes, scores, labels, class_names):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        name = class_names[label] if label < len(class_names) else str(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{name} {score:.2f}", (x1, max(0, y1 - 8)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def detect_frame(model, device, image, score_threshold=0.5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    results = model.detect(tensor, score_threshold=score_threshold)[0]
    boxes, scores, labels = results
    annotated = draw_boxes(image, boxes.cpu(), scores.cpu(), labels.cpu(), CLASSES)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="MLRefineFPN live detection app")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", default="0.0.0.0")
    args = parser.parse_args()

    model, device = build_detector(args.config, args.checkpoint)
    print(f"MLRefineFPN app ready on {device}")

    def predict(image, score_threshold):
        if image is None:
            return None
        return detect_frame(model, device, image, score_threshold)

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Webcam(source="webcam", image_type="numpy", label="Webcam"),
            gr.Slider(0.0, 1.0, value=0.5, label="Score threshold"),
        ],
        outputs=gr.Image(label="Detections"),
        live=True,
    )
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()