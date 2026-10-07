"""Inference for MLRefineFPN on a single image or webcam.

Usage:
    python inference.py --config configs/voc.yaml --checkpoint weights/mlrefinet.pth --image sample.jpg
    python inference.py --config configs/voc.yaml --checkpoint weights/mlrefinet.pth --webcam 0
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlrefinet_fpn import MLRefineFPNDetector


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_detector(cfg, device, checkpoint=None):
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
    return model


def draw_boxes(image, boxes, scores, labels, class_names):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        name = class_names[label] if label < len(class_names) else str(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{name} {score:.2f}", (x1, max(0, y1 - 8)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description="MLRefineFPN inference")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--checkpoint", default="weights/mlrefinet.pth")
    parser.add_argument("--image", default=None)
    parser.add_argument("--webcam", type=int, default=None)
    parser.add_argument("--output", default="output.jpg")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_detector(cfg, device, args.checkpoint)
    class_names = cfg.get("classes", [])

    if args.image:
        image = cv2.imread(args.image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        results = model.detect(tensor, score_threshold=args.score_threshold)[0]
        boxes, scores, labels = results
        print(f"Detected {len(boxes)} objects")
        annotated = draw_boxes(image, boxes.cpu(), scores.cpu(), labels.cpu(), class_names)
        cv2.imwrite(args.output, annotated)
        print(f"Saved to {args.output}")
    elif args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            results = model.detect(tensor, score_threshold=args.score_threshold)[0]
            boxes, scores, labels = results
            annotated = draw_boxes(frame, boxes.cpu(), scores.cpu(), labels.cpu(), class_names)
            cv2.imshow("MLRefineFPN", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Provide --image or --webcam")


if __name__ == "__main__":
    main()