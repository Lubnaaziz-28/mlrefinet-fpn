"""Generate a 15-second demo GIF for the README.

Creates a synthetic scene, runs MLRefineFPN detection over 15 seconds of
frames, and renders an annotated GIF with ffmpeg.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlrefinet_fpn import MLRefineFPNDetector

CLASSES = [
    "background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def draw_boxes(image, boxes, scores, labels, class_names, color=(0, 255, 0)):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        name = class_names[label] if label < len(class_names) else str(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{name} {score:.2f}", (x1, max(0, y1 - 8)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def make_frame(w=320, h=240, t=0.0):
    """Create a synthetic scene with moving shapes to simulate a detection scene."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # background gradient
    for y in range(h):
        img[y, :, 0] = int(20 + 30 * y / h)
        img[y, :, 1] = int(20 + 20 * y / h)
        img[y, :, 2] = int(40 + 30 * y / h)
    # moving "objects"
    cx = int(w / 2 + 80 * np.sin(t))
    cy = int(h / 2 + 40 * np.cos(t * 1.3))
    cv2.rectangle(img, (cx - 20, cy - 15), (cx + 20, cy + 15), (60, 120, 255), -1)
    cv2.circle(img, (int(w / 3 + 50 * np.cos(t * 0.7)), int(h / 3)), 15, (80, 255, 120), -1)
    cv2.rectangle(img, (int(w / 4), int(3 * h / 4)), (int(w / 4) + 30, int(3 * h / 4) + 30), (255, 200, 80), -1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate demo GIF")
    parser.add_argument("--output", default="demo.gif")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-detect", action="store_true", help="skip detection, render synthetic scene only")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = None
    if not args.no_detect:
        model = MLRefineFPNDetector(num_classes=20).to(device).eval()

    tmp_dir = Path("/tmp/gif_frames")
    tmp_dir.mkdir(exist_ok=True)
    total_frames = args.seconds * args.fps
    for i in range(total_frames):
        t = i / args.fps
        frame = make_frame(t=t)
        if model is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            results = model.detect(tensor, score_threshold=0.01)[0]
            boxes, scores, labels = results
            annotated = draw_boxes(frame, boxes.cpu(), scores.cpu(), labels.cpu(), CLASSES)
        else:
            annotated = frame
        cv2.imwrite(str(tmp_dir / f"frame_{i:04d}.png"), annotated)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Rendered frame {i+1}/{total_frames}", flush=True)

    out_path = Path(args.output)
    palette_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(args.fps),
        "-i", str(tmp_dir / "frame_%04d.png"),
        "-vf", "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(out_path),
    ]
    import subprocess
    subprocess.run(palette_cmd, check=True)
    print(f"Saved demo GIF to {out_path}")


if __name__ == "__main__":
    main()