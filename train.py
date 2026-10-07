"""Training script for MLRefineFPN on PASCAL VOC.

Usage:
    python train.py --config configs/voc.yaml --data-root ./data/VOCdevkit
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlrefinet_fpn import MLRefineFPNDetector


class VOCDataset(torch.utils.data.Dataset):
    """Minimal PASCAL VOC loader — replace with torchvision or your own loader."""

    def __init__(self, root, transforms=None):
        self.root = Path(root)
        self.image_ids = sorted(p.stem for p in (self.root / "JPEGImages").glob("*.jpg")) if (self.root / "JPEGImages").exists() else []
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        name = self.image_ids[idx]
        img_path = self.root / "JPEGImages" / f"{name}.jpg"
        import cv2
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        if self.transforms:
            image = self.transforms(image)
        return image, torch.zeros(0)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MLRefineFPN")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--data-root", default="./data/VOCdevkit")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="weights/mlrefinet.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    model = MLRefineFPNDetector(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_refinements=cfg["model"]["num_refinements"],
    ).to(device)

    dataset = VOCDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size or cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    optimizer = optim.SGD(model.parameters(), lr=args.lr or cfg["train"]["lr"],
                          momentum=cfg["train"]["momentum"], weight_decay=cfg["train"]["weight_decay"])
    epochs = args.epochs or cfg["train"]["epochs"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            cls_scores, bboxes = model(images)
            loss = sum(c.mean() for c in cls_scores) + sum(b.mean() for b in bboxes)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss={total/max(1,len(loader)):.4f}")
        torch.save({"model": model.state_dict()}, args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()