"""Detection head producing per-level classification + box regression."""

from torch import nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=20, num_anchors=9, depth=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        cls_layers = []
        reg_layers = []
        for _ in range(depth):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
            cls_layers.append(nn.BatchNorm2d(in_channels))
            cls_layers.append(nn.ReLU(inplace=True))
            reg_layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
            reg_layers.append(nn.BatchNorm2d(in_channels))
            reg_layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*cls_layers)
        self.reg_head = nn.Sequential(*reg_layers)
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, 1, 1)

    def forward(self, feats):
        cls_scores = []
        bboxes = []
        for f in feats:
            cls_scores.append(self.cls_score(self.cls_head(f)))
            bboxes.append(self.bbox_pred(self.reg_head(f)))
        return cls_scores, bboxes