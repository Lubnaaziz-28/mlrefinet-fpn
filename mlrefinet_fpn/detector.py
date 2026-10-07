"""End-to-end MLRefineFPN detector."""

import torch
from torch import nn

from .anchors import AnchorGenerator, clip_boxes, decode_boxes, nms
from .backbone import Backbone
from .detection_head import DetectionHead
from .mlrefine_fpn import MLRefineFPN


class MLRefineFPNDetector(nn.Module):
    """Full detector: Backbone -> MLRefineFPN -> DetectionHead."""

    def __init__(self, num_classes=20, in_channels=256, out_channels=256, num_refinements=2):
        super().__init__()
        self.backbone = Backbone()
        self.neck = MLRefineFPN(in_channels, out_channels, num_refinements)
        self.head = DetectionHead(out_channels, num_classes)
        self.anchor_gen = AnchorGenerator()
        self.num_classes = num_classes

    def forward(self, images):
        feats = self.backbone(images)
        pyramid = self.neck(feats)
        cls_scores, bboxes = self.head(pyramid)
        return cls_scores, bboxes

    @torch.no_grad()
    def detect(self, images, score_threshold=0.5, iou_threshold=0.5, max_dets=100):
        """Run detection and return post-processed boxes/scores/labels per image.

        Args:
            images: (B, 3, H, W) tensor
        Returns:
            list of (boxes, scores, labels) tuples, one per image.
        """
        was_training = self.training
        self.eval()
        cls_scores, bboxes = self(images)
        self.train(was_training)
        device = images.device
        img_h, img_w = images.shape[-2:]
        results = []
        for bi in range(images.shape[0]):
            per_level = []
            for cls, bbox in zip(cls_scores, bboxes):
                h, w = cls.shape[2], cls.shape[3]
                s = img_h // h
                anchors = self.anchor_gen.generate(h, w, s, cls.dtype, device)
                cls_map = cls[bi].permute(1, 2, 0).reshape(-1, self.num_classes)
                bbox_map = bbox[bi].permute(1, 2, 0).reshape(-1, 4)
                scores = cls_map.max(dim=-1).values
                labels = cls_map.argmax(dim=-1)
                boxes = decode_boxes(anchors, bbox_map, stride=s)
                boxes = clip_boxes(boxes, img_h, img_w)
                keep = scores >= score_threshold
                per_level.append((boxes[keep], scores[keep], labels[keep]))
            all_boxes = torch.cat([p[0] for p in per_level], dim=0)
            all_scores = torch.cat([p[1] for p in per_level], dim=0)
            all_labels = torch.cat([p[2] for p in per_level], dim=0)
            order = torch.argsort(all_scores, descending=True)[:max_dets]
            all_boxes = all_boxes[order]
            all_scores = all_scores[order]
            all_labels = all_labels[order]
            keep_nms = nms(all_boxes, all_scores, iou_threshold)
            results.append((all_boxes[keep_nms], all_scores[keep_nms], all_labels[keep_nms]))
        return results