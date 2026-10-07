"""Decode raw detector outputs into human-readable predictions."""

import torch

from .anchors import clip_boxes, decode_boxes, nms


def decode_predictions(cls_scores, bboxes, num_classes, strides, img_h, img_w, anchor_gen, score_threshold=0.5, iou_threshold=0.5, max_dets=100):
    """Decode raw model outputs into per-image predictions."""
    results = []
    num_images = cls_scores[0].shape[0]
    for bi in range(num_images):
        per_level = []
        for cls, bbox in zip(cls_scores, bboxes):
            h, w = cls.shape[2], cls.shape[3]
            s = strides[0] if isinstance(strides, list) else strides
            anchors = anchor_gen.generate(h, w, s, cls.dtype, cls.device)
            cls_map = cls[bi].permute(1, 2, 0).reshape(-1, num_classes)
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
        all_boxes, all_scores, all_labels = all_boxes[order], all_scores[order], all_labels[order]
        keep_nms = nms(all_boxes, all_scores, iou_threshold)
        results.append((all_boxes[keep_nms], all_scores[keep_nms], all_labels[keep_nms]))
    return results