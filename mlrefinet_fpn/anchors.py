"""Anchor generation, box decoding and non-maximum suppression."""

import torch


class AnchorGenerator:
    def __init__(self, base_size=32, scales=(2 ** 0, 2 ** 1, 2 ** 2), ratios=(0.5, 1.0, 2.0)):
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
        self.num_anchors = len(scales) * len(ratios)

    def _meshgrid(self, h, w, dtype, device):
        yy = torch.arange(h, dtype=dtype, device=device) + 0.5
        xx = torch.arange(w, dtype=dtype, device=device) + 0.5
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        return yy, xx

    def generate(self, feat_h, feat_w, stride, dtype, device):
        yy, xx = self._meshgrid(feat_h, feat_w, dtype, device)
        cx = (xx + 0.5) * stride
        cy = (yy + 0.5) * stride
        base_w = self.base_size
        base_h = self.base_size
        anchors = []
        for s in self.scales:
            for r in self.ratios:
                w = base_w * s / (r ** 0.5)
                h = base_h * s * (r ** 0.5)
                anchors.append(torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1))
        anchors = torch.stack(anchors, dim=-2).reshape(-1, 4)
        return anchors


def decode_boxes(anchors, deltas, stride=1.0):
    """Decode box deltas relative to anchors (cx, cy, w, h encoding)."""
    deltas = deltas.reshape(-1, 4)
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + widths * 0.5
    cy = anchors[:, 1] + heights * 0.5
    dx = deltas[:, 0] * stride
    dy = deltas[:, 1] * stride
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    pred_cx = dx * widths + cx
    pred_cy = dy * heights + cy
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    x1 = pred_cx - pred_w * 0.5
    y1 = pred_cy - pred_h * 0.5
    x2 = pred_cx + pred_w * 0.5
    y2 = pred_cy + pred_h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def nms(boxes, scores, iou_threshold=0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        idx = order[1:]
        xx1 = torch.maximum(x1[i], x1[idx])
        yy1 = torch.maximum(y1[i], y1[idx])
        xx2 = torch.minimum(x2[i], x2[idx])
        yy2 = torch.minimum(y2[i], y2[idx])
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[idx] - inter + 1e-10)
        order = idx[iou <= iou_threshold]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def clip_boxes(boxes, h, w):
    boxes[:, 0].clamp_(0, w)
    boxes[:, 1].clamp_(0, h)
    boxes[:, 2].clamp_(0, w)
    boxes[:, 3].clamp_(0, h)
    return boxes