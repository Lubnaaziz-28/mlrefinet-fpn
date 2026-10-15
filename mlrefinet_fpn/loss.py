import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class IoULoss(nn.Module):
    def __init__(self, loss_type='ciou', reduction='mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
    def forward(self, pred_boxes, target_boxes):
        iou = self._box_iou(pred_boxes, target_boxes)
        
        if self.loss_type == 'iou':
            loss = 1 - iou
        elif self.loss_type == 'giou':
            loss = 1 - self._giou(pred_boxes, target_boxes)
        elif self.loss_type == 'ciou':
            loss = 1 - self._ciou(pred_boxes, target_boxes)
        else:
            loss = 1 - iou
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
        
    def _box_iou(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        return inter / union
        
    def _giou(self, boxes1, boxes2):
        iou = self._box_iou(boxes1, boxes2)
        
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        area_c = wh[:, :, 0] * wh[:, :, 1]
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - iou * union
        
        return iou - (area_c - union) / area_c
        
    def _ciou(self, boxes1, boxes2):
        iou = self._box_iou(boxes1, boxes2)
        
        cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
        cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
        cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
        cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        rho2 = (cx1[:, None] - cx2) ** 2 + (cy1[:, None] - cy2) ** 2
        
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        c2 = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2
        
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]
        
        v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v)
            
        return iou - rho2 / c2 - alpha * v


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=20, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.reg_loss = IoULoss(loss_type='ciou')
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)
        
    def forward(self, cls_preds, reg_preds, targets):
        device = cls_preds[0].device
        total_cls_loss = 0
        total_reg_loss = 0
        num_pos = 0
        
        for level_idx, (cls_pred, reg_pred) in enumerate(zip(cls_preds, reg_preds)):
            batch_size, _, h, w = cls_pred.shape
            num_anchors = cls_pred.shape[1] // self.num_classes
            
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            
            for b in range(batch_size):
                target_boxes = targets['boxes'][b].to(device)
                target_labels = targets['labels'][b].to(device)
                
                if len(target_boxes) == 0:
                    # Background loss for all anchors
                    bg_labels = torch.zeros(cls_pred.shape[1], dtype=torch.long, device=device)
                    cls_loss = self.cls_loss(cls_pred[b], bg_labels)
                    total_cls_loss += cls_loss
                    continue
                    
                num_pos += len(target_boxes)
                
                # Simple assignment: use first N predictions for N targets
                # In practice, you'd use proper anchor matching
                num_targets = len(target_boxes)
                num_preds = cls_pred.shape[1]
                
                # Sample predictions for targets (simplified)
                pred_indices = torch.linspace(0, num_preds - 1, num_targets).long()
                
                cls_target = cls_pred[b, pred_indices]
                reg_target = reg_pred[b, pred_indices]
                
                cls_loss = self.cls_loss(cls_target, target_labels)
                reg_loss = self.smooth_l1(reg_target, target_boxes)
                
                total_cls_loss += cls_loss
                total_reg_loss += reg_loss
                
        if num_pos == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            
        return total_cls_loss / batch_size, total_reg_loss / batch_size