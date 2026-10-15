import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            l_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
    def forward(self, features):
        laterals = [l_conv(features[i]) for i, l_conv in enumerate(self.lateral_convs)]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )
            
        outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        return outs


class RefineBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + identity)


class MultiLevelRefinement(nn.Module):
    def __init__(self, fpn_channels=256, num_levels=5, num_refine=2):
        super().__init__()
        self.refine_blocks = nn.ModuleList([
            nn.Sequential(*[RefineBlock(fpn_channels) for _ in range(num_refine)])
            for _ in range(num_levels)
        ])
        
    def forward(self, features):
        return [refine(feat) for refine, feat in zip(self.refine_blocks, features)]


class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=20, num_anchors=9):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * num_anchors, 1)
        )
        
    def forward(self, x):
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        return cls, reg


class MLRefineFPN(nn.Module):
    def __init__(self, num_classes=20, backbone="resnet50", fpn_channels=256, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone_model = resnet50(weights=weights)
            self.backbone = nn.Sequential(
                backbone_model.conv1,
                backbone_model.bn1,
                backbone_model.relu,
                backbone_model.maxpool,
                backbone_model.layer1,
                backbone_model.layer2,
                backbone_model.layer3,
                backbone_model.layer4,
            )
            in_channels_list = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        self.fpn = FPN(in_channels_list, fpn_channels)
        self.refinement = MultiLevelRefinement(fpn_channels)
        
        self.p6 = nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1)
        
        self.head = DetectionHead(fpn_channels, num_classes)
        
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 5, 6, 7]:
                features.append(x)
                
        fpn_features = self.fpn(features)
        refined = self.refinement(fpn_features)
        
        p6 = self.p6(refined[-1])
        p7 = self.p7(F.relu(p6))
        all_features = refined + [p6, p7]
        
        cls_preds = []
        reg_preds = []
        for feat in all_features:
            cls, reg = self.head(feat)
            cls_preds.append(cls)
            reg_preds.append(reg)
            
        return cls_preds, reg_preds


def create_model(num_classes=20, backbone="resnet50", fpn_channels=256, pretrained=True):
    return MLRefineFPN(num_classes, backbone, fpn_channels, pretrained)