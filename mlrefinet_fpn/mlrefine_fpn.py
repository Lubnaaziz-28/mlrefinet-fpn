"""Feature Pyramid Network + MLRefineFPN refinement module."""

import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """Standard lateral + top-down Feature Pyramid Network.

    Produces a multi-scale pyramid P2..P5 from backbone features C2..C5.
    """

    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        # ResNet-50 Bottleneck outputs: C2=256, C3=512, C4=1024, C5=2048
        self.lateral4 = nn.Conv2d(2048, out_channels, 1)
        self.lateral3 = nn.Conv2d(1024, out_channels, 1)
        self.lateral2 = nn.Conv2d(512, out_channels, 1)
        self.lateral1 = nn.Conv2d(256, out_channels, 1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.out_channels = out_channels

    def forward(self, feats):
        c2, c3, c4, c5 = feats
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lateral1(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p5 = self.smooth4(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)
        return [p2, p3, p4, p5]


class RefinementBlock(nn.Module):
    """Multi-Level Refinement Block — re-injects refined detail across levels."""

    def __init__(self, in_channels=256, num_refinements=2):
        super().__init__()
        self.num_refinements = num_refinements
        self.refine_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False) for _ in range(num_refinements)]
        )
        self.refine_bns = nn.ModuleList(
            [nn.BatchNorm2d(in_channels) for _ in range(num_refinements)]
        )
        self.fuse_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False) for _ in range(num_refinements)]
        )
        self.fuse_bns = nn.ModuleList(
            [nn.BatchNorm2d(in_channels) for _ in range(num_refinements)]
        )

    def forward(self, pyramid):
        refined = list(pyramid)
        for i in range(self.num_refinements):
            # refine each level independently
            new = []
            for j, p in enumerate(refined):
                r = F.relu(self.refine_bns[i](self.refine_convs[i](p)))
                new.append(r)
            # fuse top-down across levels
            for j in range(len(new) - 2, -1, -1):
                up = F.interpolate(new[j + 1], size=new[j].shape[2:], mode="nearest")
                fused = F.relu(self.fuse_bns[i](self.fuse_convs[i](new[j] + up)))
                new[j] = fused
            refined = new
        return refined


class MLRefineFPN(nn.Module):
    """MLRefineFPN: FPN followed by multi-level refinement blocks."""

    def __init__(self, in_channels=256, out_channels=256, num_refinements=2):
        super().__init__()
        self.fpn = FPN(in_channels, out_channels)
        self.refinement = RefinementBlock(out_channels, num_refinements)
        self.out_channels = out_channels

    def forward(self, feats):
        pyramid = self.fpn(feats)
        refined = self.refinement(pyramid)
        return refined