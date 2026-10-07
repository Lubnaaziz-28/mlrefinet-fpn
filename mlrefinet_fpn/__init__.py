"""MLRefineFPN: Multi-Level Refinement Feature Pyramid Network.

A published CV architecture (Aziz et al., IVC 2021) that improves object
detection accuracy at zero additional compute cost by re-injecting refined
detail across all pyramid levels.
"""

from .anchors import AnchorGenerator, clip_boxes, decode_boxes, nms
from .backbone import Backbone
from .decode import decode_predictions
from .detection_head import DetectionHead
from .detector import MLRefineFPNDetector
from .mlrefine_fpn import FPN, MLRefineFPN, RefinementBlock

__all__ = [
    "FPN",
    "AnchorGenerator",
    "Backbone",
    "DetectionHead",
    "MLRefineFPN",
    "MLRefineFPNDetector",
    "RefinementBlock",
    "clip_boxes",
    "decode_boxes",
    "decode_predictions",
    "nms",
]
__version__ = "1.0.0"