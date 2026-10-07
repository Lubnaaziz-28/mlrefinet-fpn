import pytest
import torch


def test_import():
    from mlrefinet_fpn import MLRefineFPNDetector
    assert MLRefineFPNDetector is not None


def test_requirements():
    import importlib
    for pkg in ["torch", "torchvision"]:
        importlib.import_module(pkg)


def test_forward_and_detect():
    from mlrefinet_fpn import MLRefineFPNDetector
    model = MLRefineFPNDetector(num_classes=20)
    model.eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        cls_scores, bboxes = model(x)
    assert len(cls_scores) == 4
    assert len(bboxes) == 4
    res = model.detect(x, score_threshold=0.01)
    assert len(res) == 1
    boxes, scores, labels = res[0]
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.shape[0] == boxes.shape[0]
    assert labels.shape[0] == boxes.shape[0]


def test_onnx_export(tmp_path):
    from mlrefinet_fpn import MLRefineFPNDetector
    import torch.nn.functional as F
    import torch.onnx
    import onnxruntime as ort

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, det):
            super().__init__()
            self.det = det

        def forward(self, images):
            cls_scores, bboxes = self.det(images)
            ref_h = max(c.shape[2] for c in cls_scores)
            ref_w = max(c.shape[3] for c in cls_scores)

            def pad(t):
                _, _, h, w = t.shape
                return F.pad(t, (0, ref_w - w, 0, ref_h - h), mode="constant", value=0)

            cls = torch.cat([pad(c) for c in cls_scores], dim=1)
            bbox = torch.cat([pad(b) for b in bboxes], dim=1)
            return cls, bbox

    model = MLRefineFPNDetector(num_classes=20).eval()
    wrapped = ONNXWrapper(model)
    out = str(tmp_path / "mlrefinet.onnx")
    torch.onnx.export(wrapped, torch.randn(1, 3, 320, 320), out,
                      input_names=["input"], output_names=["cls", "bbox"], opset_version=14)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"input": torch.randn(1, 3, 320, 320).numpy()})
    assert o[0].ndim == 4 and o[1].ndim == 4