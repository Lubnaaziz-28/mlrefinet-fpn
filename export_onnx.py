import torch
import torch.onnx
from mlrefinet_fpn.model import create_model


def export_onnx(model, output_path, input_size=(640, 640), opset_version=17):
    model.eval()
    
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['cls_preds', 'reg_preds'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'cls_preds': {0: 'batch_size'},
            'reg_preds': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='model.onnx')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640])
    parser.add_argument('--num-classes', type=int, default=20)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--fpn-channels', type=int, default=256)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(
        num_classes=args.num_classes,
        backbone=args.backbone,
        fpn_channels=args.fpn_channels,
        pretrained=False
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    export_onnx(model, args.output, tuple(args.input_size))


if __name__ == '__main__':
    main()