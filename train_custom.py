#!/usr/bin/env python3
"""
train_custom.py - Train mlrefinet-fpn on custom COCO-format data locally.

Usage:
    python train_custom.py --data-path /path/to/coco/data --epochs 50 --batch-size 8
    python train_custom.py --data-path /path/to/data --wandb --wandb-project my-project
    python train_custom.py --data-path /path/to/data --export-onnx --output-dir ./my_outputs
"""

import os
import sys
import yaml
import argparse
import tempfile
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from mlrefinet_fpn.model import create_model
from mlrefinet_fpn.dataset import COCODataset, collate_fn
from mlrefinet_fpn.loss import DetectionLoss
from export_onnx import export_onnx


def create_config_from_args(args, num_classes):
    """Create training config from CLI arguments."""
    return {
        'model': {
            'name': 'mlrefinet_fpn',
            'num_classes': num_classes,
            'backbone': args.backbone,
            'fpn_channels': args.fpn_channels,
            'pretrained': args.pretrained
        },
        'train': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'lr_scheduler': args.lr_scheduler,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'num_workers': args.num_workers,
            'grad_clip': args.grad_clip
        },
        'data': {
            'train_ann': os.path.join(args.data_path, 'train.json'),
            'val_ann': os.path.join(args.data_path, 'val.json'),
            'img_dir': os.path.join(args.data_path, 'images'),
            'input_size': args.input_size,
            'augment': not args.no_augment
        },
        'output': {
            'save_dir': args.output_dir,
            'save_period': args.save_period,
            'export_onnx': args.export_onnx
        },
        'wandb': {
            'enabled': args.wandb,
            'project': args.wandb_project,
            'entity': args.wandb_entity
        }
    }


def validate_data(data_path):
    """Validate that COCO data exists and get class info."""
    train_ann = os.path.join(data_path, 'train.json')
    val_ann = os.path.join(data_path, 'val.json')
    img_dir = os.path.join(data_path, 'images')
    
    for f in [train_ann, val_ann]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    
    import json
    with open(train_ann) as f:
        train_data = json.load(f)
    
    categories = train_data['categories']
    num_classes = len(categories)
    class_names = [c['name'] for c in categories]
    
    print(f"Found {len(train_data['images'])} training images")
    print(f"Found {len(train_data['annotations'])} training annotations")
    print(f"Classes ({num_classes}): {class_names}")
    
    return num_classes, class_names


def train(config, device, wandb_run=None):
    """Main training loop."""
    train_dataset = COCODataset(
        config['data']['train_ann'],
        config['data']['img_dir'],
        tuple(config['data']['input_size']),
        augment=config['data']['augment']
    )
    val_dataset = COCODataset(
        config['data']['val_ann'],
        config['data']['img_dir'],
        tuple(config['data']['input_size']),
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    model = create_model(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        fpn_channels=config['model']['fpn_channels'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    criterion = DetectionLoss(num_classes=config['model']['num_classes'])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['train']['lr'],
        momentum=config['train']['momentum'],
        weight_decay=config['train']['weight_decay']
    )
    
    if config['train']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['train']['epochs']
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    save_dir = Path(config['output']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['train']['epochs']):
        # Train
        model.train()
        total_loss = total_cls = total_reg = 0
        
        for images, bboxes, labels, _, _ in train_loader:
            images = images.to(device)
            targets = {
                'boxes': [b.to(device) for b in bboxes],
                'labels': [l.to(device) for l in labels]
            }
            
            optimizer.zero_grad()
            cls_preds, reg_preds = model(images)
            cls_loss, reg_loss = criterion(cls_preds, reg_preds, targets)
            loss = cls_loss + reg_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_reg += reg_loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = val_cls = val_reg = 0
        with torch.no_grad():
            for images, bboxes, labels, _, _ in val_loader:
                images = images.to(device)
                targets = {
                    'boxes': [b.to(device) for b in bboxes],
                    'labels': [l.to(device) for l in labels]
                }
                cls_preds, reg_preds = model(images)
                cls_loss, reg_loss = criterion(cls_preds, reg_preds, targets)
                val_loss += (cls_loss + reg_loss).item()
                val_cls += cls_loss.item()
                val_reg += reg_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['train']['epochs']}: "
              f"Train Loss={avg_train_loss:.4f} (cls={total_cls/len(train_loader):.4f}, reg={total_reg/len(train_loader):.4f}), "
              f"Val Loss={avg_val_loss:.4f} (cls={val_cls/len(val_loader):.4f}, reg={val_reg/len(val_loader):.4f})")
        
        if wandb_run:
            wandb_run.log({
                'train/loss': avg_train_loss,
                'train/cls_loss': total_cls / len(train_loader),
                'train/reg_loss': total_reg / len(train_loader),
                'val/loss': avg_val_loss,
                'val/cls_loss': val_cls / len(val_loader),
                'val/reg_loss': val_reg / len(val_loader),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        # Periodic save
        if (epoch + 1) % config['output']['save_period'] == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    return model, save_dir / 'best_model.pth'


def main():
    parser = argparse.ArgumentParser(description='Train mlrefinet-fpn on custom COCO data')
    parser.add_argument('--data-path', type=str, required=True, help='Path to COCO dataset (train.json, val.json, images/)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='LR scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--grad-clip', type=float, default=10.0, help='Gradient clipping')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone architecture')
    parser.add_argument('--fpn-channels', type=int, default=256, help='FPN channels')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Don\'t use pretrained')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640], help='Input size H W')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--save-period', type=int, default=5, help='Checkpoint save period')
    parser.add_argument('--export-onnx', action='store_true', help='Export ONNX after training')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='mlrefinet-fpn', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username/team)')
    
    args = parser.parse_args()
    
    # Validate data
    print(f"Validating data at: {args.data_path}")
    num_classes, class_names = validate_data(args.data_path)
    
    # Create config
    config = create_config_from_args(args, num_classes)
    
    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'train_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup W&B
    wandb_run = None
    if config['wandb']['enabled']:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                config=config,
                name=f"mlrefinet-fpn-{args.backbone}-{num_classes}cls"
            )
            print("Weights & Biases initialized")
        except Exception as e:
            print(f"W&B initialization failed: {e}. Continuing without W&B.")
    
    # Train
    print("\nStarting training...")
    model, best_model_path = train(config, device, wandb_run)
    
    # Export ONNX
    if config['output']['export_onnx'] or args.export_onnx:
        print("\nExporting to ONNX...")
        onnx_path = Path(args.output_dir) / 'model.onnx'
        export_onnx(model, onnx_path, tuple(config['data']['input_size']))
        print(f"ONNX model saved to: {onnx_path}")
    
    # Save class names
    class_names_path = Path(args.output_dir) / 'class_names.txt'
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class names saved to: {class_names_path}")
    
    if wandb_run:
        wandb_run.finish()
    
    print("\n✅ Training complete!")
    print(f"Best model: {best_model_path}")
    if config['output']['export_onnx'] or args.export_onnx:
        print(f"ONNX model: {Path(args.output_dir) / 'model.onnx'}")


if __name__ == '__main__':
    main()