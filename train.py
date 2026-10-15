import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

from mlrefinet_fpn.model import create_model
from mlrefinet_fpn.dataset import COCODataset, collate_fn
from mlrefinet_fpn.loss import DetectionLoss


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, wandb_run=None):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, bboxes, labels, _, _ in pbar:
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'cls': total_cls_loss / (pbar.n + 1),
            'reg': total_reg_loss / (pbar.n + 1)
        })
        
    avg_loss = total_loss / len(loader)
    if wandb_run:
        wandb_run.log({
            'train/loss': avg_loss,
            'train/cls_loss': total_cls_loss / len(loader),
            'train/reg_loss': total_reg_loss / len(loader),
            'epoch': epoch
        })
    return avg_loss


def validate(model, loader, criterion, device, epoch, wandb_run=None):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    with torch.no_grad():
        for images, bboxes, labels, _, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            targets = {
                'boxes': [b.to(device) for b in bboxes],
                'labels': [l.to(device) for l in labels]
            }
            
            cls_preds, reg_preds = model(images)
            cls_loss, reg_loss = criterion(cls_preds, reg_preds, targets)
            loss = cls_loss + reg_loss
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            
    avg_loss = total_loss / len(loader)
    if wandb_run:
        wandb_run.log({
            'val/loss': avg_loss,
            'val/cls_loss': total_cls_loss / len(loader),
            'val/reg_loss': total_reg_loss / len(loader),
            'epoch': epoch
        })
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/voc.yaml')
    parser.add_argument('--data', type=str, help='Path to COCO data directory')
    parser.add_argument('--epochs', type=int, help='Override epochs from config')
    parser.add_argument('--batch-size', type=int, help='Override batch size from config')
    parser.add_argument('--lr', type=float, help='Override learning rate from config')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='mlrefinet-fpn')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--export-onnx', action='store_true', help='Export ONNX after training')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.data:
        config['data']['train_ann'] = os.path.join(args.data, 'train.json')
        config['data']['val_ann'] = os.path.join(args.data, 'val.json')
        config['data']['img_dir'] = os.path.join(args.data, 'images')
        
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.lr:
        config['train']['lr'] = args.lr
    if args.wandb:
        config['wandb']['enabled'] = True
    if args.wandb_project:
        config['wandb']['project'] = args.wandb_project
    if args.wandb_entity:
        config['wandb']['entity'] = args.wandb_entity
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
        
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
        
    wandb_run = None
    if config['wandb']['enabled']:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                config=config
            )
        except Exception as e:
            print(f"Wandb init failed: {e}. Continuing without wandb.")
            config['wandb']['enabled'] = False
            
    save_dir = Path(config['output']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, wandb_run)
        val_loss = validate(model, val_loader, criterion, device, epoch, wandb_run)
        
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, save_dir / 'best_model.pth')
            print(f"Saved best model (val_loss={best_val_loss:.4f})")
            
        if (epoch + 1) % config['output']['save_period'] == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
            
    if config['output']['export_onnx'] or args.export_onnx:
        from mlrefinet_fpn.export_onnx import export_onnx
        export_onnx(model, save_dir / 'best_model.onnx', config['data']['input_size'])
        
    if wandb_run:
        wandb_run.finish()
        
    print("Training complete!")


if __name__ == '__main__':
    main()