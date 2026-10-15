#!/usr/bin/env python3
"""
Generate synthetic COCO-format dataset for testing.
"""
import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path


def generate_synthetic_data(output_dir, num_images=20, num_classes=3, img_size=(640, 640)):
    output_dir = Path(output_dir)
    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    
    categories = [
        {'id': i+1, 'name': f'class_{i+1}', 'supercategory': 'object'}
        for i in range(num_classes)
    ]
    
    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    
    ann_id = 1
    split_idx = int(num_images * 0.8)
    
    for img_id in range(1, num_images + 1):
        # Create synthetic image with random shapes
        img = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
        
        # Add some structured patterns
        for _ in range(np.random.randint(1, 5)):
            x = np.random.randint(0, img_size[1] - 50)
            y = np.random.randint(0, img_size[0] - 50)
            w = np.random.randint(20, 100)
            h = np.random.randint(20, 100)
            color = np.random.randint(0, 255, 3).tolist()
            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
        filename = f'image_{img_id:04d}.jpg'
        cv2.imwrite(str(img_dir / filename), img)
        
        # Generate annotations
        num_objects = np.random.randint(1, 4)
        for _ in range(num_objects):
            cat_id = np.random.randint(1, num_classes + 1)
            x = np.random.randint(0, img_size[1] - 80)
            y = np.random.randint(0, img_size[0] - 80)
            w = np.random.randint(30, 120)
            h = np.random.randint(30, 120)
            
            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [float(x), float(y), float(w), float(h)],
                'area': float(w * h),
                'iscrowd': 0
            }
            ann_id += 1
            
            if img_id <= split_idx:
                train_annotations.append(ann)
            else:
                val_annotations.append(ann)
        
        img_info = {
            'id': img_id,
            'file_name': filename,
            'width': img_size[1],
            'height': img_size[0]
        }
        
        if img_id <= split_idx:
            train_images.append(img_info)
        else:
            val_images.append(img_info)
    
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }
    
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(output_dir / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Generated synthetic dataset at {output_dir}")
    print(f"  Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"  Val: {len(val_images)} images, {len(val_annotations)} annotations")
    print(f"  Classes: {num_classes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-images', type=int, default=20)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 640])
    args = parser.parse_args()
    
    generate_synthetic_data(args.output, args.num_images, args.num_classes, tuple(args.img_size))


if __name__ == '__main__':
    main()