import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCODataset(Dataset):
    def __init__(self, ann_file, img_dir, input_size=(640, 640), augment=False):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.input_size = input_size
        self.augment = augment
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        if augment:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=max(input_size)),
                A.PadIfNeeded(min_height=input_size[0], min_width=input_size[1], border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=max(input_size)),
                A.PadIfNeeded(min_height=input_size[0], min_width=input_size[1], border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
            
    def __len__(self):
        return len(self.img_ids)
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        bboxes = []
        labels = []
        for ann in anns:
            if ann['iscrowd'] or ann['area'] <= 0:
                continue
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                bboxes.append([x, y, w, h])
                labels.append(self.cat_id_to_idx[ann['category_id']])
                
        if len(bboxes) == 0:
            bboxes = [[0, 0, 1, 1]]
            labels = [0]
            
        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        
        return {
            'image': transformed['image'],
            'bboxes': torch.tensor(transformed['bboxes'], dtype=torch.float32),
            'labels': torch.tensor(transformed['labels'], dtype=torch.long),
            'img_id': img_id,
            'orig_size': (img_info['height'], img_info['width'])
        }


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    img_ids = [item['img_id'] for item in batch]
    orig_sizes = [item['orig_size'] for item in batch]
    return images, bboxes, labels, img_ids, orig_sizes