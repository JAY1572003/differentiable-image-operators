"""
Evaluation for DiffHaralick on carpet/cut texture-disruption data.
Same protocol as evaluate_proper.py: train on 4, test on 2 held-out, IoU only.

Images are resized to 256x256 before processing -- DiffHaralick's local
soft-GLCM computation takes ~8.4s per forward+backward pass at full 1024x1024
resolution (confirmed via timing test), which is impractical for training.
At 256x256 it runs in ~0.26s, which is workable.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage import io, color
import os
import sys
sys.path.insert(0, '.')

from operators.haralick import DiffHaralick
from operations.losses import BCEDiceLoss, compute_iou

RESIZE_TO = 256

def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
    return t

def load_mask(path):
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = color.rgb2gray(mask)
    mask = (mask > 0.5).astype(np.float32)
    t = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(RESIZE_TO, RESIZE_TO), mode='nearest')
    return t

def load_dataset(folder, indices):
    images, masks = [], []
    for i in indices:
        img  = load_image(f'{folder}/images/{i:03d}.png')
        mask = load_mask(f'{folder}/masks/{i:03d}_mask.png')
        images.append(img)
        masks.append(mask)
    return images, masks


DATA_FOLDER = 'carpet_cut_pairs'
TRAIN_IDS   = [0, 1, 2, 3]
TEST_IDS    = [4, 5]

print("Loading carpet/cut data...")
train_images, train_masks = load_dataset(DATA_FOLDER, TRAIN_IDS)
test_images,  test_masks  = load_dataset(DATA_FOLDER, TEST_IDS)
print(f"Train: {len(train_images)} images, Test: {len(test_images)} images (resized to {RESIZE_TO}x{RESIZE_TO})")

op = DiffHaralick()
loss_fn = BCEDiceLoss(pos_weight=15.0)
optimizer = torch.optim.Adam(op.parameters(), lr=0.01)

epochs = 150
print(f"\nTraining DiffHaralick for {epochs} epochs...")
for epoch in range(epochs):
    total_loss = 0
    for img, mask in zip(train_images, train_masks):
        optimizer.zero_grad()
        pred = op(img)
        # DiffHaralick outputs 3 channels (contrast/homogeneity/energy).
        # Use the contrast channel as the primary defect-response map --
        # texture disruption typically shows as a local contrast spike.
        pred_single = pred[:, 0:1, :, :]
        loss = loss_fn(pred_single, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1}/{epochs} loss={total_loss/len(train_images):.4f}")

print("\nEvaluating on held-out test images...")
ious = []
with torch.no_grad():
    for img, mask in zip(test_images, test_masks):
        pred = op(img)[:, 0:1, :, :]
        pred_binary = (pred > 0.5).float()
        mask_binary = (mask > 0.5).float()
        iou = compute_iou(pred_binary, mask_binary)
        ious.append(iou)
        print(f"  IoU: {iou:.4f}")

print(f"\nDiffHaralick Test IoU (mean) = {np.mean(ious):.4f}")