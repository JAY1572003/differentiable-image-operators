"""
Debug script — trains DiffMorphologyDisk on all 4 training images
(matching evaluate_proper.py), prints loss + radius + gradient every
10 epochs so we can see exactly where/why it stalls.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage import io, color
import sys
sys.path.insert(0, '.')

from operators.diff_morphology import DiffMorphologyDisk
from operations.losses import BCEDiceLoss

def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

def load_mask(path):
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = color.rgb2gray(mask)
    mask = (mask > 0.5).astype(np.float32)
    return torch.tensor(mask).unsqueeze(0).unsqueeze(0)

train_ids = [0, 1, 2, 3]
images = [load_image(f'tile_crack_pairs/images/{i:03d}.png') for i in train_ids]
masks  = [load_mask(f'tile_crack_pairs/masks/{i:03d}_mask.png') for i in train_ids]

for i, m in zip(train_ids, masks):
    print(f"Image {i:03d}: mask positive fraction = {m.mean():.4f}")

op = DiffMorphologyDisk(operation='bottom_hat')
loss_fn = BCEDiceLoss()
optimizer = torch.optim.Adam(op.parameters(), lr=0.01)

print(f"\nInitial radius: {op.params['radius'].item():.4f}")
print(f"Radius allowed range: {op.param_ranges['radius']}\n")

for epoch in range(200):
    total_loss = 0.0
    for img, mask in zip(images, masks):
        optimizer.zero_grad()
        pred = op(img)
        if pred.shape != mask.shape:
            pred = F.interpolate(pred, size=mask.shape[2:])
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0 or epoch == 199:
        grad = op.params['radius'].grad
        grad_val = grad.item() if grad is not None else None
        print(f"Epoch {epoch:3d} | avg_loss={total_loss/len(images):.6f} | "
              f"radius={op.params['radius'].item():.4f} | grad={grad_val}")

print("\nDone. Look at whether radius moves at all over 200 epochs, "
      "and whether it ends up sitting near 0.129 (the floor) or 2.87 (the ceiling) again.")