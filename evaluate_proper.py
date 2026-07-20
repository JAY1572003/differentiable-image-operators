"""
Proper evaluation following professor's requirements:
- Train on 4 images using BCEDiceLoss
- Test on 2 held-out images
- Report IoU on test images only
- No threshold searching on test images
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os
import sys
sys.path.insert(0, '.')

from operators.frangi import DiffFrangi
from operators.multiscale_contrast import DiffMultiscaleContrast
from operators.diff_morphology import DiffMorphologyDisk, DiffMorphologyLine
from operations.losses import BCEDiceLoss

def compute_iou(pred_binary, target_binary):
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()

# ── Data loading ─────────────────────────────────────────────────────────────

def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

def load_mask(path):
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = color.rgb2gray(mask)
    mask = (mask > 0.5).astype(np.float32)
    return torch.tensor(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

def load_dataset(folder, indices):
    images, masks = [], []
    for i in indices:
        img  = load_image(f'{folder}/images/{i:03d}.png')
        mask = load_mask(f'{folder}/masks/{i:03d}_mask.png')
        images.append(img)
        masks.append(mask)
    return images, masks

# ── Training ──────────────────────────────────────────────────────────────────

def train_operator(op, train_images, train_masks, epochs=200, lr=0.05):
    loss_fn  = BCEDiceLoss()
    optimizer = torch.optim.Adam(op.parameters(), lr=lr)
    
    print(f"  Training {op.name}...")
    for epoch in range(epochs):
        total_loss = 0
        for img, mask in zip(train_images, train_masks):
            optimizer.zero_grad()
            pred = op(img)
            # Resize pred to match mask if needed
            if pred.shape != mask.shape:
                pred = F.interpolate(pred, size=mask.shape[2:])
            loss = loss_fn(pred, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs} loss={total_loss/len(train_images):.4f}")
    return op

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_operator(op, test_images, test_masks):
    ious = []
    with torch.no_grad():
        for img, mask in zip(test_images, test_masks):
            pred = op(img)
            if pred.shape != mask.shape:
                pred = F.interpolate(pred, size=mask.shape[2:])
            # Binarize at 0.5 as professor specified
            pred_binary = (pred > 0.5).float()
            mask_binary = (mask > 0.5).float()
            iou = compute_iou(pred_binary, mask_binary)
            ious.append(iou)
    return float(np.mean(ious))

# ── Visualize ─────────────────────────────────────────────────────────────────

def visualize_results(op, test_images, test_masks, name, save_path):
    fig, axes = plt.subplots(len(test_images), 3,
                             figsize=(12, 4 * len(test_images)))
    if len(test_images) == 1:
        axes = [axes]

    with torch.no_grad():
        for i, (img, mask) in enumerate(zip(test_images, test_masks)):
            pred = op(img)
            if pred.shape != mask.shape:
                pred = F.interpolate(pred, size=mask.shape[2:])
            pred_binary = (pred > 0.5).float()
            iou = compute_iou(pred_binary, (mask > 0.5).float())

            axes[i][0].imshow(img.squeeze().numpy(),  cmap='gray')
            axes[i][0].set_title('Input')
            axes[i][0].axis('off')
            axes[i][1].imshow(mask.squeeze().numpy(), cmap='gray')
            axes[i][1].set_title('Ground Truth')
            axes[i][1].axis('off')
            axes[i][2].imshow(pred_binary.squeeze().numpy(), cmap='gray')
            axes[i][2].set_title(f'{name} IoU={iou:.3f}')
            axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved: {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

DATA_FOLDER  = 'tile_crack_pairs'
TRAIN_IDS    = [0, 1, 2, 3]   # train on these
TEST_IDS     = [4, 5]          # never seen during training

print("Loading data...")
train_images, train_masks = load_dataset(DATA_FOLDER, TRAIN_IDS)
test_images,  test_masks  = load_dataset(DATA_FOLDER, TEST_IDS)
print(f"Train: {len(train_images)} images, Test: {len(test_images)} images")

# Define operators to evaluate
operators = [
    ("DiffFrangi",            DiffFrangi()),
    ("DiffMultiscaleContrast",DiffMultiscaleContrast()),
    ("DiffMorphologyDisk",    DiffMorphologyDisk(operation='bottom_hat')),
    ("DiffMorphologyLine",    DiffMorphologyLine(operation='bottom_hat')),
]

results = {}
os.makedirs('results', exist_ok=True)

for name, op in operators:
    print(f"\n{'='*40}")
    print(f"Operator: {name}")

    # Train
    op = train_operator(op, train_images, train_masks, epochs=200, lr=0.05)

    # Evaluate on held-out test images
    iou = evaluate_operator(op, test_images, test_masks)
    results[name] = iou
    print(f"  Test IoU = {iou:.4f}")

    # Save visual results
    visualize_results(op, test_images, test_masks, name,
                      f'results/proper_eval_{name}.png')

# Summary
print(f"\n{'='*40}")
print("RESULTS SUMMARY")
print(f"{'='*40}")
for name, iou in results.items():
    print(f"{name:30s} IoU = {iou:.4f}")