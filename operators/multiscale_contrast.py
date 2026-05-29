import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects
from skimage.filters import gaussian

def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def load_mask(path):
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = color.rgb2gray(mask)
    mask = (mask > 0.5).astype(np.float32)
    return mask

def calculate_iou(pred, gt):
    pred_bin = (pred > 0).astype(np.float32)
    intersection = (pred_bin * gt).sum()
    union = ((pred_bin + gt) > 0).astype(np.float32).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

class DiffMultiscaleContrast(torch.nn.Module):
    def __init__(self, sigma_small=1.0, sigma_large=5.0, epsilon=1e-6):
        super().__init__()
        self.sigma_small = torch.nn.Parameter(torch.tensor(sigma_small))
        self.sigma_large = torch.nn.Parameter(torch.tensor(sigma_large))
        self.epsilon     = epsilon

    def gaussian_kernel(self, sigma):
        sigma = torch.clamp(sigma, 0.5, 10.0)
        k = int(6 * sigma.item() + 1) | 1
        ax = torch.arange(-k//2 + 1, k//2 + 1, dtype=torch.float32)
        kernel = torch.exp(-ax**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return (kernel[:, None] * kernel[None, :]).unsqueeze(0).unsqueeze(0), k

    def local_stats(self, x, sigma):
        kernel, k = self.gaussian_kernel(sigma)
        mean    = F.conv2d(x, kernel, padding=k//2)
        mean_sq = F.conv2d(x**2, kernel, padding=k//2)
        std     = torch.sqrt(torch.clamp(mean_sq - mean**2, min=self.epsilon))
        return mean, std

    def forward(self, x):
        mean_s, std_s = self.local_stats(x, self.sigma_small)
        mean_l, std_l = self.local_stats(x, self.sigma_large)

        # Contrast = difference between local mean and large-scale mean
        # This highlights dark cracks against bright background
        mean_diff = torch.abs(mean_l - mean_s)

        # Also use std difference
        std_diff  = std_l - std_s

        # Combine both signals
        contrast  = mean_diff + 0.5 * std_diff
        contrast  = (contrast - contrast.min()) / (contrast.max() - contrast.min() + self.epsilon)
        return contrast

# Load images
img      = load_image('tile.png')
mask     = load_mask('mask.png')

# Pre-smooth to reduce noise
smoothed = gaussian(img, sigma=1.0)
x        = torch.tensor(smoothed).unsqueeze(0).unsqueeze(0)

# Search best parameters
print("Searching best parameters...")
best_iou    = 0
best_params = {}

for sigma_small in [0.5, 1.0, 2.0, 3.0]:
    for sigma_large in [5.0, 8.0, 10.0, 15.0]:
        if sigma_small >= sigma_large:
            continue
        model     = DiffMultiscaleContrast(sigma_small=sigma_small, sigma_large=sigma_large)
        output    = model(x)
        output_np = output.squeeze().detach().numpy()

        for thresh in np.arange(0.1, 0.9, 0.05):
            binary = output_np > thresh
            for minsize in [50, 100, 200, 500]:
                cleaned = remove_small_objects(binary, min_size=minsize)
                for dil in [5, 10, 15, 20, 25]:
                    thick = binary_dilation(cleaned, iterations=dil).astype(np.float32)
                    iou   = calculate_iou(thick, mask)
                    if iou > best_iou:
                        best_iou    = iou
                        best_params = {
                            'sigma_small': sigma_small,
                            'sigma_large': sigma_large,
                            'thresh'     : round(float(thresh), 2),
                            'minsize'    : minsize,
                            'dil'        : dil
                        }

print(f"\n✅ Best IoU = {best_iou:.4f}")
print(f"   Best params = {best_params}")

# Plot best result
model     = DiffMultiscaleContrast(
                sigma_small=best_params['sigma_small'],
                sigma_large=best_params['sigma_large'])
output    = model(x)
output_np = output.squeeze().detach().numpy()
binary    = output_np > best_params['thresh']
cleaned   = remove_small_objects(binary, min_size=best_params['minsize'])
best_pred = binary_dilation(cleaned, iterations=best_params['dil']).astype(np.float32)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Input')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(best_pred, cmap='gray')
axes[2].set_title(f'Multi-scale Contrast (IoU={best_iou:.4f})')
plt.tight_layout()
plt.savefig('results/multiscale_result.png')
plt.show()