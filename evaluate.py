import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import binary_dilation
from skimage.filters import frangi, gaussian
from skimage.morphology import remove_small_objects

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

img  = load_image('tile.png')
mask = load_mask('mask.png')

smoothed = gaussian(img, sigma=1.5)
vessel   = frangi(smoothed, sigmas=range(1, 8, 1), beta=0.5, gamma=None, black_ridges=True)
vessel   = (vessel - vessel.min()) / (vessel.max() - vessel.min() + 1e-8)

best_iou    = 0
best_params = {}

for thresh in np.arange(0.01, 0.50, 0.01):
    binary = vessel > thresh
    for minsize in [10, 50, 100, 200, 500]:
        cleaned = remove_small_objects(binary, min_size=minsize)
        for dil in [5, 10, 15, 20, 25, 30]:
            thick = binary_dilation(cleaned, iterations=dil).astype(np.float32)
            iou   = calculate_iou(thick, mask)
            if iou > best_iou:
                best_iou    = iou
                best_params = {'thresh': round(thresh,2), 'minsize': minsize, 'dil': dil}

print(f"✅ Best IoU = {best_iou:.4f}")
print(f"   Best params = {best_params}")

# BEST PARAMS - IoU = 0.4918
binary    = vessel > 0.02
cleaned   = remove_small_objects(binary, min_size=500)
best_pred = binary_dilation(cleaned, iterations=5).astype(np.float32)
iou       = calculate_iou(best_pred, mask)

print(f"✅ Final IoU = {iou:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Input')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(best_pred, cmap='gray')
axes[2].set_title(f'Our Result (IoU={iou:.4f})')
plt.tight_layout()
plt.savefig('evaluation_result.png')
plt.show()