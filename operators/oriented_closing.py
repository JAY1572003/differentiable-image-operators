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

class DiffOrientedClosing(torch.nn.Module):
    def __init__(self, length=15, num_angles=8, epsilon=1e-6):
        super().__init__()
        self.length     = length
        self.num_angles = num_angles
        self.epsilon    = epsilon

    def line_kernel(self, angle, length):
        # Create a line structuring element at given angle
        k    = length if length % 2 == 1 else length + 1
        kern = torch.zeros(k, k)
        cx   = k // 2
        cy   = k // 2
        for i in range(k):
            x = int(cx + (i - cx) * np.cos(angle))
            y = int(cy + (i - cx) * np.sin(angle))
            if 0 <= x < k and 0 <= y < k:
                kern[y, x] = 1.0
        if kern.sum() == 0:
            kern[cy, cx] = 1.0
        kern = kern / kern.sum()
        return kern.unsqueeze(0).unsqueeze(0)

    def soft_dilate(self, x, kernel):
        k = kernel.shape[-1]
        return F.conv2d(x, kernel, padding=k//2)

    def soft_erode(self, x, kernel):
        k   = kernel.shape[-1]
        neg = 1.0 - x
        out = F.conv2d(neg, kernel, padding=k//2)
        return 1.0 - out

    def forward(self, x):
        responses = []
        angles    = np.linspace(0, np.pi, self.num_angles, endpoint=False)

        for angle in angles:
            kern = self.line_kernel(angle, self.length)
            # Soft closing = dilate then erode
            dilated  = self.soft_dilate(x, kern)
            closed   = self.soft_erode(dilated, kern)
            # Response = difference between closed and original
            response = closed - x
            responses.append(response)

        # Take maximum response across all angles
        stacked  = torch.stack(responses, dim=0)
        result   = stacked.max(dim=0).values
        result   = (result - result.min()) / (result.max() - result.min() + self.epsilon)
        return result

# Load images
img      = load_image('tile.png')
mask     = load_mask('mask.png')
smoothed = gaussian(img, sigma=1.0)
x        = torch.tensor(smoothed).unsqueeze(0).unsqueeze(0)

# Search best parameters
print("Searching best parameters...")
best_iou    = 0
best_params = {}

for length in [11, 15, 21]:
    for num_angles in [8, 12]:
        model     = DiffOrientedClosing(length=length, num_angles=num_angles)
        output    = model(x)
        output_np = output.squeeze().detach().numpy()

        for thresh in np.arange(0.1, 0.9, 0.05):
            binary = output_np > thresh
            for minsize in [50, 100, 500]:
                cleaned = remove_small_objects(binary, min_size=minsize)
                for dil in [5, 10, 15, 20]:
                    thick = binary_dilation(cleaned, iterations=dil).astype(np.float32)
                    iou   = calculate_iou(thick, mask)
                    if iou > best_iou:
                        best_iou    = iou
                        best_params = {
                            'length'    : length,
                            'num_angles': num_angles,
                            'thresh'    : round(float(thresh), 2),
                            'minsize'   : minsize,
                            'dil'       : dil
                        }

print(f"\n✅ Best IoU = {best_iou:.4f}")
print(f"   Best params = {best_params}")

# Plot best result
model     = DiffOrientedClosing(
                length=best_params['length'],
                num_angles=best_params['num_angles'])
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
axes[2].set_title(f'Oriented Closing (IoU={best_iou:.4f})')
plt.tight_layout()
plt.savefig('results/oriented_closing_result.png')
plt.show()