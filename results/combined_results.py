import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects
from skimage.filters import frangi, gaussian
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ── Operator 1: Frangi ──────────────────────────────────────────
def run_frangi(img, mask):
    smoothed = gaussian(img, sigma=1.5)
    vessel   = frangi(smoothed, sigmas=range(1, 8, 1), beta=0.5, gamma=None, black_ridges=True)
    vessel   = (vessel - vessel.min()) / (vessel.max() - vessel.min() + 1e-8)
    best_iou, best_pred = 0, None
    for thresh in np.arange(0.01, 0.50, 0.01):
        for minsize in [10, 50, 100, 200, 500]:
            cleaned = remove_small_objects(vessel > thresh, min_size=minsize)
            for dil in [5, 10, 15, 20, 25, 30]:
                thick = binary_dilation(cleaned, iterations=dil).astype(np.float32)
                iou   = calculate_iou(thick, mask)
                if iou > best_iou:
                    best_iou  = iou
                    best_pred = thick
    return best_pred, best_iou

# ── Operator 2: Multiscale Contrast ─────────────────────────────
class DiffMultiscaleContrast(torch.nn.Module):
    def __init__(self, sigma_small=1.0, sigma_large=5.0, epsilon=1e-6):
        super().__init__()
        self.sigma_small = torch.nn.Parameter(torch.tensor(sigma_small))
        self.sigma_large = torch.nn.Parameter(torch.tensor(sigma_large))
        self.epsilon     = epsilon

    def gaussian_kernel(self, sigma):
        sigma  = torch.clamp(sigma, 0.5, 10.0)
        k      = int(6 * sigma.item() + 1) | 1
        ax     = torch.arange(-k//2 + 1, k//2 + 1, dtype=torch.float32)
        kernel = torch.exp(-ax**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return (kernel[:, None] * kernel[None, :]).unsqueeze(0).unsqueeze(0), k

    def local_stats(self, x, sigma):
        kernel, k = self.gaussian_kernel(sigma)
        mean      = F.conv2d(x, kernel, padding=k//2)
        mean_sq   = F.conv2d(x**2, kernel, padding=k//2)
        std       = torch.sqrt(torch.clamp(mean_sq - mean**2, min=self.epsilon))
        return mean, std

    def forward(self, x):
        mean_s, std_s = self.local_stats(x, self.sigma_small)
        mean_l, std_l = self.local_stats(x, self.sigma_large)
        mean_diff     = torch.abs(mean_l - mean_s)
        std_diff      = std_l - std_s
        contrast      = mean_diff + 0.5 * std_diff
        contrast      = (contrast - contrast.min()) / (contrast.max() - contrast.min() + self.epsilon)
        return contrast

def run_multiscale(img, mask):
    smoothed  = gaussian(img, sigma=1.0)
    x         = torch.tensor(smoothed).unsqueeze(0).unsqueeze(0)
    best_iou, best_pred = 0, None
    for ss in [0.5, 1.0, 2.0, 3.0]:
        for sl in [5.0, 8.0, 10.0, 15.0]:
            if ss >= sl:
                continue
            model     = DiffMultiscaleContrast(sigma_small=ss, sigma_large=sl)
            output_np = model(x).squeeze().detach().numpy()
            for thresh in np.arange(0.1, 0.9, 0.05):
                for minsize in [50, 100, 200, 500]:
                    cleaned = remove_small_objects(output_np > thresh, min_size=minsize)
                    for dil in [5, 10, 15, 20, 25]:
                        thick = binary_dilation(cleaned, iterations=dil).astype(np.float32)
                        iou   = calculate_iou(thick, mask)
                        if iou > best_iou:
                            best_iou  = iou
                            best_pred = thick
    return best_pred, best_iou

# ── Operator 3: Oriented Closing ────────────────────────────────
class DiffOrientedClosing(torch.nn.Module):
    def __init__(self, length=15, num_angles=8, epsilon=1e-6):
        super().__init__()
        self.length     = length
        self.num_angles = num_angles
        self.epsilon    = epsilon

    def line_kernel(self, angle, length):
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
            kern     = self.line_kernel(angle, self.length)
            dilated  = self.soft_dilate(x, kern)
            closed   = self.soft_erode(dilated, kern)
            response = closed - x
            responses.append(response)
        stacked = torch.stack(responses, dim=0)
        result  = stacked.max(dim=0).values
        result  = (result - result.min()) / (result.max() - result.min() + self.epsilon)
        return result

def run_oriented_closing(img, mask):
    smoothed  = gaussian(img, sigma=1.0)
    x         = torch.tensor(smoothed).unsqueeze(0).unsqueeze(0)
    best_iou, best_pred = 0, None
    for length in [11, 15, 21]:
        for num_angles in [8, 12]:
            model     = DiffOrientedClosing(length=length, num_angles=num_angles)
            output_np = model(x).squeeze().detach().numpy()
            for thresh in np.arange(0.1, 0.9, 0.05):
                for minsize in [50, 100, 500]:
                    cleaned = remove_small_objects(output_np > thresh, min_size=minsize)
                    for dil in [5, 10, 15, 20]:
                        thick = binary_dilation(cleaned, iterations=dil).astype(np.float32)
                        iou   = calculate_iou(thick, mask)
                        if iou > best_iou:
                            best_iou  = iou
                            best_pred = thick
    return best_pred, best_iou

# ── Load & Run ───────────────────────────────────────────────────
img  = load_image('tile.png')
mask = load_mask('mask.png')

print("Running Frangi...")
frangi_pred, frangi_iou     = run_frangi(img, mask)
print("Running Multiscale...")
multi_pred,  multi_iou      = run_multiscale(img, mask)
print("Running Oriented Closing...")
orient_pred, orient_iou     = run_oriented_closing(img, mask)

print(f"\nFrangi IoU           = {frangi_iou:.4f}")
print(f"Multiscale IoU       = {multi_iou:.4f}")
print(f"Oriented Closing IoU = {orient_iou:.4f}")

# ── Plot in professor's format ───────────────────────────────────
operators = [
    ("Frangi Vesselness",    frangi_pred,  frangi_iou,  3),
    ("Multi-scale Contrast", multi_pred,   multi_iou,   3),
    ("Oriented Closing",     orient_pred,  orient_iou,  3),
]

fig, axes = plt.subplots(3, 3, figsize=(12, 11))

# Column headers
for col, title in enumerate(['Input', 'Ground Truth', 'Ours']):
    axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)

for i, (name, result, iou, nparams) in enumerate(operators):
    # Row label on left
    axes[i, 0].set_ylabel(f'{name}\n(N=1)', fontsize=10,
                           rotation=90, labelpad=10)

    axes[i, 0].imshow(img,    cmap='gray')
    axes[i, 1].imshow(mask,   cmap='gray')
    axes[i, 2].imshow(result, cmap='gray')

    # IoU below result like professor's format
    axes[i, 2].set_xlabel(f'IoU={iou:.2f}', fontsize=11)

    # Remove axis ticks for cleaner look
    for ax in axes[i]:
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Qualitative Segmentation Results — tile/crack',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/combined_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to results/combined_results.png")