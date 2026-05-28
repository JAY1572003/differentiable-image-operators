import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Load image
def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img

# Differentiable Frangi Filter
class DiffFrangi(torch.nn.Module):
    def __init__(self, scale=2.0, beta=0.5, gamma=0.01):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.beta  = torch.nn.Parameter(torch.tensor(beta))
        self.gamma = torch.nn.Parameter(torch.tensor(gamma))

    def forward(self, x):
        sigma = torch.clamp(self.scale, 0.5, 8.0)
        k = int(6 * sigma.item() + 1) | 1
        ax = torch.arange(-k//2 + 1, k//2 + 1, dtype=torch.float32)
        kernel = torch.exp(-ax**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel_2d = (kernel[:, None] * kernel[None, :]).unsqueeze(0).unsqueeze(0)

        x_smooth = F.conv2d(x, kernel_2d, padding=k//2)

        Ixx = F.conv2d(x_smooth, torch.tensor([[[[1, -2, 1]]]], dtype=torch.float32), padding=(0,1))
        Iyy = F.conv2d(x_smooth, torch.tensor([[[[1], [-2], [1]]]], dtype=torch.float32), padding=(1,0))
        Ixy = F.conv2d(x_smooth, torch.tensor([[[[0.25, 0, -0.25],
                                                  [0,    0,  0],
                                                  [-0.25,0,  0.25]]]], dtype=torch.float32), padding=1)

        tmp     = torch.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2 + 1e-8)
        lambda1 = 0.5 * (Ixx + Iyy + tmp)
        lambda2 = 0.5 * (Ixx + Iyy - tmp)

        Rb = (lambda1 / (lambda2 + 1e-8))**2
        S2 = lambda1**2 + lambda2**2

        beta  = torch.clamp(self.beta,  0.1, 2.0)
        gamma = torch.clamp(self.gamma, 1e-4, 1.0)

        vesselness = torch.exp(-Rb / (2 * beta**2)) * \
                     (1 - torch.exp(-S2 / (2 * gamma**2)))

        vesselness = torch.where(lambda2 > 0,
                                 torch.zeros_like(vesselness),
                                 vesselness)
        return vesselness


# Load real image
img = load_image('tile.png')
x   = torch.tensor(img).unsqueeze(0).unsqueeze(0)

# Try multiple scales and combine
results = []
for scale in [1.0, 2.0, 3.0, 4.0]:
    model  = DiffFrangi(scale=scale, beta=0.5, gamma=0.01)
    out    = model(x)
    results.append(out)

# Take maximum response across scales
final = torch.stack(results).max(dim=0).values
final_np = final.squeeze().detach().numpy()

# Normalize for display
final_np = (final_np - final_np.min()) / (final_np.max() - final_np.min() + 1e-8)


# Keep only strong crack responses
final_np[final_np < 0.95] = 0

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Input Tile Image')
axes[1].imshow(final_np, cmap='hot')
axes[1].set_title('Frangi Crack Detection')
plt.tight_layout()
plt.savefig('frangi_result.png')
plt.show()
print("Done! Max response value:", final_np.max())