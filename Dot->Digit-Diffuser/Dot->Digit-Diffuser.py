#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dot-conditioned Diffusion Model (DDPM) on MNIST
(ONLY CHANGE: improved UNet architecture; rest of the pipeline kept the same.)

Task:
- For each sample, create a "dot image" same size as MNIST digit.
- Number of dots = class label (0..9), each dot is a single bright pixel.
- Condition the diffusion model ONLY on this dot image (no explicit label),
  and learn to generate the corresponding handwritten digit.

So the model learns:  (dot image)  ->  digit image
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import random

# ------------------------------------------------------------
# 0. Hyperparameters
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESOLUTION  = 14
IMG_SIZE    = RESOLUTION

BATCH_SIZE  = 128
LR          = 1e-4
EPOCHS      = 100
TIMESTEPS   = 1000

BASE_CH     = 32
UNET_DROPOUT = 0.05

LATENT_DIM  = 1024


# ------------------------------------------------------------
# 1. Beta / alpha schedule
# ------------------------------------------------------------
betas      = torch.linspace(1e-4, 0.02, TIMESTEPS)
alphas     = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

betas      = betas.to(device)
alphas     = alphas.to(device)
alphas_bar = alphas_bar.to(device) #last element should be small << 0.01-0.001

alphas_bar_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar) # some correction on beta (eq 7 of ddpm paper). Check other papers too.


# ============================================================
# 2. IMPROVED 1-level UNet Noise Predictor (Residual + Time-FiLM + Bottleneck Attention)
#    - Still conditioned only on (dot image + derived count_map) + timestep
#    - Keeps your FC latent bottleneck and returns (pred_noise, z) unchanged API
# ============================================================

def _gn_groups(ch: int, max_groups: int = 8) -> int:
    for g in reversed(range(1, max_groups + 1)):
        if ch % g == 0:
            return g
    return 1


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps -> float
        t = t.float()
        half = self.dim // 2
        # frequencies
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class ResBlock(nn.Module):
    """
    Residual block with timestep FiLM injection:
      h = conv1(silu(norm1(x))) + proj_t(temb)
      h = conv2(drop(silu(norm2(h))))
      out = h + skip(x)
    """
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.temb_proj = nn.Linear(temb_dim, out_ch)

        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.drop  = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Simple multi-head self-attention over H×W at the bottleneck."""
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        assert ch % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

        self.norm = nn.GroupNorm(_gn_groups(ch), ch)
        self.qkv  = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # (B, heads, head_dim, HW)
        HW = H * W
        q = q.reshape(B, self.num_heads, self.head_dim, HW)
        k = k.reshape(B, self.num_heads, self.head_dim, HW)
        v = v.reshape(B, self.num_heads, self.head_dim, HW)

        # attention: (B, heads, HW, HW)
        attn = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)

        # out: (B, heads, head_dim, HW) -> (B, C, H, W)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class BetterUNet1Level(nn.Module):
    """
    1-level UNet with:
      - sinusoidal timestep embedding + MLP
      - residual blocks with time injection
      - attention at bottleneck
      - your FC latent bottleneck preserved

    forward(x, cond, t) -> (pred_noise, z)
    """
    def __init__(self,
                 img_channels=1,
                 cond_channels=1,
                 base_ch=32,
                 dropout=0.05,
                 img_resolution=28,
                 latent_dim=256):
        super().__init__()
        assert img_resolution % 2 == 0, "img_resolution must be even (one MaxPool2d(2))."

        C = base_ch
        self.img_resolution = img_resolution
        self.latent_dim = latent_dim

        # time embedding: sinusoidal -> MLP
        temb_dim = 4 * C
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(temb_dim),
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # input: x_t + dot + count_map
        in_ch_total = img_channels + cond_channels + 1
        self.in_conv = nn.Conv2d(in_ch_total, C, 3, padding=1)

        # encoder (full res)
        self.enc1 = ResBlock(C, C, temb_dim, dropout)
        self.enc2 = ResBlock(C, 2 * C, temb_dim, dropout)  # richer skip

        self.pool = nn.MaxPool2d(2)

        # bottleneck (half res)
        self.mid1 = ResBlock(2 * C, 2 * C, temb_dim, dropout)
        self.attn = AttentionBlock(2 * C, num_heads=4)
        self.mid2 = ResBlock(2 * C, 2 * C, temb_dim, dropout)

        # FC latent bottleneck (same idea as your original)
        self.spatial_h = img_resolution // 2
        self.spatial_w = img_resolution // 2
        self.bottleneck_in = 2 * C * self.spatial_h * self.spatial_w
        self.fc_enc = nn.Linear(self.bottleneck_in, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.bottleneck_in)
        self.fc_drop = nn.Dropout(dropout)

        # decoder
        self.up = nn.ConvTranspose2d(2 * C, 2 * C, kernel_size=2, stride=2)
        self.dec1 = ResBlock(4 * C, C, temb_dim, dropout)
        self.dec2 = ResBlock(C, C, temb_dim, dropout)

        self.out_conv = nn.Conv2d(C, 1, 1)

    def forward(self, x, cond, t):
        B, _, H, W = x.shape

        # derived count_map from dot image (kept identical in spirit)
        cond_pos = (cond + 1.0) / 2.0
        count = cond_pos.sum(dim=[2, 3], keepdim=True)      # (B,1,1,1)
        count_map = count.expand(-1, 1, H, W)               # (B,1,H,W)

        temb = self.t_embed(t)                              # (B, temb_dim)

        x_in = torch.cat([x, cond, count_map], dim=1)       # (B,3,H,W)
        h = self.in_conv(x_in)

        # encoder
        h = self.enc1(h, temb)
        skip = self.enc2(h, temb)                           # (B,2C,H,W)
        h = self.pool(skip)                                 # (B,2C,H/2,W/2)

        # bottleneck
        h = self.mid1(h, temb)
        h = self.attn(h)
        h = self.mid2(h, temb)

        # FC latent bottleneck
        b = h.reshape(B, -1)                                # (B, bottleneck_in)
        z = self.fc_enc(b)                                  # (B, latent_dim)
        z_act = self.fc_drop(F.silu(z))
        b_dec = F.silu(self.fc_dec(z_act)).reshape(
            B, skip.shape[1], self.spatial_h, self.spatial_w
        )

        # small residual connection for stability
        h = h + b_dec

        # decoder
        h = self.up(h)                                      # (B,2C,H,W)
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")

        h = torch.cat([h, skip], dim=1)                     # (B,4C,H,W)
        h = self.dec1(h, temb)
        h = self.dec2(h, temb)

        out = self.out_conv(h)                              # (B,1,H,W)
        return out, z


model = BetterUNet1Level(
    img_channels=1,
    cond_channels=1,
    base_ch=BASE_CH,
    dropout=UNET_DROPOUT,
    img_resolution=RESOLUTION,
    latent_dim=LATENT_DIM
).to(device)

model.train()


# ------------------------------------------------------------
# 3. Generate dot images (single-pixel dots)
# ------------------------------------------------------------
def generate_dot_image(label: int, size: int, *args, **kwargs) -> torch.Tensor:
    num_dots = int(label)
    img = torch.zeros(1, size, size)

    if num_dots == 0:
        return img * 2.0 - 1.0

    positions = set()
    for _ in range(num_dots):
        while True:
            cx = np.random.randint(0, size)
            cy = np.random.randint(0, size)
            if (cx, cy) not in positions:
                positions.add((cx, cy))
                img[0, cy, cx] = 1.0
                break

    return img * 2.0 - 1.0


# ------------------------------------------------------------
# 4. Dataset: MNIST + dot images
# ------------------------------------------------------------
transform = T.Compose([
    T.Resize((RESOLUTION, RESOLUTION)),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2 - 1),
])


class DotMNIST(Dataset):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__()
        self.mnist = torchvision.datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )
        self.img_size = RESOLUTION

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        x_digit, y = self.mnist[idx]
        dot_img = generate_dot_image(y, self.img_size)
        return x_digit, dot_img, y


dataset = DotMNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ------------------------------------------------------------
# 5. Forward diffusion on the digit image only
# ------------------------------------------------------------
def sample_xt(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab  = torch.sqrt(alphas_bar[t]).view(-1, 1, 1, 1)
    sqrt_mab = torch.sqrt(1.0 - alphas_bar[t]).view(-1, 1, 1, 1)

    xt = sqrt_ab * x0 + sqrt_mab * noise
    return xt, noise


# ------------------------------------------------------------
# 6. Training loop
# ------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

print("Training model...\n")

for epoch in range(EPOCHS):
    for x0, dot_img, _y in dataloader:
        x0      = x0.to(device)
        dot_img = dot_img.to(device)

        t = torch.randint(0, TIMESTEPS, (x0.size(0),), device=device)
        xt, noise = sample_xt(x0, t)

        pred, z = model(xt, dot_img, t)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {loss.item():.4f}")

model.eval()
print("Finished training. Switched model to eval() mode for sampling.")


# ------------------------------------------------------------
# 7. Reverse Sampling (single image, given dot image)
# ------------------------------------------------------------
@torch.no_grad()
def sample_reverse(num_dots: int):
    xt = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    dot = generate_dot_image(num_dots, IMG_SIZE).unsqueeze(0).to(device)

    last_z = None
    for t in reversed(range(TIMESTEPS)):
        ts = torch.tensor([t], device=device)
        eps, z = model(xt, dot, ts)
        last_z = z

        alpha     = alphas[t]
        alpha_bar = alphas_bar[t]
        beta      = betas[t]

        mean = (1.0 / torch.sqrt(alpha)) * (
            xt - (beta / torch.sqrt(1.0 - alpha_bar)) * eps
        )

        # kept identical to your original script
        if t > 5:
            xt = mean + torch.sqrt(posterior_var[t]) * torch.randn_like(xt)
        else:
            xt = mean

    return xt.cpu(), dot.cpu(), last_z.cpu()


# ------------------------------------------------------------
# 8. Visualize 20 dot → digit pairs
# ------------------------------------------------------------
print("\nVisualizing dot–digit pairs (no trajectories)...")

NUM_PAIRS = 20
dot_counts = list(range(10)) * 2
random.shuffle(dot_counts)

fig, axes = plt.subplots(2, NUM_PAIRS, figsize=(16, 9))
fig.subplots_adjust(wspace=0.05, hspace=0.15)

all_latents = []

for i, n_dots in enumerate(dot_counts):
    digit_img, dot_img, z = sample_reverse(n_dots)
    all_latents.append(z.numpy())

    dot_ax = axes[0, i]
    dot_ax.imshow(dot_img[0, 0].clamp(-1, 1), cmap="gray")
    dot_ax.set_xticks([]); dot_ax.set_yticks([]); dot_ax.set_frame_on(False)
    dot_ax.set_title(f"n={n_dots}", fontsize=9, pad=4)

    digit_ax = axes[1, i]
    digit_ax.imshow(digit_img[0, 0].clamp(-1, 1), cmap="gray")
    digit_ax.set_xticks([]); digit_ax.set_yticks([]); digit_ax.set_frame_on(False)

    if i == 0:
        dot_ax.set_ylabel("Dots", fontsize=11)
        digit_ax.set_ylabel("Digit", fontsize=11)

fig.suptitle(
    f"Dot-conditioned generation (dot image only → handwritten digit)\nLatent dim = {LATENT_DIM}",
    fontsize=16, y=0.98
)
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.show()

plt.savefig("./DotDigit.png")
