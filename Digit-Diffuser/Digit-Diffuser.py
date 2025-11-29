#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-Conditional Diffusion Model (DDPM) on MNIST
WITH:
• Simple 1-level UNet noise predictor (dimensionally safe)
• dropout inside UNet blocks (small)
• weight decay (L2)
• gradient clipping
• arbitrary image resolution (here: 14x14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 0. Hyperparameters
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESOLUTION  = 14        # new MNIST resolution (native 28)
IMG_SIZE    = RESOLUTION

BATCH_SIZE  = 128
LR          = 1e-4
EPOCHS      = 100
TIMESTEPS   = 1000
NUM_CLASSES = 10

BASE_CH       = 32      
UNET_DROPOUT  = 0.05    


# ------------------------------------------------------------
# 1. Beta / alpha schedule
# ------------------------------------------------------------
betas      = torch.linspace(1e-4, 0.02, TIMESTEPS)
alphas     = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

betas      = betas.to(device)
alphas     = alphas.to(device)
alphas_bar = alphas_bar.to(device)

alphas_bar_prev = torch.cat(
    [torch.tensor([1.0], device=device), alphas_bar[:-1]]
)

posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)


# ============================================================
# 2. Simple 1-level UNet Noise Predictor (with dropout)
# ============================================================

class ConvBlock(nn.Module):
    """Conv → GN → SiLU → Dropout → Conv → GN → SiLU."""
    def __init__(self, in_ch, out_ch, dropout=0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """
    1-level UNet:

    Channels:
        (img+emb) → C
        C → 2C (down1) → bottleneck (2C) → up → concat (2C+2C=4C) → C → 1
    """
    def __init__(self, img_channels=1, base_ch=32, num_classes=10, dropout=0.05):
        super().__init__()

        C = base_ch

        # time + label embedding (richer embedding: 2C)
        self.emb_dim   = 2 * C
        self.time_emb  = nn.Embedding(TIMESTEPS, self.emb_dim)
        self.label_emb = nn.Embedding(num_classes, self.emb_dim)

        # Encoder
        self.in_conv = ConvBlock(img_channels + self.emb_dim, C, dropout=dropout)  # (1 + emb_dim) -> C
        self.down1   = ConvBlock(C, 2 * C, dropout=dropout)                        # C -> 2C
        self.pool1   = nn.MaxPool2d(2)                                             # H,W -> H/2,W/2

        # Bottleneck
        self.bot = ConvBlock(2 * C, 2 * C, dropout=dropout)                        # 2C -> 2C

        # Decoder
        self.up1   = nn.ConvTranspose2d(2 * C, 2 * C, kernel_size=2, stride=2)     # 2C -> 2C
        self.dec1  = ConvBlock(4 * C, C, dropout=dropout)                          # concat(2C,2C)=4C -> C

        # Output
        self.out_conv = nn.Conv2d(C, 1, kernel_size=1)                             # C -> 1

    def forward(self, x, t, y):
        B, _, H, W = x.shape

        # ---- Class + time embedding as extra channels ----
        emb = self.time_emb(t) + self.label_emb(y)      # (B, emb_dim)
        emb = emb[:, :, None, None].repeat(1, 1, H, W)  # (B, emb_dim, H, W)

        x = torch.cat([x, emb], dim=1)                  # (B, 1+emb_dim, H, W)

        # ------------ Encoder ------------
        x0 = self.in_conv(x)                            # (B, C, H, W)

        d1 = self.down1(x0)                             # (B, 2C, H, W)
        p1 = self.pool1(d1)                             # (B, 2C, H/2, W/2)

        # ------------ Bottleneck ------------
        b  = self.bot(p1)                               # (B, 2C, H/2, W/2)

        # ------------ Decoder ------------
        u1 = self.up1(b)                                # (B, 2C, H, W)
        if u1.shape[-2:] != d1.shape[-2:]:
            u1 = F.interpolate(u1, size=d1.shape[-2:], mode="nearest")

        u1_cat = torch.cat([u1, d1], dim=1)             # (B, 4C, H, W)
        u1_dec = self.dec1(u1_cat)                      # (B, C, H, W)

        out = self.out_conv(u1_dec)                     # (B, 1, H, W)
        return out


model = SimpleUNet(
    img_channels=1,
    base_ch=BASE_CH,
    num_classes=NUM_CLASSES,
    dropout=UNET_DROPOUT
).to(device)

model.train()


# ------------------------------------------------------------
# 3. Load + Resize MNIST (normalized to [-1, 1])
# ------------------------------------------------------------
transform = T.Compose([
    T.Resize((RESOLUTION, RESOLUTION)),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2 - 1),
])

dataset = torchvision.datasets.MNIST(
    root="./data", download=True, transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# ------------------------------------------------------------
# 4. Forward diffusion
# ------------------------------------------------------------
def sample_xt(x0, t, noise=None):
    """
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab  = torch.sqrt(alphas_bar[t]).view(-1, 1, 1, 1)
    sqrt_mab = torch.sqrt(1.0 - alphas_bar[t]).view(-1, 1, 1, 1)

    xt = sqrt_ab * x0 + sqrt_mab * noise
    return xt, noise


# ------------------------------------------------------------
# 5. Training loop WITH REGULARIZATION
# ------------------------------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,       # L2 regularization
)

print("Training model...\n")

for epoch in range(EPOCHS):
    for x0, y in dataloader:
        x0, y = x0.to(device), y.to(device)

        # random timestep per sample
        t = torch.randint(0, TIMESTEPS, (x0.size(0),), device=device)
        xt, noise = sample_xt(x0, t)

        pred = model(xt, t, y)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {loss.item():.4f}")

# IMPORTANT: turn off dropout & keep BN/GN in eval mode for sampling
model.eval()
print("Finished training. Switched model to eval() mode for sampling.")


# ------------------------------------------------------------
# 6. Reverse Sampling (single image)
# ------------------------------------------------------------
@torch.no_grad()
def sample_reverse(digit_label: int):
    """
    Generate a single image conditioned on digit_label (0–9).
    """
    xt = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    y  = torch.tensor([digit_label], device=device)

    for t in reversed(range(TIMESTEPS)):
        ts  = torch.tensor([t], device=device)
        eps = model(xt, ts, y)

        alpha     = alphas[t]
        alpha_bar = alphas_bar[t]
        beta      = betas[t]

        mean = (1.0 / torch.sqrt(alpha)) * (
            xt - (beta / torch.sqrt(1.0 - alpha_bar)) * eps
        )

        # small amount of noise except at very small t
        if t > 5:
            xt = mean + torch.sqrt(posterior_var[t]) * torch.randn_like(xt)
        else:
            xt = mean

    return xt.cpu()


# ------------------------------------------------------------
# 7. Trajectory Sampling (x_T → x_0)
# ------------------------------------------------------------
@torch.no_grad()
def sample_trajectory(digit_label: int):
    """
    Return list [x_T, ..., x_0] for a given digit_label.
    """
    xt = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    y  = torch.tensor([digit_label], device=device)

    traj = [xt.cpu()]

    for t in reversed(range(TIMESTEPS)):
        ts  = torch.tensor([t], device=device)
        eps = model(xt, ts, y)

        alpha     = alphas[t]
        alpha_bar = alphas_bar[t]
        beta      = betas[t]

        mean = (1.0 / torch.sqrt(alpha)) * (
            xt - (beta / torch.sqrt(1.0 - alpha_bar)) * eps
        )

        if t > 5:
            xt = mean + torch.sqrt(posterior_var[t]) * torch.randn_like(xt)
        else:
            xt = mean

        traj.append(xt.cpu())

    return traj


# ------------------------------------------------------------
# 8. Generate and Plot Trajectories
# ------------------------------------------------------------
print("\nGenerating conditional denoising trajectories...")

NUM_EXAMPLES = 6
NUM_FRAMES   = 8
target_digits = [0, 1, 2, 3, 4, 5]

plt.figure(figsize=(2 * NUM_FRAMES, 2 * NUM_EXAMPLES))

for row_idx, digit in enumerate(target_digits):
    traj = sample_trajectory(digit)
    idxs = np.linspace(0, len(traj) - 1, NUM_FRAMES, dtype=int)
    frames = [traj[i] for i in idxs]

    for col_idx, img in enumerate(frames):
        plt.subplot(NUM_EXAMPLES, NUM_FRAMES,
                    row_idx * NUM_FRAMES + col_idx + 1)
        # clamp to [-1,1] and show as grayscale
        plt.imshow(img[0, 0].clamp(-1, 1), cmap="gray")
        plt.axis("off")
        if col_idx == 0:
            plt.ylabel(f"{digit}", rotation=0, labelpad=40, fontsize=12)

plt.suptitle(
    f"Digit generation >>>>>",
    fontsize=18,y=0.99
)
plt.tight_layout()
plt.show()
