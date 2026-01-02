#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dot-conditioned Diffusion Transformer (DiT-style) on MNIST (14x14)

Task:
- Create a dot image (same size as digit).
- Number of dots = label (0..9), each dot is a single bright pixel.
- Condition the diffusion model ONLY on dot image (no explicit label).
- Learn mapping: (dot image) -> digit image, via DDPM.

This script is DiT-inspired:
- Patchify the input (x_t + cond [+ count_map]) into tokens
- Transformer backbone with timestep-conditioned AdaLN modulation
- Unpatchify into predicted noise epsilon
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 0. Hyperparameters
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESOLUTION   = 14
IMG_SIZE     = RESOLUTION

BATCH_SIZE   = 128
LR           = 1e-4
EPOCHS       = 100
TIMESTEPS    = 1000

# DiT-ish config
PATCH_SIZE   = 2               # 14x14 -> 7x7 tokens = 49 tokens
EMBED_DIM    = 256
DEPTH        = 8
NUM_HEADS    = 8
MLP_RATIO    = 4.0
DROPOUT      = 0.0

# Conditioning channels
# x_t: 1 channel
# cond(dot): 1 channel
# count_map: 1 channel (derived from cond, kept like your UNet)
USE_COUNT_MAP = True
IN_CH = 1 + 1 + (1 if USE_COUNT_MAP else 0)
OUT_CH = 1  # predict noise for digit channel only


# ------------------------------------------------------------
# 1. Beta / alpha schedule
# ------------------------------------------------------------
betas      = torch.linspace(1e-4, 0.02, TIMESTEPS, device=device)
alphas     = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

alphas_bar_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)


# ------------------------------------------------------------
# 2. Utilities: timestep embedding
# ------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


# ------------------------------------------------------------
# 3. Patchify / unpatchify
# ------------------------------------------------------------
def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """
    x: (B, C, H, W)
    return: (B, N, patch*patch*C) where N=(H/patch)*(W/patch)
    """
    B, C, H, W = x.shape
    assert H % patch == 0 and W % patch == 0
    h = H // patch
    w = W // patch
    x = x.reshape(B, C, h, patch, w, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, patch, patch, C)
    x = x.reshape(B, h * w, patch * patch * C)
    return x


def unpatchify(tokens: torch.Tensor, patch: int, H: int, W: int, C: int) -> torch.Tensor:
    """
    tokens: (B, N, patch*patch*C)
    return: (B, C, H, W)
    """
    B, N, D = tokens.shape
    h = H // patch
    w = W // patch
    assert N == h * w
    tokens = tokens.reshape(B, h, w, patch, patch, C)
    tokens = tokens.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, patch, w, patch)
    x = tokens.reshape(B, C, H, W)
    return x


# ------------------------------------------------------------
# 4. DiT-style blocks (AdaLN modulation)
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """
    Transformer block with AdaLN modulation from timestep embedding.
    DiT idea: use t-embedding to produce shift/scale/gates for attention+MLP.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, mlp_ratio, drop)

        # produce modulation params from temb:
        # (shift1, scale1, gate1, shift2, scale2, gate2) each dim
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

        # "adaLN-Zero"-like init: start near identity / no conditioning effect
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        temb: (B, D)
        """
        B, N, D = x.shape
        params = self.ada(temb).view(B, 6, D)
        shift1, scale1, gate1, shift2, scale2, gate2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        # Attention
        h = self.norm1(x)
        h = h * (1 + scale1[:, None, :]) + shift1[:, None, :]
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1[:, None, :] * attn_out

        # MLP
        h = self.norm2(x)
        h = h * (1 + scale2[:, None, :]) + shift2[:, None, :]
        x = x + gate2[:, None, :] * self.mlp(h)
        return x


class DotDiT(nn.Module):
    """
    DiT-style diffusion model:
      input: x_t (digit) and cond (dot image) [+ count_map]
      patchify -> token embedding -> transformer blocks -> unpatchify
      output: predicted epsilon for digit channel
    """
    def __init__(self,
                 img_size: int = 14,
                 patch: int = 2,
                 in_ch: int = 3,
                 out_ch: int = 1,
                 dim: int = 256,
                 depth: int = 8,
                 heads: int = 8,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        assert img_size % patch == 0
        self.img_size = img_size
        self.patch = patch
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.num_patches = (img_size // patch) * (img_size // patch)
        self.patch_dim = patch * patch * in_ch

        # token embed
        self.x_embed = nn.Linear(self.patch_dim, dim)

        # learnable positional embedding (DiT uses fixed or learned; learned is fine here)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # timestep embed -> project to model dim
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # transformer backbone
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, mlp_ratio, drop) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim)

        # project tokens back to pixel space (noise prediction)
        self.out_proj = nn.Linear(dim, patch * patch * out_ch)

        # initialize out_proj near zero for stable early training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        """
        x_t: (B,1,H,W)
        cond: (B,1,H,W)
        t: (B,)
        returns: (eps_pred, z_dummy)
          - eps_pred: (B,1,H,W)
          - z_dummy:  a placeholder latent (keeps your prior API shape idea)
        """
        B, _, H, W = x_t.shape

        if USE_COUNT_MAP:
            cond_pos = (cond + 1.0) / 2.0
            count = cond_pos.sum(dim=[2, 3], keepdim=True)  # (B,1,1,1)
            count_map = count.expand(-1, 1, H, W)
            x_in = torch.cat([x_t, cond, count_map], dim=1)  # (B,3,H,W)
        else:
            x_in = torch.cat([x_t, cond], dim=1)            # (B,2,H,W)

        # patchify -> tokens
        tokens = patchify(x_in, self.patch)                 # (B,N,patch_dim)
        tokens = self.x_embed(tokens)                       # (B,N,D)
        tokens = tokens + self.pos_embed                    # (B,N,D)

        temb = self.t_embed(t)                              # (B,D)

        for blk in self.blocks:
            tokens = blk(tokens, temb)

        tokens = self.final_norm(tokens)                    # (B,N,D)

        out_tokens = self.out_proj(tokens)                  # (B,N,patch*patch*out_ch)
        eps = unpatchify(out_tokens, self.patch, H, W, self.out_ch)  # (B,1,H,W)

        # Keep an API-compatible "latent" if you want to log something:
        # Use temb as a compact representation (B,D).
        z_dummy = temb

        return eps, z_dummy


# ------------------------------------------------------------
# 5. Generate dot images (single-pixel dots)
# ------------------------------------------------------------
def generate_dot_image(label: int, size: int) -> torch.Tensor:
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
# 6. Dataset: MNIST + dot images
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
# 7. Forward diffusion on the digit image only
# ------------------------------------------------------------
def sample_xt(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab  = torch.sqrt(alphas_bar[t]).view(-1, 1, 1, 1)
    sqrt_mab = torch.sqrt(1.0 - alphas_bar[t]).view(-1, 1, 1, 1)

    xt = sqrt_ab * x0 + sqrt_mab * noise
    return xt, noise


# ------------------------------------------------------------
# 8. Train
# ------------------------------------------------------------
model = DotDiT(
    img_size=RESOLUTION,
    patch=PATCH_SIZE,
    in_ch=IN_CH,
    out_ch=OUT_CH,
    dim=EMBED_DIM,
    depth=DEPTH,
    heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    drop=DROPOUT
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print("Training DiT-style model...\n")
model.train()

for epoch in range(EPOCHS):
    for x0, dot_img, _y in dataloader:
        x0      = x0.to(device)
        dot_img = dot_img.to(device)

        t = torch.randint(0, TIMESTEPS, (x0.size(0),), device=device)
        xt, noise = sample_xt(x0, t)

        pred_eps, _z = model(xt, dot_img, t)
        loss = F.mse_loss(pred_eps, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {loss.item():.4f}")

model.eval()
print("Finished training. Switched model to eval() mode for sampling.")


# ------------------------------------------------------------
# 9. Reverse sampling (single image, given dot image)
# ------------------------------------------------------------
@torch.no_grad()
def sample_reverse(num_dots: int):
    xt = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)
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

        if t > 5:
            xt = mean + torch.sqrt(posterior_var[t]) * torch.randn_like(xt)
        else:
            xt = mean

    return xt.cpu(), dot.cpu(), last_z.cpu()


# ------------------------------------------------------------
# 10. Visualize 20 dot → digit pairs
# ------------------------------------------------------------
print("\nVisualizing dot–digit pairs (no trajectories)...")

NUM_PAIRS = 20
dot_counts = list(range(10)) * 2
random.shuffle(dot_counts)

fig, axes = plt.subplots(2, NUM_PAIRS, figsize=(16, 9))
fig.subplots_adjust(wspace=0.05, hspace=0.15)

for i, n_dots in enumerate(dot_counts):
    digit_img, dot_img, z = sample_reverse(n_dots)

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
    f"Dot-conditioned generation (DiT-style): dot image only → handwritten digit\n"
    f"patch={PATCH_SIZE}, dim={EMBED_DIM}, depth={DEPTH}, heads={NUM_HEADS}",
    fontsize=16, y=0.98
)
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.show()

plt.savefig("./DotDigit_DiT.png", dpi=200)
print("Saved: ./DotDigit_DiT.png")
