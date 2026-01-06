#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from models.baseline_wm_v3 import Encoder2D, Dec192
from data.datasets import TrainDataset
from torch.utils.data import DataLoader

from losses import SSIMLoss, LNCCLoss

import matplotlib.pyplot as plt

import glob

# ❸ SSIM loss
# from your_loss_file import SSIMLoss


################################################################################
# SliceVAE (Encoder2D + KL bottleneck + Dec192)
################################################################################

class SliceVAE(nn.Module):
    def __init__(self, in_channel=1, first_channel=8, z_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.encoder2d = Encoder2D(in_channel, first_channel)
        self.dec192 = Dec192(z_dim)

        c = first_channel * 8
        self.pool = nn.AdaptiveMaxPool2d((6, 6))

        c_backbone = first_channel * 8
        latent_ch_factor = 1
        
        latent_ch = int(c_backbone * latent_ch_factor)

        self.down_conv = nn.Sequential(
            # 24x24 → 12x12
            nn.Conv2d(c_backbone, latent_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 12x12 → 6x6
            nn.Conv2d(latent_ch, latent_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # 线性层输出 2*z_dim，用来拆分为 μ 和 logσ²
        self.fc_mu_logvar = nn.Linear(latent_ch * 6 * 6, 2 * z_dim)

    def encode(self, x):
        f = self.encoder2d(x)
        f = self.down_conv(f)
        # f = self.pool(f)
        B = f.size(0)
        f = f.view(B, -1)

        stats = self.fc_mu_logvar(f)
        mu, logvar = stats.chunk(2, dim=-1)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z

    def decode(self, z):
        return self.dec192(z)

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar, z

################################################################################
# 可视化：保存若干样本的 input vs recon
################################################################################

def _to_np_img(t):
    """
    t: (1,H,W) 或 (B,1,H,W)
    """
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        t = t[0]
    t = t.detach().cpu().float()
    t = t - t.min()
    if t.max() > 0:
        t = t / t.max()
    return t.numpy()


def save_recon_vis(model, loader, device, save_path, num_samples=4):
    """
    从 loader 里取一个 batch，画 num_samples 个样本的：
    上排：原图；下排：重建
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with torch.no_grad():
        data = next(iter(loader))
        frame = data[1].to(device)   # (B,1,192,192)
        recon, mu, logvar, z = model(frame)

    B = min(num_samples, frame.size(0))
    plt.figure(figsize=(3 * B, 6))

    for i in range(B):
        img_in  = _to_np_img(frame[i])
        img_rec = _to_np_img(recon[i])

        # 原图
        ax1 = plt.subplot(2, B, i + 1)
        ax1.imshow(img_in, cmap='gray')
        ax1.set_title(f"Input {i}")
        ax1.axis("off")

        # 重建
        ax2 = plt.subplot(2, B, B + i + 1)
        ax2.imshow(img_rec, cmap='gray')
        ax2.set_title(f"Recon {i}")
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VAE] Saved recon visualization to {save_path}")


################################################################################
# KL loss
################################################################################

def kl_normal(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


################################################################################
# Training function
################################################################################

def train_slice_vae(
    train_dir,
    batch_size=16,
    num_epochs=50,
    lr=1e-3,
    beta_kl=1e-3,
    device="cuda",
    vis_interval=5,
    vis_dir="./vae_vis"
):
    device = torch.device(device)

    # TODO：改成你的真实 CAMUS Dataset 类
    train_set = TrainDataset(glob.glob(train_dir + '*.pkl'), (192,192), [10]*6)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = SliceVAE(in_channel=1, first_channel=8, z_dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # TODO: 改成你的 SSIMLoss
    SSIM = SSIMLoss()
    L1 = nn.L1Loss()

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        recon_sum = 0
        kl_sum = 0
        n = 0

        for vol, frame, dof in train_loader:
            frame = frame.to(device)

            recon, mu, logvar, z = model(frame)

            recon_loss = L1(recon, frame) + SSIM(recon, frame)
            # kl = kl_normal(mu, logvar)

            loss = recon_loss
            # loss = recon_loss + beta_kl * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            B = frame.size(0)
            loss_sum += loss.item() * B
            recon_sum += recon_loss.item() * B
            # kl_sum += kl.item() * B
            n += B

        print(f"[VAE] Epoch {epoch} | "
              f"Loss {loss_sum/n:.4f} | "
              f"Recon {recon_sum/n:.4f} | ")
            #   f"KL {kl_sum/n:.4f}")

        # 每 vis_interval 个 epoch 保存一次重建可视化
        if (epoch + 1) % vis_interval == 0 or epoch == 0:
            vis_path = os.path.join(vis_dir, f"recon_epoch_{epoch+1:03d}.png")
            save_recon_vis(model, train_loader, device, vis_path, num_samples=4)

    return model


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    train_dir = "/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/training/"

    model = train_slice_vae(
        train_dir=train_dir,
        batch_size=32,
        num_epochs=1000,
        lr=1e-3,
        beta_kl=1e-1,
        device="cuda",
        vis_interval=5,
        vis_dir="./vae_vis"
    )

    os.makedirs("./vae_ckpt", exist_ok=True)
    ckpt_path = "./vae_ckpt/slice_vae_pretrained.pth"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved to {ckpt_path}")
