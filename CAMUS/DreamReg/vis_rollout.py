#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wm_rollout_demo.py

使用训练好的 EUReg_WM_Belief 权重，做一次 rollout 可视化：
- 生成 GIF: [目标 slice | 当前 env slice | world model 重建]
- 生成 reward 曲线 (trans, rot)
- 生成 NCC(goal, env slice) 曲线
"""

import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 服务器上无显示环境也能画图
import matplotlib.pyplot as plt
import imageio

from models.baseline_wm_v3 import EUReg_WM_Belief 

from data.datasets import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import glob

# ============================================================
# 2. 通用小工具函数
# ============================================================

def ensure_dir_for(path: str):
    """确保 path 对应的目录存在。"""
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """
    img: (1,H,W) 或 (H,W) torch tensor
    返回归一化到 [0,1] 的 numpy (H,W)
    """
    if img.dim() == 3:
        img = img[0]  # 去掉 channel 维
    img = img.detach().cpu().float()
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img.numpy()


def compute_ncc(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    计算两个灰度图之间的 NCC。
    x, y: (1,H,W) 或 (H,W) 的 tensor
    返回 scalar float
    """
    if x.dim() == 3:
        x = x[0]
    if y.dim() == 3:
        y = y[0]
    x = x.detach().cpu().float()
    y = y.detach().cpu().float()

    x = x - x.mean()
    y = y - y.mean()
    vx = (x * x).mean()
    vy = (y * y).mean()
    if vx < eps or vy < eps:
        return 0.0
    ncc = (x * y).mean() / torch.sqrt(vx * vy + eps)
    return float(ncc.item())


# ============================================================
# 3. rollout 可视化（核心）
# ============================================================

def visualize_rollout(
    model: torch.nn.Module,
    vol: torch.Tensor,          # (B,1,D,H,W)
    goal_sl: torch.Tensor,      # (B,1,H,W)
    T0: torch.Tensor,           # (B,6)
    steps: int,
    step_scale: float,
    gif_path: str,
    reward_curve_path: str,
    ncc_curve_path: str,
    sample_idx: int = 0,
    fps: int = 2,
    figsize=(9, 3),
):
    """
    对一个 batch 的第 sample_idx 个样本做 rollout，可视化为：
      - GIF: [target | env slice | WM recon]
      - reward 曲线（两维）
      - NCC(goal, env slice) 曲线

    依赖模型的 forward 接口：
      T_final, cur_final, traj = model(vol, goal_sl, T0,
                                       steps=steps, step_scale=step_scale,
                                       return_all=True)
    其中 traj 是 list，元素为：
      (T_t, cur_t, dT_t, r_pred_t, sl_pred_t)
    """

    ensure_dir_for(gif_path)
    ensure_dir_for(reward_curve_path)
    ensure_dir_for(ncc_curve_path)

    model.eval()
    with torch.no_grad():
        T_final, cur_final, traj = model(
            vol, goal_sl, T0,
            steps=steps,
            step_scale=step_scale,
            return_all=True
        )

    # 选一个样本
    idx = sample_idx
    target_np = tensor_to_numpy(goal_sl[idx])  # (H,W)

    frames = []
    reward_trans_list = []
    reward_rot_list = []
    ncc_list = []

    for t, (T_t, cur_t, dT_t, r_pred_t, sl_pred_t) in enumerate(traj):
        # 当前真实 slice & 重建 slice
        cur_np   = tensor_to_numpy(cur_t[idx])
        recon_np = tensor_to_numpy(sl_pred_t[idx])

        # 记录 reward（假设 r_pred = [cos_trans, cos_rot]）
        r = r_pred_t[idx].detach().cpu().numpy()
        reward_trans_list.append(float(r[0]))
        reward_rot_list.append(float(r[1]))

        # 记录 NCC(goal, 当前真实 slice)
        ncc_t = compute_ncc(goal_sl[idx], cur_t[idx])
        ncc_list.append(ncc_t)

        # 画三列图：target / env slice / WM recon
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        ax1, ax2, ax3 = axes

        ax1.imshow(target_np, cmap='gray')
        ax1.set_title('Target (goal)')
        ax1.axis('off')

        ax2.imshow(cur_np, cmap='gray')
        ax2.set_title(f'Env slice t={t+1}')
        ax2.axis('off')

        ax3.imshow(recon_np, cmap='gray')
        ax3.set_title('WM recon')
        ax3.axis('off')

        fig.suptitle(
            f"Step {t+1} | "
            f"Reward: ({reward_trans_list[-1]:.3f}, {reward_rot_list[-1]:.3f}) | "
            f"NCC: {ncc_list[-1]:.3f}",
            fontsize=10
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        # 保存当前帧到 numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(img)
        plt.close(fig)

    # 保存 GIF
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[wm_rollout_demo] GIF saved to: {gif_path}")

    # 画 reward 曲线
    steps_axis = np.arange(1, len(reward_trans_list) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(steps_axis, reward_trans_list, marker='o', label='reward_trans')
    plt.plot(steps_axis, reward_rot_list, marker='o', label='reward_rot')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward over rollout')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_curve_path)
    plt.close()
    print(f"[wm_rollout_demo] Reward curve saved to: {reward_curve_path}")

    # 画 NCC 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(steps_axis, ncc_list, marker='o')
    plt.xlabel('Step')
    plt.ylabel('NCC(goal, env slice)')
    plt.title('NCC over rollout')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(ncc_curve_path)
    plt.close()
    print(f"[wm_rollout_demo] NCC curve saved to: {ncc_curve_path}")


# ============================================================
# 4. 主函数：加载模型、权重，取一批数据做可视化
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="World Model rollout visualization demo"
    )
    parser.add_argument("--ckpt", type=str, default="/Media_HDD/lykang/EUReg/CAMUS_diff/EUReg10/experiments/CAMUS2/wm_v3_noise_2/best_model.pth.tar",
                        help="Path to trained checkpoint (e.g., best.pth)")
    parser.add_argument("--save_dir", type=str, default="./wm_vis",
                        help="Directory to save GIF and curves")
    parser.add_argument("--wm_steps", type=int, default=7,
                        help="Rollout steps (must match or be <=训练时)")
    parser.add_argument("--step_scale", type=float, default=1.0,
                        help="Step scale for ΔT")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for test loader")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Num workers for DataLoader")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Which sample in the batch to visualize")
    # 你可以加一个 --data_root / --split 等参数，用来构建 test_loader
    return parser.parse_args()


def build_test_loader(args) -> torch.utils.data.DataLoader:
    
    test_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/testing/'
    test_set  = TestDataset(glob.glob(test_dir + '*.pkl'))
    test_loader  = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=8)

    return test_loader


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[wm_rollout_demo] Using device: {device}")

    # 1) 构建 test_loader（这里需要你根据项目改 build_test_loader）
    test_loader = build_test_loader(args)

    # 2) 取一批数据
    data_iter = iter(test_loader)
    data = next(data_iter)

    vol   = data[0].to(device)  # (B,1,D,H,W)
    frame = data[1].to(device)  # (B,1,H,W)
    dof   = data[2].to(device)  # (B,6)

    B = vol.size(0)
    T0 = torch.zeros(B, 6, device=device, dtype=vol.dtype)

    # 3) 根据这批数据的 vol.shape 创建模型
    #    注意：这里假设你的 EUReg_WM_Belief(vol_shape=vol.shape)
    #    如果你定义不一样，请按你自己的方式初始化。
    model = EUReg_WM_Belief((32, 192, 192)).to(device)

    # 4) 加载 checkpoint
    print(f"[wm_rollout_demo] Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("[wm_rollout_demo] Checkpoint loaded.")

    # 5) 准备输出路径
    ensure_dir_for(os.path.join(args.save_dir, "dummy"))
    gif_path          = os.path.join(args.save_dir, "rollout.gif")
    reward_curve_path = os.path.join(args.save_dir, "reward_curve.png")
    ncc_curve_path    = os.path.join(args.save_dir, "ncc_curve.png")

    # 6) 做可视化
    visualize_rollout(
        model,
        vol=vol,
        goal_sl=frame,
        T0=T0,
        steps=args.wm_steps,
        step_scale=args.step_scale,
        gif_path=gif_path,
        reward_curve_path=reward_curve_path,
        ncc_curve_path=ncc_curve_path,
        sample_idx=args.sample_idx,
        fps=2,
    )
    print("[wm_rollout_demo] Done.")


if __name__ == "__main__":
    main()
