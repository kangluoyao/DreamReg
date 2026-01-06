import os
import sys
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ======= 根据你自己项目结构修改这些 import =======
from models.baseline_wm_v2 import EUReg_WM_Belief      # e.g. from models.wm import EUReg_WM_Belief
from data.datasets import TrainDataset, TestDataset
import losses
import utils
# from logger import Logger                 # 或者 from utils import Logger
# from utils import normalize_vec
# ================================================


@torch.no_grad()
def rollout_one_sample(model, vol, frame, dof,
                       device,
                       wm_steps=20,
                       step_scale=1.0,
                       noise_std=0.0):
    """
    对一个样本做 world model rollout，返回：
      - T_seq:   (T,6)
      - dT_seq:  (T,6)
      - reward_seq: (T,2)
      - frames:  (T,H,W)  每一步的 cur slice
    """
    model.eval()

    vol   = vol.to(device)
    frame = frame.to(device)
    dof   = dof.to(device)

    B = vol.size(0)
    assert B == 1, "rollout_one_sample 默认 batch_size=1 来可视化"

    T0 = torch.zeros(B, 6, device=device)

    # 对测试可视化来说，不需要 dir_info，关掉就好
    T_final, frame_last, traj = model(
        vol, frame, dof, T0,
        steps=wm_steps,
        step_scale=step_scale,
        noise_std=noise_std,
        return_all=True,
        return_dir_info=False
    )

    T_list  = []
    dT_list = []
    R_list  = []
    F_list  = []

    for (T_t, frame_t, r_pred, sl_dec, dT) in traj:
        # shapes:
        #   T_t:     (B,6)
        #   dT:      (B,6)
        #   r_pred:  (B,2)
        #   frame_t: (B,1,H,W)
        T_list.append(T_t[0].detach().cpu().numpy())        # (6,)
        dT_list.append(dT[0].detach().cpu().numpy())        # (6,)
        R_list.append(r_pred[0].detach().cpu().numpy())     # (2,)
        F_list.append(frame_t[0, 0].detach().cpu().numpy()) # (H,W)

    T_seq      = np.stack(T_list, axis=0)   # (T,6)
    dT_seq     = np.stack(dT_list, axis=0)  # (T,6)
    reward_seq = np.stack(R_list, axis=0)   # (T,2)
    frames     = np.stack(F_list, axis=0)   # (T,H,W)

    return T_seq, dT_seq, reward_seq, frames


def save_reward_plot(reward_seq, out_path):
    """
    reward_seq: (T,2) → 画两条 reward 曲线
    """
    steps = np.arange(len(reward_seq))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.plot(steps, reward_seq[:, 0], label="reward[0]")
    if reward_seq.shape[1] > 1:
        ax.plot(steps, reward_seq[:, 1], label="reward[1]")

    ax.set_xlabel("step")
    ax.set_title("Reward over rollout steps")
    ax.grid(True)
    ax.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Vis] saved reward plot to {out_path}")


def save_frames_grid(frames, out_path, max_cols=6):
    """
    frames: (T,H,W) → 拼成一张大图，观察 rollout 过程中 image 变化。
    """
    T, H, W = frames.shape
    max_cols = min(max_cols, T)
    n_rows = int(np.ceil(T / max_cols))

    fig, axes = plt.subplots(n_rows, max_cols,
                             figsize=(2*max_cols, 2*n_rows))
    axes = np.array(axes).reshape(n_rows, max_cols)

    for i in range(n_rows * max_cols):
        r = i // max_cols
        c = i % max_cols
        ax = axes[r, c]
        ax.axis("off")

        if i < T:
            img = frames[i]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_vis = (img - img_min) / (img_max - img_min)
            else:
                img_vis = np.zeros_like(img)
            ax.imshow(img_vis, cmap="gray")
            ax.set_title(f"step {i}", fontsize=8)
        else:
            ax.imshow(np.zeros((H, W)), cmap="gray")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Vis] saved frames grid to {out_path}")


def save_frames_gif(frames, out_path, fps=3):
    """
    frames: (T,H,W) → 生成一个 GIF，动态展示图像随 step 的变化。
    """
    if not HAS_IMAGEIO:
        print("[Vis] imageio not installed, skip GIF.")
        return

    imgs = []
    for img in frames:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)
        imgs.append((img_norm * 255).astype(np.uint8))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, imgs, fps=fps)
    print(f"[Vis] saved GIF to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_steps',   type=int,   default=20)
    parser.add_argument('--step_scale', type=float, default=1.0)
    parser.add_argument('--noise_std',  type=float, default=0.0)
    parser.add_argument('--batch_size', type=int,   default=1,
                        help="只用来加载 test set，可视化默认只看每个 batch 的第一个样本")
    parser.add_argument('--n_cases',    type=int,   default=5,
                        help="在测试集上可视化多少个样本的 rollout")
    parser.add_argument('--save_dir',   type=str,   default='wm_base_1',
                        help="和你训练时的 save_dir 对齐，用来放可视化结果")
    parser.add_argument('--ckpt',       type=str,   default='',
                        help="(可选) 训练好的模型权重路径 .pth 或 .pth.tar")
    args = parser.parse_args()

    print("=== World Model Rollout Visualization on TEST SET ===")
    print("wm_steps:", args.wm_steps,
          "step_scale:", args.step_scale,
          "noise_std:", args.noise_std)

    # 和你训练脚本里一致的路径 / 形状
    test_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/testing/'
    img_size = (32, 192, 192)
    slice_size = (192, 192)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 构建模型 ----------
    model = EUReg_WM_Belief(img_size).to(device)

    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"[Load] loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        # 你训练时保存的 key 可能不同，注意改这里：
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    else:
        print("[Warn] no checkpoint loaded, using randomly initialized model.")

    # ---------- 构建 TEST loader ----------
    test_set = TestDataset(glob.glob(test_dir + '*.pkl'))
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4)

    out_root = os.path.join('experiments/CAMUS2', args.save_dir, 'test_rollout_vis')
    os.makedirs(out_root, exist_ok=True)

    # ---------- 在测试集上滚动若干 case ----------
    n_vis = 0
    for idx, data in enumerate(test_loader):
        if n_vis >= args.n_cases:
            break

        vol   = data[0]   # (B,1,32,192,192)
        frame = data[1]   # (B,1,192,192)
        dof   = data[2]   # (B,6)

        print(f"\n[Case {idx}] running rollout ...")
        T_seq, dT_seq, reward_seq, frames = rollout_one_sample(
            model, vol, frame, dof,
            device=device,
            wm_steps=args.wm_steps,
            step_scale=args.step_scale,
            noise_std=args.noise_std
        )

        case_prefix = os.path.join(out_root, f'case_{idx:03d}')
        reward_png  = case_prefix + '_reward.png'
        frames_grid = case_prefix + '_frames_grid.png'
        frames_gif  = case_prefix + '_frames.gif'

        save_reward_plot(reward_seq, reward_png)
        save_frames_grid(frames, frames_grid)
        save_frames_gif(frames, frames_gif, fps=3)

        n_vis += 1

    print("\n[Done] rollout visualization finished.")


if __name__ == "__main__":
    main()
