#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import pickle
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from tqdm import tqdm

# -------------------
# Config
# -------------------
PROC_PKL = "/Media_HDD/lykang/dataset/Reg2Prostate/processed_pkl_normalized"
OUT_ROOT = "/Media_HDD/lykang/dataset/Reg2Prostate/slices"
os.makedirs(OUT_ROOT, exist_ok=True)

SLICE_HW = (64, 64)
NUM_SLICES_PER_VOL = 4
SCALER = np.array([10, 10, 10, 10, 10, 10], dtype=np.float32)
DELTA, LOWER_BOUND = 0.35, 0.0


def save_slice_pkl(path_out, vol_zyx, slice_hw, dof):
    dummy_mask = np.zeros_like(vol_zyx, dtype=np.uint8)
    dummy_slice_mask = np.zeros_like(slice_hw, dtype=np.uint8)
    payload = (
        vol_zyx.astype(np.float32, copy=False),
        dummy_mask,
        slice_hw.astype(np.float32, copy=False),
        dummy_slice_mask,
        dof.astype(np.float32, copy=False),
    )
    with open(path_out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class FrameRigidTransformer(nn.Module):
    def __init__(self, slice_size_hw, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.slice_size = [1] + list(slice_size_hw)
        vectors = [torch.linspace(-0.5 * (s - 1), 0.5 * (s - 1), steps=s) for s in self.slice_size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack([grids[2], grids[1], grids[0], torch.ones_like(grids[0])], dim=0)
        self.register_buffer("grid", grid.view(4, -1).float().contiguous())

    def dof2mat(self, input_dof):
        rad = tgm.deg2rad(input_dof[:, 3:])
        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk
        M = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
        M[:, 0, 0] = cj * ck
        M[:, 0, 1] = sj * sc - cs
        M[:, 0, 2] = sj * cc + ss
        M[:, 1, 0] = cj * sk
        M[:, 1, 1] = sj * ss + cc
        M[:, 1, 2] = sj * cs - sc
        M[:, 2, 0] = -sj
        M[:, 2, 1] = cj * si
        M[:, 2, 2] = cj * ci
        M[:, :3, 3] = input_dof[:, :3]
        return M

    def forward(self, vol, dof):
        mat = self.dof2mat(dof)
        new_locs = (mat @ self.grid)[:, :3]
        D, H, W = vol.shape[2], vol.shape[3], vol.shape[4]
        for i, sz in enumerate((D, H, W)):
            new_locs[:, i] = 2 * ((new_locs[:, i] + 0.5 * ((D, H, W)[2 - i] - 1)) / ((D, H, W)[2 - i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 1).contiguous().view(vol.shape[0], *([1] + list(self.slice_size[1:])), 3)
        return F.grid_sample(vol, new_locs, align_corners=True, mode=self.mode, padding_mode="border")


def generate_slices_for_volume(path: str, out_dir: str):
    with open(path, "rb") as f:
        vol_zyx, _ = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frt_img = FrameRigidTransformer(SLICE_HW, mode="bilinear").to(device)

    vol_th = torch.from_numpy(vol_zyx).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    base = osp.basename(path).replace("_z64y64x64.pkl", "")
    rng = np.random.RandomState(abs(hash(base)) % (2**31))

    got, trials = 0, 0
    while got < NUM_SLICES_PER_VOL and trials < 500:
        trials += 1
        dof_np = (rng.rand(6).astype(np.float32) - 0.5) * 2.0 * SCALER
        dof = torch.from_numpy(dof_np[None, :]).to(device)

        slc = frt_img(vol_th, dof)[0, 0, 0].detach().cpu().numpy()
        score = (slc > LOWER_BOUND).mean()
        if score <= DELTA:
            continue

        out_name = f"{base}_slice{got:02d}_z64y64x64.pkl"
        out_path = osp.join(out_dir, out_name)
        save_slice_pkl(out_path, vol_zyx, slc, dof_np)
        got += 1


def main():
    paths = sorted(glob(osp.join(PROC_PKL, "*.pkl")))
    for path in tqdm(paths, desc="Generating prostate US slices"):
        generate_slices_for_volume(path, OUT_ROOT)
    print(f"Done. Output -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
