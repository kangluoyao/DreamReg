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
from PIL import Image

# -------------------
# Config
# -------------------
PROC_PKL = "/path/to/your/dataset"
OUT_ROOT = "/path/to/your/preprocessed_data"
os.makedirs(OUT_ROOT, exist_ok=True)

SLICE_HW = (64, 64)
NUM_SLICES_PER_VOL = 8

TRANS_SCALER_VOX = np.array([10, 10, 10], dtype=np.float32)
ROT_SCALER_DEG = np.array([10, 10, 10], dtype=np.float32)

DELTA, LOWER_BOUND = 0.8, 0.0

SAVE_PREVIEW = True
PREVIEW_LIMIT = 8


def save_slice_pkl(path_out, vol_3d, slice_hw, dof_6, meta):
    dummy_mask = np.zeros_like(vol_3d, dtype=np.uint8)
    dummy_slice_mask = np.zeros_like(slice_hw, dtype=np.uint8)
    payload = (
        vol_3d.astype(np.float32, copy=False),
        dummy_mask,
        slice_hw.astype(np.float32, copy=False),
        dummy_slice_mask,
        dof_6.astype(np.float32, copy=False),
        meta,
    )
    with open(path_out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class FrameRigidTransformer(nn.Module):
    def __init__(self, slice_size_hw, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.slice_size = [1] + list(slice_size_hw)  # [z=1, y, x]

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

        # translation (voxel units)
        M[:, :3, 3] = input_dof[:, :3]
        return M

    def forward(self, vol, dof):
        mat = self.dof2mat(dof)
        new_locs = (mat @ self.grid)[:, :3]

        D, H, W = vol.shape[2], vol.shape[3], vol.shape[4]
        sizes = (W, H, D)
        for i, sz in enumerate(sizes):  # i=0:x,1:y,2:z
            new_locs[:, i] = 2.0 * ((new_locs[:, i] + 0.5 * (sz - 1)) / (sz - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 1).contiguous().view(vol.shape[0], 1, self.slice_size[1], self.slice_size[2], 3)
        return F.grid_sample(vol, new_locs, align_corners=True, mode=self.mode, padding_mode="border")


def save_preview_slice(slice_hw: np.ndarray, out_path: str):
    img = np.clip(slice_hw, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def list_processed_volumes():
    return sorted(glob(osp.join(PROC_PKL, "*_z64y64x64.pkl")))


def generate_slices_for_volume(path: str, out_dir: str, preview_state: dict):
    with open(path, "rb") as f:
        vol_3d, meta = pickle.load(f)

    vol_3d = vol_3d.astype(np.float32, copy=False)
    assert vol_3d.ndim == 3, f"Expected 3D volume, got {vol_3d.shape}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frt_img = FrameRigidTransformer(SLICE_HW, mode="bilinear").to(device)

    vol_th = torch.from_numpy(vol_3d).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    base = osp.basename(path).replace("_z64y64x64.pkl", "")
    rng = np.random.RandomState(abs(hash(base)) % (2**31))

    got, trials = 0, 0
    while got < NUM_SLICES_PER_VOL and trials < 500:
        trials += 1

        trans = (rng.rand(3).astype(np.float32) - 0.5) * 2.0 * TRANS_SCALER_VOX
        rot = (rng.rand(3).astype(np.float32) - 0.5) * 2.0 * ROT_SCALER_DEG
        dof_np = np.concatenate([trans, rot], axis=0).astype(np.float32)

        dof = torch.from_numpy(dof_np[None, :]).to(device)
        slc = frt_img(vol_th, dof)[0, 0, 0].detach().cpu().numpy()  # [H,W]

        score = (slc > LOWER_BOUND).mean()
        if score <= DELTA:
            continue

        out_name = f"{base}_slice{got:02d}_z64y64x64.pkl"
        out_path = osp.join(out_dir, out_name)

        slice_meta = {
            "source_volume_pkl": path,
            "source_base": base,
            "volume_meta": meta,
            "array_order": meta.get("array_order", "UNKNOWN"),
            "volume_shape": tuple(int(x) for x in vol_3d.shape),
            "filter_lower_bound": float(LOWER_BOUND),
            "filter_delta": float(DELTA),
        }
        save_slice_pkl(out_path, vol_3d, slc, dof_np, slice_meta)

        if SAVE_PREVIEW and preview_state["count"] < PREVIEW_LIMIT:
            preview_name = out_name.replace(".pkl", ".png")
            save_preview_slice(slc, osp.join(out_dir, preview_name))
            preview_state["count"] += 1

        got += 1


def main():
    vols = list_processed_volumes()
    preview_state = {"count": 0}
    for path in tqdm(vols, desc="Generating prostate slices"):
        generate_slices_for_volume(path, OUT_ROOT, preview_state)
    print(f"Done. Output -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
