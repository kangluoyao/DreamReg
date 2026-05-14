#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, os.path as osp, pickle
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import torch, torch.nn as nn, torch.nn.functional as F
import torchgeometry as tgm

# -------------------
# Config
# -------------------
ROOT_DB   = "/Media_HDD/lykang/dataset/CAMUS_public/database_nifti"
PROC_PKL  = "/Media_HDD/lykang/dataset/CAMUS_public/processed_pkl_normalized"
SPLIT_DIR = "/Media_HDD/lykang/dataset/CAMUS_public/database_split"

OUT_ROOT  = "/Media_HDD/lykang/dataset/CAMUS_public/splits"
OUT_TRAIN = osp.join(OUT_ROOT, "training")
OUT_VAL   = osp.join(OUT_ROOT, "validation")
OUT_TEST  = osp.join(OUT_ROOT, "testing")
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

TRAIN_TXT = osp.join(SPLIT_DIR, "subgroup_training.txt")
VAL_TXT   = osp.join(SPLIT_DIR, "subgroup_validation.txt")
TEST_TXT  = osp.join(SPLIT_DIR, "subgroup_testing.txt")

DICOM_ORIENT = "RAI"
SPACING_XYZ  = (1.0, 1.0, 1.0)
SIZE_XYZ     = (128, 128, 32)
SLICE_HW     = (128, 128)               
NUM_SLICES_PER_VIEW_VAL  = 4 
NUM_SLICES_PER_VIEW_TEST = 4 
SCALER = np.array([10,10,10, 10,10,10], dtype=np.float32)  # [tx,ty,tz,rx,ry,rz]
DELTA, LOWER_BOUND = 0.35, 0.0

# -------------------
# Helpers
# -------------------
def read_ids(txt_path):
    ids = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s.split()[0])
    return sorted(set(ids))

def reorient_to(img, orient_code=DICOM_ORIENT):
    return sitk.DICOMOrient(img, orient_code)

def physical_center(img: sitk.Image):
    size = np.array(img.GetSize(), dtype=np.float64)        # (x,y,z)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)  # (x,y,z)
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3,3)
    center_index = (size - 1.0) / 2.0
    return origin + direction.dot(center_index * spacing)

def resample_iso_centered(img: sitk.Image,
                          out_spacing=SPACING_XYZ,
                          out_size_xyz=SIZE_XYZ,
                          interp=sitk.sitkLinear) -> sitk.Image:
    img = reorient_to(img, DICOM_ORIENT)
    out_direction = img.GetDirection()
    C = physical_center(img)
    out_size = np.array(out_size_xyz, dtype=np.int64)
    out_spacing = np.array(out_spacing, dtype=np.float64)
    dir_mat = np.array(out_direction).reshape(3,3)
    half_idx = (out_size - 1.0) / 2.0
    out_origin = C - dir_mat.dot(half_idx * out_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(tuple(out_size.tolist()))
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetOutputDirection(tuple(out_direction))
    resampler.SetOutputOrigin(tuple(out_origin.tolist()))
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    return resampler.Execute(img)

def percentile_normalize(a: np.ndarray, p1=1, p99=99):
    a = a.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, (p1, p99))
    if hi > lo:
        a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return a

def save_test_sample_pkl(path_out, vol_zyx, msk_zyx, slc_hw, slc_msk_hw, dof):
    payload = (
        vol_zyx.astype(np.float32, copy=False),   # (Z,Y,X)
        msk_zyx.astype(np.uint8,  copy=False),    # (Z,Y,X)
        slc_hw.astype(np.float32, copy=False),    # (H,W)
        slc_msk_hw.astype(np.uint8,  copy=False), # (H,W)
        dof.astype(np.float32, copy=False),       # (6,)
    )
    with open(path_out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class FrameRigidTransformer(nn.Module):
    def __init__(self, slice_size_hw, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.slice_size = [1] + list(slice_size_hw)   # [1,H,W]
        vectors = [torch.linspace(-0.5*(s-1), 0.5*(s-1), steps=s) for s in self.slice_size]
        grids = torch.meshgrid(vectors, indexing='ij')  # z,y,x
        grid = torch.stack([grids[2], grids[1], grids[0], torch.ones_like(grids[0])], dim=0)  # (x,y,z,1)
        self.register_buffer('grid', grid.view(4, -1).float().contiguous())

    def dof2mat(self, input_dof):
        rad = tgm.deg2rad(input_dof[:, 3:])
        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci*ck, ci*sk
        sc, ss = si*ck, si*sk
        M = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
        M[:,0,0]=cj*ck; M[:,0,1]=sj*sc-cs; M[:,0,2]=sj*cc+ss
        M[:,1,0]=cj*sk; M[:,1,1]=sj*ss+cc; M[:,1,2]=sj*cs-sc
        M[:,2,0]=-sj;   M[:,2,1]=cj*si;    M[:,2,2]=cj*ci
        M[:, :3, 3] = input_dof[:, :3]
        return M

    def forward(self, vol, dof):
        mat = self.dof2mat(dof)
        new_locs = (mat @ self.grid)[:, :3]
        D,H,W = vol.shape[2], vol.shape[3], vol.shape[4]
        for i,sz in enumerate((D,H,W)):
            new_locs[:, i] = 2 * ((new_locs[:, i] + 0.5*( (D,H,W)[2-i]-1)) / ((D,H,W)[2-i]-1) - 0.5)
        new_locs = new_locs.permute(0,2,1).contiguous().view(vol.shape[0], *([1]+list(self.slice_size[1:])), 3)
        return F.grid_sample(vol, new_locs, align_corners=True, mode=self.mode,
                             padding_mode='border' if self.mode=='bilinear' else 'zeros')

def generate_fixed_slices_for_patient(pid: str, out_dir: str, num_per_view: int, seed_offset: int):
    views = [("2CH", f"{pid}_2CH_half_sequence.nii.gz", f"{pid}_2CH_half_sequence_gt.nii.gz"),
             ("4CH", f"{pid}_4CH_half_sequence.nii.gz", f"{pid}_4CH_half_sequence_gt.nii.gz")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frt_img  = FrameRigidTransformer(SLICE_HW, mode='bilinear').to(device)
    frt_mask = FrameRigidTransformer(SLICE_HW, mode='nearest').to(device)

    for view, img_fn, msk_fn in views:
        img_p = osp.join(ROOT_DB, pid, img_fn)
        msk_p = osp.join(ROOT_DB, pid, msk_fn)
        if not (osp.isfile(img_p) and osp.isfile(msk_p)):
            continue

        img = sitk.ReadImage(img_p)
        msk = sitk.ReadImage(msk_p)
        img_r = resample_iso_centered(img, out_spacing=SPACING_XYZ, out_size_xyz=SIZE_XYZ, interp=sitk.sitkLinear)
        msk_r = resample_iso_centered(msk, out_spacing=SPACING_XYZ, out_size_xyz=SIZE_XYZ, interp=sitk.sitkNearestNeighbor)

        vol_zyx = sitk.GetArrayFromImage(img_r)
        msk_zyx = sitk.GetArrayFromImage(msk_r)

        vol_zyx = percentile_normalize(vol_zyx, 1, 99)

        vol_th = torch.from_numpy(vol_zyx).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1,1,Z,Y,X)
        msk_th = torch.from_numpy(msk_zyx.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)

        base = 13 if view=="2CH" else 17
        seed = (int(pid[-4:]) * base + seed_offset) & 0x7fffffff
        rng = np.random.RandomState(seed)

        got, trials = 0, 0
        while got < num_per_view and trials < 500:
            trials += 1
            dof_np = (rng.rand(6).astype(np.float32) - 0.5) * 2.0 * SCALER
            dof = torch.from_numpy(dof_np[None,:]).to(device)

            slc  = frt_img(vol_th,  dof)[0,0,0].detach().cpu().numpy()
            slcm = frt_mask(msk_th, dof)[0,0,0].detach().cpu().numpy().round().astype(np.uint8)

            score = (slc > LOWER_BOUND).mean()
            if score <= DELTA:
                continue

            out_name = f"{pid}_{view}_half_sequence_slice{got:02d}_z32x192x192.pkl"
            out_path = osp.join(out_dir, out_name)
            save_test_sample_pkl(out_path, vol_zyx, msk_zyx, slc, slcm, dof_np)
            got += 1

def copy_training_pkls():
    train_ids = read_ids(TRAIN_TXT)
    for pid in tqdm(train_ids, desc="Copy training pkls"):
        prefix = f"{pid}_"
        for fn in os.listdir(PROC_PKL):
            if fn.startswith(prefix) and fn.endswith(".pkl"):
                src = osp.join(PROC_PKL, fn)
                dst = osp.join(OUT_TRAIN, fn)
                if not osp.isfile(dst):
                    try:
                        os.link(src, dst)
                    except Exception:
                        import shutil
                        shutil.copy2(src, dst)

def build_validation_pkls():
    val_ids = read_ids(VAL_TXT)
    for pid in tqdm(val_ids, desc="Generate validation slices"):
        generate_fixed_slices_for_patient(pid, OUT_VAL, NUM_SLICES_PER_VIEW_VAL, seed_offset=12345)

def build_testing_pkls():
    test_ids = read_ids(TEST_TXT)
    for pid in tqdm(test_ids, desc="Generate testing slices"):
        generate_fixed_slices_for_patient(pid, OUT_TEST, NUM_SLICES_PER_VIEW_TEST, seed_offset=54321)

def main():
    copy_training_pkls()
    build_validation_pkls()
    build_testing_pkls()
    print("\nDone.")
    print(f"Training  → {OUT_TRAIN}")
    print(f"Validation→ {OUT_VAL}")
    print(f"Testing   → {OUT_TEST}")

if __name__ == "__main__":
    main()
