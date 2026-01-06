#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import pickle
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# -------------------
# Config
# -------------------
ROOT = "/Media_HDD/lykang/dataset/Reg2Prostate"
OUT_ROOT = "/Media_HDD/lykang/dataset/Reg2Prostate/processed_pkl_normalized"
os.makedirs(OUT_ROOT, exist_ok=True)

TARGET_SHAPE_ZYX = (64, 64, 64)
SAVE_FIRST_NII_GZ = True
_saved_preview = False


def center_crop_pad(arr_zyx: np.ndarray, target_shape_zyx, pad_value: float) -> np.ndarray:
    target_shape_zyx = np.array(target_shape_zyx, dtype=np.int64)
    cur_shape = np.array(arr_zyx.shape, dtype=np.int64)

    out = arr_zyx
    for axis in range(3):
        if cur_shape[axis] > target_shape_zyx[axis]:
            start = (cur_shape[axis] - target_shape_zyx[axis]) // 2
            end = start + target_shape_zyx[axis]
            slicer = [slice(None)] * 3
            slicer[axis] = slice(start, end)
            out = out[tuple(slicer)]
            cur_shape = np.array(out.shape, dtype=np.int64)

    pad_width = []
    for axis in range(3):
        if cur_shape[axis] < target_shape_zyx[axis]:
            total = target_shape_zyx[axis] - cur_shape[axis]
            before = total // 2
            after = total - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))

    if any(p != (0, 0) for p in pad_width):
        out = np.pad(out, pad_width, mode="constant", constant_values=pad_value)

    return out


def percentile_normalize(arr: np.ndarray, p1=1, p99=99) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, (p1, p99))
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr


def save_pkl(array_zyx: np.ndarray, meta: dict, out_path: str):
    array_zyx = array_zyx.astype(np.float32, copy=False)
    with open(out_path, "wb") as f:
        pickle.dump((array_zyx, meta), f, protocol=pickle.HIGHEST_PROTOCOL)


def collect_us_images():
    paths = []
    for split in ("train", "val"):
        split_dir = osp.join(ROOT, split, "us_images")
        paths.extend(sorted(glob(osp.join(split_dir, "*.nii.gz"))))
    return paths


def process_one(path: str):
    global _saved_preview
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    img = sitk.ReadImage(path)
    spacing_xyz = img.GetSpacing()

    arr_zyx = sitk.GetArrayFromImage(img)
    if arr_zyx.ndim > 3:
        arr_zyx = arr_zyx[0]

    pad_value = float(arr_zyx.min()) if arr_zyx.size else 0.0
    arr_zyx = center_crop_pad(arr_zyx, TARGET_SHAPE_ZYX, pad_value)
    arr_zyx = percentile_normalize(arr_zyx, 1, 99)

    base = osp.basename(path).replace(".nii.gz", "")
    out_path = osp.join(OUT_ROOT, f"{base}_z64y64x64.pkl")
    meta = {
        "source_path": path,
        "spacing_xyz_mm": spacing_xyz,
        "shape_zyx": arr_zyx.shape,
    }
    save_pkl(arr_zyx, meta, out_path)

    if SAVE_FIRST_NII_GZ and not _saved_preview:
        preview_path = osp.join(OUT_ROOT, f"{base}_z64y64x64.nii.gz")
        preview_img = sitk.GetImageFromArray(arr_zyx)
        preview_img.SetSpacing(spacing_xyz)
        sitk.WriteImage(preview_img, preview_path)
        _saved_preview = True


def main():
    paths = collect_us_images()
    for path in tqdm(paths, desc="Processing prostate US volumes"):
        process_one(path)
    print(f"Done. Output -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
