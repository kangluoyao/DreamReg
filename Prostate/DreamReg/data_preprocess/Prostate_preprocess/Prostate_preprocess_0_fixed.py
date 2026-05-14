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
ROOT = "/path/to/your/dataset"
OUT_ROOT = "/path/to/your/preprocessed_data"
os.makedirs(OUT_ROOT, exist_ok=True)

TARGET_SHAPE = (64, 64, 64)

RESAMPLE_TO_ISO = True
TARGET_SPACING_XYZ_MM = (1.0, 1.0, 1.0)
SAVE_FIRST_NII_GZ = True
_saved_preview = False
KEEP_LEGACY_TRANSPOSE = True
LEGACY_TRANSPOSE_AXES = (2, 0, 1)


def center_crop_pad(arr_3d: np.ndarray, target_shape, pad_value: float) -> np.ndarray:
    """Center crop then pad to target (generic 3D array)."""
    target_shape = np.array(target_shape, dtype=np.int64)
    cur_shape = np.array(arr_3d.shape, dtype=np.int64)

    out = arr_3d
    for axis in range(3):
        if cur_shape[axis] > target_shape[axis]:
            start = (cur_shape[axis] - target_shape[axis]) // 2
            end = start + target_shape[axis]
            slicer = [slice(None)] * 3
            slicer[axis] = slice(start, end)
            out = out[tuple(slicer)]
            cur_shape = np.array(out.shape, dtype=np.int64)

    pad_width = []
    for axis in range(3):
        if cur_shape[axis] < target_shape[axis]:
            total = target_shape[axis] - cur_shape[axis]
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
    else:
        # degenerate volume
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def resample_sitk(img: sitk.Image, out_spacing_xyz) -> sitk.Image:
    in_spacing = img.GetSpacing()  # (x,y,z)
    in_size = img.GetSize()        # (x,y,z)

    out_spacing = tuple(float(s) for s in out_spacing_xyz)

    out_size = [
        int(np.round(in_size[i] * (in_spacing[i] / out_spacing[i])))
        for i in range(3)
    ]
    out_size = [max(1, s) for s in out_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    # Identity transform (resample into same physical space)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    return resampler.Execute(img)


def save_pkl(array_3d, meta, out_path):
    with open(out_path, "wb") as f:
        pickle.dump((array_3d, meta), f, protocol=pickle.HIGHEST_PROTOCOL)


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
    orig_spacing = img.GetSpacing()  # xyz
    orig_size = img.GetSize()        # xyz

    # Optional: cast to float for stable interpolation
    if img.GetPixelID() not in (sitk.sitkFloat32, sitk.sitkFloat64):
        img = sitk.Cast(img, sitk.sitkFloat32)

    if RESAMPLE_TO_ISO:
        img_rs = resample_sitk(img, TARGET_SPACING_XYZ_MM)
    else:
        img_rs = img

    rs_spacing = img_rs.GetSpacing()
    rs_size = img_rs.GetSize()

    arr_zyx = sitk.GetArrayFromImage(img_rs)  # (Z, Y, X)
    if arr_zyx.ndim > 3:
        # keep first channel/time if present
        arr_zyx = arr_zyx[0]

    if KEEP_LEGACY_TRANSPOSE:
        arr = arr_zyx.transpose(*LEGACY_TRANSPOSE_AXES)  # -> (X, Z, Y)
        array_order = "XZY"  # axis0=X, axis1=Z, axis2=Y
    else:
        arr = arr_zyx
        array_order = "ZYX"  # axis0=Z, axis1=Y, axis2=X

    pad_value = float(arr.min()) if arr.size else 0.0
    arr = center_crop_pad(arr, TARGET_SHAPE, pad_value)
    arr = percentile_normalize(arr, 1, 99)

    base = osp.basename(path).replace(".nii.gz", "")
    out_path = osp.join(OUT_ROOT, f"{base}_z64y64x64.pkl")
    meta = {
        "source_path": path,
        "orig_spacing_xyz_mm": tuple(float(x) for x in orig_spacing),
        "orig_size_xyz": tuple(int(x) for x in orig_size),
        "resampled": bool(RESAMPLE_TO_ISO),
        "spacing_xyz_mm": tuple(float(x) for x in rs_spacing),
        "size_xyz": tuple(int(x) for x in rs_size),
        "array_order": array_order,
        "shape_after_crop": tuple(int(x) for x in arr.shape),
        "target_shape": tuple(int(x) for x in TARGET_SHAPE),
        "legacy_transpose_enabled": bool(KEEP_LEGACY_TRANSPOSE),
        "legacy_transpose_axes": tuple(int(x) for x in LEGACY_TRANSPOSE_AXES) if KEEP_LEGACY_TRANSPOSE else None,
    }
    save_pkl(arr, meta, out_path)

    if SAVE_FIRST_NII_GZ and not _saved_preview:
        preview_path = osp.join(OUT_ROOT, f"{base}_z64y64x64_preview.nii.gz")
        preview_img = sitk.GetImageFromArray(arr)  # SITK expects array in (Z,Y,X) semantics
        preview_img.SetSpacing(rs_spacing)
        sitk.WriteImage(preview_img, preview_path)
        _saved_preview = True


def main():
    paths = collect_us_images()
    for path in tqdm(paths, desc="Processing prostate US volumes"):
        process_one(path)
    print(f"Done. Output -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
