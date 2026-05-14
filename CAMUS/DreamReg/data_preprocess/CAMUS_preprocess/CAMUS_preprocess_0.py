#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

ROOT = "/path/to/your/dataset/"
OUT_ROOT = "/path/to/your/preprocessed_dataset"
os.makedirs(OUT_ROOT, exist_ok=True)

TARGET_SIZE_ZXY = (32, 128, 128)           # (Z, X, Y)
TARGET_SPACING = (1.0, 1.0, 1.0)           # (sx, sy, sz) in mm (sitk uses (x,y,z))
DICOM_ORIENT = "RAI"

def reorient_to(img, orient_code=DICOM_ORIENT):
    return sitk.DICOMOrient(img, orient_code)

def physical_center(img: sitk.Image):
    size = np.array(list(img.GetSize()), dtype=np.float64)        # (x,y,z)
    spacing = np.array(list(img.GetSpacing()), dtype=np.float64)  # (x,y,z)
    origin = np.array(list(img.GetOrigin()), dtype=np.float64)
    direction = np.array(list(img.GetDirection()), dtype=np.float64).reshape(3,3)
    center_index = (size - 1.0) / 2.0
    return origin + direction.dot(center_index * spacing)

def resample_iso_centered(img: sitk.Image,
                          out_spacing=(1.0,1.0,1.0),
                          out_size_xyz=(192,192,32),
                          interp=sitk.sitkLinear) -> sitk.Image:
    img = reorient_to(img, DICOM_ORIENT)

    out_direction = img.GetDirection()

    C = physical_center(img)
    out_size = np.array(out_size_xyz, dtype=np.int64)             # (x,y,z)
    out_spacing = np.array(out_spacing, dtype=np.float64)         # (x,y,z)
    dir_mat = np.array(out_direction).reshape(3,3)

    half_idx = (out_size - 1.0) / 2.0
    out_origin = C - dir_mat.dot(half_idx * out_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(tuple(out_size.tolist()))
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetOutputDirection(tuple(out_direction))
    resampler.SetOutputOrigin(tuple(out_origin.tolist()))
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()) if img.GetPixelIDValue()!=sitk.sitkUInt8 else 0.0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    out = resampler.Execute(img)
    return out

def save_pkl(array_zxy: np.ndarray, meta: dict, out_path: str):
    array_zxy = array_zxy.astype(np.float32, copy=False)
    with open(out_path, 'wb') as f:
        pickle.dump((array_zxy, meta), f, protocol=pickle.HIGHEST_PROTOCOL)

def process_case(case_id: int):
    pid = f"patient{case_id:04d}"
    case_dir = osp.join(ROOT, pid)
    if not osp.isdir(case_dir):
        return False, f"{pid}: folder missing"

    in_files = [
        (f"{pid}_2CH_half_sequence.nii.gz", "2CH"),
        (f"{pid}_4CH_half_sequence.nii.gz", "4CH"),
    ]

    for fname, view in in_files:
        in_path = osp.join(case_dir, fname)
        if not osp.isfile(in_path):
            continue

        img = sitk.ReadImage(in_path)              # (x,y,z) spacing/order internally
        img_res = resample_iso_centered(
            img,
            out_spacing=(TARGET_SPACING[1], TARGET_SPACING[2], TARGET_SPACING[0]) if False else TARGET_SPACING,
            out_size_xyz=(TARGET_SIZE_ZXY[1], TARGET_SIZE_ZXY[2], TARGET_SIZE_ZXY[0]),
            interp=sitk.sitkLinear
        )

        arr_zyx = sitk.GetArrayFromImage(img_res)  # (Z, Y, X)

        if arr_zyx.dtype != np.float32:
            arr_zyx = arr_zyx.astype(np.float32)
        p1, p99 = np.percentile(arr_zyx, (1, 99))
        if p99 > p1:
            arr_zyx = np.clip((arr_zyx - p1) / (p99 - p1), 0.0, 1.0)

        out_name = f"{pid}_{view}_half_sequence_zxy32x192x192.pkl"
        out_path = osp.join(OUT_ROOT, out_name)


        # visualization
        # nii_path = osp.join(OUT_ROOT, out_name.replace(".pkl", ".nii.gz"))
        # sitk.WriteImage(img_res, nii_path)
        # print(f"Saved NIfTI for visualization: {nii_path}")

        meta = {
            "patient_id": pid,
            "view": view,
            "orientation": DICOM_ORIENT,
            "spacing_xyz_mm": (1.0, 1.0, 1.0),      # (x,y,z)
            "shape_zxy": arr_zyx.shape,            # (Z,X,Y) = (32,192,192)
            "source_path": in_path,
        }
        save_pkl(arr_zyx, meta, out_path)

    return True, pid

def main():
    ok_cnt = 0
    for i in tqdm(range(1, 500+1)):
        ok, msg = process_case(i)
        if ok:
            ok_cnt += 1
        else:
            print(f"[WARN] {msg}")
    print(f"Done. processed={ok_cnt} / 500. Output -> {OUT_ROOT}")

if __name__ == "__main__":
    main()
