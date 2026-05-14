#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM

import losses
import utils
from models.baseline_wm import EUReg_WM_Belief 
from data.datasets import TestDataset


def same_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


same_seeds(17)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_steps', type=int, default=5)
    parser.add_argument('--step_scale', type=float, default=1.0)
    parser.add_argument('--gpu', type=int, default=0)
    args, _ = parser.parse_known_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    val_dir = '/path/to/your/testing_set'
    img_size = (64, 64, 64)
    save_dir = 'save_dir/'
    model_dir = os.path.join('experiments/Prostate/', save_dir)
    best_model_path = os.path.join(model_dir, 'best_model.pth.tar')

    pic_path = os.path.join('showdata', save_dir)
    os.makedirs(pic_path, exist_ok=True)

    # -------- Init model & load weights --------
    print(">>> Building model EUReg_WM_Belief...")
    model = EUReg_WM_Belief(img_size).to(device)

    print(">>> Loading checkpoint from:", best_model_path)
    ckpt = torch.load(best_model_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(">>> Model loaded.")

    # -------- Data --------
    test_set = TestDataset(glob.glob(os.path.join(val_dir, '*.pkl')))
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # -------- Metrics --------
    Lcorner = losses.CornerDistLoss()
    Lsml1 = nn.SmoothL1Loss()
    Sim_ssim = SSIM(data_range=1.0, size_average=True, channel=1)

    eval_DistErr   = utils.AverageMeter()
    eval_NCC       = utils.AverageMeter()
    eval_ParaErr   = utils.AverageMeter()
    eval_TransErr  = utils.AverageMeter()
    eval_RotErr    = utils.AverageMeter()
    eval_ParamNCC  = utils.AverageMeter()
    eval_SSIM      = utils.AverageMeter()
    eval_FPS       = utils.AverageMeter()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    stdy_idx = 0
    with torch.no_grad():
        for data in test_loader:
            model.eval()

            vol   = data[0].to(device, non_blocking=True)  # (B,1,D,H,W)
            frame = data[1].to(device, non_blocking=True)  # (B,1,192,192
            dof   = data[2].to(device, non_blocking=True)  # (B,6) GT pose

            starter.record()

            T0 = torch.zeros(vol.size(0), 6, device=device)

            # ===== world model inference=====
            # forward(self, vol, goal_sl, T0, steps, step_scale, return_all=False)
            pred_dof, sampled_frame = model(
                vol, frame, T0,
                steps=args.wm_steps,
                step_scale=args.step_scale,
                return_all=False
            )

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # ms

            # ===== metrics =====
            trans_l2 = losses.L2dist(pred_dof[:, :3], dof[:, :3]).item()
            rot_l2   = losses.L2dist(pred_dof[:, 3:], dof[:, 3:]).item()
            dist_err = Lcorner(pred_dof, dof).item()
            param_ncc = losses.transformation_parameter_NCC(pred_dof, dof).item() * 100
            param_err = (
                Lsml1(pred_dof[:, :3], dof[:, :3]).item() +
                Lsml1(pred_dof[:, 3:], dof[:, 3:]).item()
            )
            ncc  = losses.normalized_cross_correlation(frame, sampled_frame).item() * 100
            ssim = Sim_ssim(frame, sampled_frame).item() * 100

            bs = vol.size(0)
            eval_DistErr.update(dist_err, bs)
            eval_TransErr.update(trans_l2, bs)
            eval_RotErr.update(rot_l2, bs)
            eval_ParamNCC.update(param_ncc, bs)
            eval_ParaErr.update(param_err, bs)
            eval_NCC.update(ncc, bs)
            eval_SSIM.update(ssim, bs)
            if stdy_idx > 20:
                eval_FPS.update(1000.0 / curr_time, bs)

            print(f"[{stdy_idx:03d}] DistErr: {dist_err:.4f}, NCC: {ncc:.4f}, ParaErr: {param_err:.4f}")

            plt.imsave(
                os.path.join(
                    pic_path,
                    f'{stdy_idx:03d}_dist_{dist_err:.2f}_ncc_{ncc:.2f}.png'
                ),
                sampled_frame.squeeze().detach().cpu().numpy(),
                cmap='gray'
            )

            stdy_idx += 1

    log_str = (
        'DistErr: {:.2f} +- {:.2f} mm, NCC: {:.2f} +- {:.2f} %, SSIM: {:.2f} +- {:.2f} %, '
        'TransErr: {:.2f} +- {:.2f} mm , RotErr: {:.2f} +- {:.2f}, ParamNCC: {:.2f} +- {:.2f} %, FPS: {}'
    ).format(
        eval_DistErr.avg, eval_DistErr.std,
        eval_NCC.avg,     eval_NCC.std,
        eval_SSIM.avg,    eval_SSIM.std,
        eval_TransErr.avg, eval_TransErr.std,
        eval_RotErr.avg,   eval_RotErr.std,
        eval_ParamNCC.avg, eval_ParamNCC.std,
        int(eval_FPS.avg)
    )

    print("\n>>> FINAL TEST METRICS")
    print(log_str)

    test_log_path = os.path.join('logs', save_dir, 'test_log.txt')
    os.makedirs(os.path.dirname(test_log_path), exist_ok=True)
    with open(test_log_path, 'a') as f:
        f.write(log_str + '\n')


if __name__ == '__main__':
    main()
