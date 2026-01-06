import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import losses
import random
import sys
import utils

import numpy as np
import torch
import torch.nn as nn
from models.baseline_wm_v2 import EUReg_WM_Belief   # ✅ 用新 world model
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM
from data.datasets import TestDataset
import matplotlib.pyplot as plt


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

same_seeds(2032)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


GPU_iden = 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_steps', type=int, default=20)
    parser.add_argument('--step_scale', type=float, default=1.0)
    args, _ = parser.parse_known_args()

    val_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/testing/'
    dim = 16
    save_dir = 'wm_base_1/'
    img_size = (32, 192, 192)

    # ===== Init model =====
    model = EUReg_WM_Belief(img_size).cuda()
    model_dir = 'experiments/CAMUS2/' + save_dir
    best_model = torch.load(model_dir + 'best_model.pth.tar', map_location='cuda:0')['state_dict']
    model.load_state_dict(best_model)
    print("Loaded model:", model_dir + 'best_model.pth.tar')
    del best_model

    # ===== Data =====
    test_set = TestDataset(glob.glob(val_dir + '*.pkl'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    # ===== Metrics =====
    Lcorner = losses.CornerDistLoss()
    Lsml1 = nn.SmoothL1Loss()
    Sim_ssim = SSIM(data_range=1.0, size_average=True, channel=1)

    eval_DistErr = utils.AverageMeter()
    eval_NCC = utils.AverageMeter()
    eval_ParaErr = utils.AverageMeter()
    eval_TransErr = utils.AverageMeter()
    eval_RotErr = utils.AverageMeter()
    eval_ParamNCC = utils.AverageMeter()
    eval_SSIM = utils.AverageMeter()
    eval_FPS = utils.AverageMeter()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    pic_path = 'showdata/' + save_dir
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    stdy_idx = 0
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            vol   = data[0].cuda()
            frame = data[1].cuda()
            dof   = data[2].cuda()

            starter.record()

            T0 = torch.zeros(vol.size(0),6,device=vol.device)

            # ✅ world model inference
            pred_dof, sampled_frame = model(
                vol, frame, dof, T0,                      # <--- ✅ 追加 dof
                steps=args.wm_steps,
                step_scale=args.step_scale,
                noise_std=0.0,
                return_all=False
            )

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            # ===== metrics (unchanged) =====
            trans_l2 = losses.L2dist(pred_dof[:, :3], dof[:, :3]).item() * 0.62
            rot_l2 = losses.L2dist(pred_dof[:, 3:], dof[:, 3:]).item()
            dist_err = Lcorner(pred_dof, dof).item() * 0.62
            param_ncc = losses.transformation_parameter_NCC(pred_dof, dof).item()*100
            param_err = Lsml1(pred_dof[:, :3], dof[:, :3]).item() + \
                        Lsml1(pred_dof[:, 3:], dof[:, 3:]).item()
            ncc = losses.normalized_cross_correlation(frame, sampled_frame).item()*100
            ssim = Sim_ssim(frame, sampled_frame).item()*100

            eval_DistErr.update(dist_err, vol.size(0))
            eval_TransErr.update(trans_l2, vol.size(0))
            eval_RotErr.update(rot_l2, vol.size(0))
            eval_ParamNCC.update(param_ncc, vol.size(0))
            eval_ParaErr.update(param_err, vol.size(0))
            eval_NCC.update(ncc, vol.size(0))
            eval_SSIM.update(ssim, vol.size(0))
            if stdy_idx > 20:
                eval_FPS.update(1000 / curr_time, vol.size(0))

            print(f"DistErr: {dist_err:.6f}, NCC: {ncc:.6f}, ParaErr: {param_err:.6f}")

            # save prediction
            plt.imsave(
                pic_path + f'/{stdy_idx:03d}_dist_{dist_err:.2f}_ncc_{ncc:.2f}.png',
                sampled_frame.squeeze().detach().cpu().numpy(), cmap='gray'
            )

            stdy_idx += 1

    log_str = (
        'DistErr: {:.2f} +- {:.2f} mm, NCC: {:.2f} +- {:.2f} %, SSIM: {:.2f} +- {:.2f} %, '
        'TransErr: {:.2f} +- {:.2f} mm , RotErr: {:.2f} +- {:.2f}, ParamNCC: {:.2f} +- {:.2f} %, FPS: {}'
    ).format(
        eval_DistErr.avg, eval_DistErr.std,
        eval_NCC.avg, eval_NCC.std,
        eval_SSIM.avg, eval_SSIM.std,
        eval_TransErr.avg, eval_TransErr.std,
        eval_RotErr.avg, eval_RotErr.std,
        eval_ParamNCC.avg, eval_ParamNCC.std,
        int(eval_FPS.avg)
    )

    print(log_str)
    test_log_path = 'logs/' + save_dir + 'test_log.txt'
    os.makedirs(os.path.dirname(test_log_path), exist_ok=True)
    with open(test_log_path, 'a') as f:
        f.write(log_str + '\n')


if __name__ == '__main__':
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print(f'     GPU #{GPU_idx}: {GPU_name}')
    torch.cuda.set_device(GPU_iden)
    print('Using GPU:', torch.cuda.get_device_name(GPU_iden))
    print('GPU available:', torch.cuda.is_available())
    main()
