import glob
import losses as pre_losses
import os
import random
import sys
import utils
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from data.datasets import TrainDataset, TestDataset
from models.baseline_wm_v3 import EUReg_WM_Belief  # ✅ 用新的 world model
import torch.nn.functional as F
from rewards import safe_ncc, sobel_grad
from rewards import soft_hog, hog_cosine_reward, normalize_vec
from losses import SSIMLoss
import losses


def angle_wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0

def unit_vec(x, eps=1e-8):
    # x: (B, D)
    return x / (x.norm(dim=1, keepdim=True) + eps)

def cos_align_loss(a, b, eps=1e-8):
    # a,b: (B, D)
    au = unit_vec(a, eps)
    bu = unit_vec(b, eps)
    return 1.0 - (au * bu).sum(dim=1).mean()

def forward_progress_loss(delta, dir_to_target, margin=0.0, eps=1e-8):
    # delta, dir_to_target: (B, D)
    dir_u = unit_vec(dir_to_target, eps)
    proj = (delta * dir_u).sum(dim=1)   # 每个样本在目标方向的投影
    # 可选限幅，防止极端值（按需保留）
    proj = torch.clamp(proj, min=-10.0, max=10.0)
    return F.relu(margin - proj).mean()


def same_seeds(seed):
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


def run_eval(loader, epoch, model, steps, step_scale, noise_std=0.0, split_name='val'):
    Lcorner = pre_losses.CornerDistLoss()
    Lsml1 = nn.SmoothL1Loss()

    eval_DistErr = utils.AverageMeter()
    eval_NCC     = utils.AverageMeter()
    eval_ParaErr = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for data in loader:
            vol   = data[0].cuda(non_blocking=True)
            frame = data[1].cuda(non_blocking=True)
            dof   = data[2].cuda(non_blocking=True)

            T0 = torch.zeros(dof.size(0), 6, device=vol.device, dtype=vol.dtype)

            # ✅ eval 时也走 world model，只不过 dT 多步累积
            pred_dof, sampled_frame = model(
                vol, frame, dof, T0,
                steps=steps, step_scale=step_scale, noise_std=0.0, return_all=False
            )

            param_err = Lsml1(pred_dof[:, :3], dof[:, :3]).item() + \
                        Lsml1(pred_dof[:, 3:], dof[:, 3:]).item()
            dist_err  = Lcorner(pred_dof, dof).item()
            ncc       = pre_losses.normalized_cross_correlation(frame, sampled_frame).item()

            bs = vol.size(0)
            eval_ParaErr.update(param_err, bs)
            eval_DistErr.update(dist_err,  bs)
            eval_NCC.update(ncc,           bs)

    print(f'[{split_name}] Epoch {epoch}  DistErr: {eval_DistErr.avg:.6f}, '
          f'NCC: {eval_NCC.avg:.6f}, ParaErr: {eval_ParaErr.avg:.6f}')
    return eval_DistErr.avg, eval_NCC.avg, eval_ParaErr.avg


torch.cuda.init()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_steps', type=int, default=10)
    parser.add_argument('--step_scale', type=float, default=1.0)
    parser.add_argument('--noise_std', type=float, default=0.00)
    args, _ = parser.parse_known_args()
    print("start training world model with steps:", args.wm_steps,
          " step_scale:", args.step_scale, " noise_std:", args.noise_std)

    batch_size = 64
    train_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/training/'
    val_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/validation/'
    test_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/testing/'
    img_size = (32, 192, 192)
    slice_size = (192,192)
    dim=16
    lr = 0.0001

    save_dir = 'wm_v3_10steps/'
    if not os.path.exists('experiments/CAMUS2/' + save_dir):
        os.makedirs('experiments/CAMUS2/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    best_path = os.path.join('experiments/CAMUS2/', save_dir, 'best_model.pth.tar')
    best_ParaErr = float('inf')

    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/' + save_dir, 'losses' + ".txt"), "a")

    device = torch.device("cuda")
    epoch_start = 0
    max_epoch = 200
    updated_lr = lr

    # ✅ Dreamer world model
    model = EUReg_WM_Belief(img_size).cuda()

    train_set = TrainDataset(glob.glob(train_dir + '*.pkl'), slice_size, [10]*6)
    val_set   = TestDataset(glob.glob(val_dir + '*.pkl'))
    test_set  = TestDataset(glob.glob(test_dir + '*.pkl'))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr)
    scheduler_warm = lr_scheduler.StepLR(optimizer,step_size=1, gamma=1.2)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    Lsml1 = nn.SmoothL1Loss()
    Lsml2 = nn.SmoothL1Loss()
    Llncc = losses.LNCCLoss()

    it = 0
    for epoch in range(epoch_start, max_epoch):
        loss_all = utils.AverageMeter()
        model.train()

        for data in train_loader:
            it += 1
            vol = data[0].cuda()
            frame = data[1].cuda()
            dof = data[2].cuda()

            T0 = torch.zeros(dof.size(0), 6, device=vol.device)

            # ===== world-model rollout =====
            # 多接一个 dir_info
            T_final, frame_last, traj, dir_info = model(
                vol, frame, dof, T0,
                steps=args.wm_steps,
                step_scale=args.step_scale,
                noise_std=args.noise_std,
                return_all=True
            )

            recon_loss = 0
            reward_loss = 0
            smooth_loss = 0
            # pose_step_loss = 0
            Lssim = SSIMLoss()
            prev_dT = None
            # edge_a = sobel_grad(frame)

            # # --- 方向一致性损失的累积器 ---
            # L_dir_to_target = 0.0
            # L_dir_pose      = 0.0

            # # 可调权重（也可放到 args 里）
            # lambda_dir_tgt  = getattr(args, "lambda_dir_tgt", 1.0)
            # lambda_dir_pose = getattr(args, "lambda_dir_pose", 1.0)
            # T_prev = T0
            # frame_prev = frame
            # dist_trans_prev = torch.norm(T_prev[:, :3] - dof[:, :3], dim=1, keepdim=True)  # (B,1)
            # dist_rot_prev   = torch.norm(T_prev[:, 3:] - dof[:, 3:], dim=1, keepdim=True)  # (B,1)
            # dist_prev       = dist_trans_prev + dist_rot_prev                              # (B,1)
            # ncc_prev = safe_ncc(frame_prev, frame).unsqueeze(-1)  # (B,1)
            for step_i, ((T_t, frame_t, r_pred, sl_dec, dT), dir_pack) in enumerate(zip(traj, dir_info)):
                # dzs, dir_tgt, dzp_in_z = dir_pack  # (B, z_dim), (B, z_dim), (B, z_dim or pose_dim)
                # # 方向对齐到当前动作方向
                # L_dir_pose += cos_align_loss(dzs, dzp_in_z)
                # # 方向对齐到目标
                # # L_dir_to_target += cos_align_loss(dzs, dir_tgt)
                # p = F.softmax(dzs, dim=-1)
                # q = F.softmax(dir_tgt, dim=-1)
                # L_dir_to_target += (F.kl_div(p.log(), q, reduction='batchmean') + F.kl_div(q.log(), p, reduction='batchmean'))
                # direction
                dT_gt = dof - T0
                dT_pred_trans = normalize_vec(dT[:, :3])
                dT_gt_trans   = normalize_vec(dT_gt[:, :3])
                dT_pred_rot = normalize_vec(dT[:, 3:])
                dT_gt_rot   = normalize_vec(dT_gt[:, 3:])
                # direction reward
                cos_trans = (dT_pred_trans * dT_gt_trans).sum(dim=-1, keepdim=True)  # (B,1)
                cos_rot   = (dT_pred_rot   * dT_gt_rot).sum(dim=-1, keepdim=True) 
                reward_gt_pair = torch.cat([cos_trans, cos_rot], dim=1)  # (B,2)
                reward_loss += F.mse_loss(r_pred, reward_gt_pair)

                # pose_step_loss += (1.0 * F.mse_loss(dT_pred_trans, dT_gt_trans)
                #                 + 1.0 * F.mse_loss(dT_pred_rot, dT_gt_rot))
                             

                # slice reconstruction
                recon_loss += (1.0*F.l1_loss(sl_dec, frame_t)
                            + 1.0*Lssim(sl_dec, frame_t))

                # reward
                # gt_ncc = safe_ncc(frame, frame_t).unsqueeze(-1)
                # edge_b = sobel_grad(frame_t)
                # gt_gncc = safe_ncc(edge_a, edge_b).unsqueeze(-1)
                # gt_pair = torch.cat([gt_ncc, gt_gncc], dim=1)  # (B,2)
                # reward_loss += F.l1_loss(r_pred, gt_pair)

                # ---------- 1) pose-space distance reward ----------
                # 当前这一步的 pose 距离（平移 + 角度），越小越好
                # dist_trans_now = torch.norm(T_t[:, :3] - dof[:, :3], dim=1, keepdim=True)  # (B,1)
                # dist_rot_now   = torch.norm(T_t[:, 3:] - dof[:, 3:], dim=1, keepdim=True)  # (B,1)
                # dist_now       = dist_trans_now + dist_rot_now 
                # ncc_now = safe_ncc(frame, frame_t).unsqueeze(-1)  # (B,1)
                # dist_trans_reward = dist_trans_prev - dist_trans_now  # (B,1)
                # dist_rot_reward   = dist_rot_prev   - dist_rot_now    # (B,1)
                # ---- 增益型 distance reward & NCC reward ----
                # pose_reward > 0 表示距离变小（有进步）
                # pose_reward = dist_prev - dist_now        # (B,1)
                # ncc_reward > 0 表示相似度变高（有进步）
                # ncc_reward  = ncc_now - ncc_prev          # (B,1)

                # pose_reward = 1.0 * pose_reward
                # ncc_reward  = 1.0 * ncc_reward

                # reward_gt_pair = torch.cat([dist_trans_reward, dist_rot_reward], dim=1)
                # reward_loss += F.mse_loss(r_pred, reward_gt_pair)

                # dist_prev = dist_now.detach()
                # dist_trans_prev = dist_trans_now.detach()
                # dist_rot_prev   = dist_rot_now.detach()
                # ncc_prev  = ncc_now.detach()
                # T_prev    = T_t.detach()
                # frame_prev = frame_t.detach()

                # dT smoothness
                if prev_dT is not None:
                    smooth_loss += F.mse_loss(dT, prev_dT)
                prev_dT = dT

            # 平均化
            recon_loss      /= args.wm_steps
            reward_loss     /= args.wm_steps
            smooth_loss     /= args.wm_steps
            # pose_step_loss  /= args.wm_steps
            # L_dir_to_target /= args.wm_steps
            # L_dir_pose      /= args.wm_steps

            # 末步监督
            final_T = T_t
            final_frame = frame_t
            # l_trans = Lsml1(final_T[:, :3], dof[:, :3])
            # l_rot   = Lsml2(final_T[:, 3:], dof[:, 3:])
            # lssim = Lssim(frame, final_frame)

            # === 汇总总损失 ===
            # dir_loss = (lambda_dir_tgt * L_dir_to_target
            #             + lambda_dir_pose * L_dir_pose)

            loss = (1.0*reward_loss + 1.0*smooth_loss + 1.0*recon_loss)


            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_all.update(loss.item(), vol.size(0))

            print(f"[Train] Epoch {epoch} iter {it} | "
                  f"Loss {loss.item():.4f} | "
                    # NCC {lncc.item():.4f} | "
                  f"Reward {reward_loss.item():.4f} | "
                  f"Smooth {smooth_loss.item():.4f} | "
                  f"Recon {recon_loss.item():.4f} | "
                #   f"PoseStep {pose_step_loss.item():.4f} | "
                #   f"Rotation {l_rot.item():.4f} | "
                #   f"Translation {l_trans.item():.4f} | "
                #   f"SSIM {(lssim).item():.4f} "
                #   f"DirectionLoss {dir_loss.item():.4f}"
                  )

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')

        # ===== validation =====
        val_DistErr, val_NCC, val_ParaErr = run_eval(
            val_loader, epoch, model, args.wm_steps, args.step_scale, split_name='val'
        )

        print(epoch, val_DistErr, val_NCC, val_ParaErr, file=f, flush=True)
        if epoch <= 5:
            scheduler_warm.step()
            pass 
        else:            
            scheduler.step(val_ParaErr + val_DistErr)

        if val_ParaErr < best_ParaErr:
            best_ParaErr = val_ParaErr
            torch.save({'state_dict': model.state_dict()},
                       best_path)
            print(f">> New best @ epoch {epoch}: ParaErr={best_ParaErr:.6f}")

    # ===== Final test =====
    ckpt = torch.load(best_path, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    test_DistErr, test_NCC, test_ParaErr = run_eval(
        test_loader, epoch='final', model=model,
        steps=args.wm_steps, step_scale=args.step_scale, split_name='test'
    )

    print(f'>>> TEST RESULTS <<<')
    print(f'DistErr {test_DistErr:.6f}')
    print(f'NCC     {test_NCC:.6f}')
    print(f'ParaErr {test_ParaErr:.6f}')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    print('Using GPU:', torch.cuda.get_device_name(0))
    main()
