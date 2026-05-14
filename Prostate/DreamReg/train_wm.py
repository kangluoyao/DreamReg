import glob
import losses as pre_losses
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import sys
import utils
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from data.datasets import TrainDataset, TestDataset
from models.baseline_wm import EUReg_WM_Belief
import torch.nn.functional as F
from rewards import normalize_vec
from losses import SSIMLoss


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
    dir_u = unit_vec(dir_to_target, eps)
    proj = (delta * dir_u).sum(dim=1)
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

same_seeds(2026)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def run_eval(loader, epoch, model, steps, step_scale, split_name='val'):
    Lcorner = pre_losses.CornerDistLoss()
    Lsml1   = nn.SmoothL1Loss()

    eval_DistErr = utils.AverageMeter()
    eval_NCC     = utils.AverageMeter()
    eval_ParaErr = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for data in loader:
            vol   = data[0].cuda(non_blocking=True)
            frame = data[1].cuda(non_blocking=True)
            dof   = data[2].cuda(non_blocking=True)

            T0 = torch.zeros(dof.size(0), 6,
                             device=vol.device, dtype=vol.dtype)

            pred_dof, sampled_frame = model(
                vol, frame, T0,
                steps=steps,
                step_scale=step_scale,
                return_all=False
            )

            param_err = (
                Lsml1(pred_dof[:, :3], dof[:, :3]).item()
                + Lsml1(pred_dof[:, 3:], dof[:, 3:]).item()
            )

            dist_err = Lcorner(pred_dof, dof).item()

            ncc = pre_losses.normalized_cross_correlation(
                frame, sampled_frame
            ).item()

            bs = vol.size(0)
            eval_ParaErr.update(param_err, bs)
            eval_DistErr.update(dist_err,  bs)
            eval_NCC.update(ncc,           bs)

    print(f'[{split_name}] Epoch {epoch}  DistErr: {eval_DistErr.avg:.6f}, '
          f'NCC: {eval_NCC.avg:.6f}, ParaErr: {eval_ParaErr.avg:.6f}')

    return eval_DistErr.avg, eval_NCC.avg, eval_ParaErr.avg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_steps', type=int, default=5)
    parser.add_argument('--step_scale', type=float, default=1.0)
    parser.add_argument('--noise_std', type=float, default=2)
    args, _ = parser.parse_known_args()
    print("start training world model with steps:", args.wm_steps,
          " step_scale:", args.step_scale, " noise_std:", args.noise_std)

    batch_size = 64
    train_dir = '/path/to/training_set/'
    val_dir = '/path/to/validation_set/'
    test_dir = '/path/to/testing_set/'
    img_size = (64, 64, 64)
    slice_size = (64, 64)
    dim=16
    lr = 0.00001

    save_dir = 'save_dir/'
    if not os.path.exists('experiments/Prostate/' + save_dir):
        os.makedirs('experiments/Prostate/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    best_path = os.path.join('experiments/Prostate/', save_dir, 'best_model.pth.tar')
    best_ParaErr = float('inf')

    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/' + save_dir, 'losses' + ".txt"), "a")

    epoch_start = 0
    max_epoch = 100
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
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)
    
    it = 0
    SSIM = SSIMLoss()

    for epoch in range(epoch_start, max_epoch):
        loss_all = utils.AverageMeter()
        model.train()

        for data in train_loader:
            it += 1
            vol   = data[0].cuda()
            frame = data[1].cuda()
            dof   = data[2].cuda()

            B = vol.size(0)
            device = vol.device

            T0 = torch.zeros(B, 6, device=device)

            L = args.wm_steps
            dT_gt = dof - T0
            
            step_dT_ideal = dT_gt / float(L)

            step_idx = torch.arange(1, L+1, device=device).view(1, L, 1)
            T_seq_ideal = T0.unsqueeze(1) + step_idx * step_dT_ideal.unsqueeze(1)

            noise_level = args.noise_std if args.noise_std > 0 else 1.0
            
            noise = torch.randn_like(T_seq_ideal) * noise_level
            
            noise[:, -1, :] *= 0.1 
            
            T_seq = T_seq_ideal + noise

            T_seq_full = torch.cat([T0.unsqueeze(1), T_seq], dim=1) # (B, L+1, 6)
            
            a_seq = T_seq_full[:, 1:] - T_seq_full[:, :-1]          # (B, L, 6)

            # ============================================================
            # 1) World model observe rollout (Dreamer-style)
            # ============================================================
            wm_out = model.wm_observe_rollout(
                vol    = vol,
                goal_sl= frame,
                T_seq  = T_seq,
                a_seq  = a_seq,
                h0     = None
            )

            kl_loss     = wm_out["kl_loss"]        # scalar
            cur_seq     = wm_out["cur_seq"]        # (B,L,1,H,W)
            sl_pred_seq = wm_out["sl_pred_seq"]    # (B,L,1,H,W)
            # r_pred_seq  = wm_out["r_pred_seq"]     # (B,L,2)
            h_seq       = wm_out["h_seq"]          # (B,L,h_dim)
            z_goal      = wm_out["z_goal"]         # (B,z_dim)
            # h_last      = wm_out["h_last"]         # (B,h_dim)
            zv          = wm_out["zv"]               # (B,L,z_dim)


            # ============================================================
            # 2) Recon loss: L1 + SSIM
            # ============================================================
            recon_loss = 0.0
            reward_loss = 0.0
            act_loss = 0.0

            T = T0
            for t in range(L):
                cur_t   = cur_seq[:, t]          # (B,1,H,W)
                sl_dec  = sl_pred_seq[:, t]      # (B,1,H,W)
                h_t     = h_seq[:, t]            # (B,h_dim)
                a_t     = a_seq[:, t]            # (B,6)
                zv_t    = zv  

                # ---------- recon ----------
                recon_loss_t = F.l1_loss(sl_dec, cur_t) + SSIM(sl_dec, cur_t)
                recon_loss += recon_loss_t

                # ---------- policy imitation ----------
                pol_inp = torch.cat([h_t, z_goal], dim=-1)      # (B,h_dim+z_dim)
                T_prev  = T
                dT_hat  = model.delta_head(pol_inp) * args.step_scale  # (B,6)
                T = T + dT_hat

                act_loss_t = F.mse_loss(dT_hat, a_t)
                act_loss += act_loss_t

                # ===================== reward (aligned to policy action) =====================
                dir_to_target = dof - T_prev
                step_vec = dT_hat

                step_trans = step_vec[:, :3]
                step_rot   = step_vec[:, 3:]
                dir_trans  = dir_to_target[:, :3]
                dir_rot    = dir_to_target[:, 3:]

                step_trans_u = normalize_vec(step_trans)
                dir_trans_u  = normalize_vec(dir_trans)
                step_rot_u   = normalize_vec(step_rot)
                dir_rot_u    = normalize_vec(dir_rot)

                cos_trans = (step_trans_u * dir_trans_u).sum(dim=-1, keepdim=True)  # (B,1)
                cos_rot   = (step_rot_u   * dir_rot_u  ).sum(dim=-1, keepdim=True)  # (B,1)
                reward_gt_pair = torch.cat([cos_trans, cos_rot], dim=-1)            # (B,2)

                wm_hat = model.wm_imagine_step(goal_z=z_goal, zv=zv_t, T_t=T_prev, a_t=dT_hat, h_prev=h_t)
                r_pred_hat = wm_hat["r_pred"]  # (B,2)

                reward_loss_t = F.mse_loss(r_pred_hat, reward_gt_pair)
                reward_loss += reward_loss_t


            pose_loss = F.smooth_l1_loss(T[:, :3], dof[:, :3]) + F.smooth_l1_loss(T[:, 3:], dof[:, 3:])
            recon_loss  = recon_loss  / float(L)
            reward_loss = reward_loss / float(L)
            act_loss    = act_loss    / float(L)

            beta_kl   = 1.0
            w_recon   = 1.0
            w_reward  = 1.0
            w_act     = 0.0
            w_pose    = 1.0
            # w_sim     = 1.0

            loss = (beta_kl * kl_loss
                    + w_recon  * recon_loss
                    + w_reward * reward_loss
                    + w_act    * act_loss
                    + w_pose   * pose_loss
                    # + w_sim    * sim_loss
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), vol.size(0))

            print(f"[Train] Epoch {epoch} iter {it} | "
                f"Loss {loss.item():.4f} | "
                f"KL {kl_loss.item():.4f} | "
                f"Recon {recon_loss.item():.4f} | "
                f"Reward {reward_loss.item():.4f} | "
                f"Act {act_loss.item():.4f} | "
                f"Pose {pose_loss.item():.4f} | "
                # f"SSIM {sim_loss.item():.4f}"
                )

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')

        val_DistErr, val_NCC, val_ParaErr = run_eval(
            val_loader, epoch, model,
            steps=args.wm_steps, step_scale=args.step_scale, split_name='val'
        )

        print(epoch, val_DistErr, val_NCC, val_ParaErr, file=f, flush=True)
        if epoch <= 5:
            scheduler_warm.step()
        else:
            scheduler.step(val_ParaErr + val_DistErr)

        if val_ParaErr < best_ParaErr:
            best_ParaErr = val_ParaErr
            torch.save({'state_dict': model.state_dict()}, best_path)
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
    print('Using GPU:', torch.cuda.get_device_name(0))
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")



    print("=== ENV ===")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<None>"))

    print("\n=== CUDA devices ===")
    print("torch.cuda.is_available() ->", torch.cuda.is_available())
    print("device_count ->", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} ->", torch.cuda.get_device_name(i))

    print("\n=== Default device test ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Default torch.device('cuda') ->", dev)

    x = torch.randn(1, device=dev)
    print("Tensor device:", x.device)
    main()
