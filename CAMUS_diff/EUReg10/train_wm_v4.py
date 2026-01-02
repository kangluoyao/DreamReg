import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import utils
import losses as pre_losses
from losses import SSIMLoss
import losses

from data.datasets import TrainDataset, TestDataset
from models.baseline_wm_v4 import EUReg_WM_Belief   # [+] 使用新的模型文件
from rewards import safe_ncc, sobel_grad
from rewards import soft_hog, hog_cosine_reward, normalize_vec

from slice_vae import SliceVAE


# ----------------------------
# Utils
# ----------------------------

def load_pretrained_ae_to_wm(
    wm_model,
    vae_ckpt_path,
    device="cuda",
    load_enc2d=True,
    load_dec192=True,
    in_channel=1,
    first_channel=8,
    z_dim=256,
    freeze_after_load_enc2d=False,
    freeze_after_load_dec192=False
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[load_pretrained_ae_to_wm] device = {device}")

    vae = SliceVAE(
        in_channel=in_channel,
        first_channel=first_channel,
        z_dim=z_dim
    ).to(device)

    print(f"[load_pretrained_ae_to_wm] Loading VAE ckpt from {vae_ckpt_path}")
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print("[load_pretrained_ae_to_wm] VAE loaded. "
          f"missing={len(missing)}, unexpected={len(unexpected)}")

    if load_enc2d:
        wm_model.encoder2d.load_state_dict(vae.encoder2d.state_dict())
        print("[load_pretrained_ae_to_wm] encoder2d weights loaded into world model.")
        if freeze_after_load_enc2d:
            for p in wm_model.encoder2d.parameters():
                p.requires_grad = False
            print(">> encoder2d frozen.")

    if load_dec192:
        wm_model.dec192.load_state_dict(vae.dec192.state_dict())
        print("[load_pretrained_ae_to_wm] dec192 weights loaded into world model.")
        if freeze_after_load_dec192:
            for p in wm_model.dec192.parameters():
                p.requires_grad = False
            print(">> dec192 frozen.")

    print("[load_pretrained_ae_to_wm] Done.")


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


# ----------------------------
# Eval
# ----------------------------

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


# ----------------------------
# Main Training Loop (Dreamer-RL)
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_steps', type=int, default=10)
    parser.add_argument('--step_scale', type=float, default=0.6)
    parser.add_argument('--noise_std', type=float, default=0.00)
    args, _ = parser.parse_known_args()
    print("start training world model with steps:", args.wm_steps,
          " step_scale:", args.step_scale, " noise_std:", args.noise_std)

    batch_size = 64
    train_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/training/'
    val_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/validation/'
    test_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/testing/'
    img_size = (32, 192, 192)
    slice_size = (192, 192)
    lr = 0.0001

    save_dir = 'wm_v3_dreamer_rl/'   # [+] 新实验目录
    if not os.path.exists('experiments/CAMUS2/' + save_dir):
        os.makedirs('experiments/CAMUS2/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    best_path = os.path.join('experiments/CAMUS2/', save_dir, 'best_model.pth.tar')
    best_ParaErr = float('inf')

    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/' + save_dir, 'losses' + ".txt"), "a")

    epoch_start = 0
    max_epoch = 200
    updated_lr = lr

    model = EUReg_WM_Belief(img_size).cuda()

    load_pretrained_ae_to_wm(
        wm_model=model,
        vae_ckpt_path="vae_ckpt/slice_vae_pretrained.pth",
        device="cuda",
        load_enc2d=True,
        load_dec192=True,
        in_channel=1,
        first_channel=8,
        z_dim=256,
        freeze_after_load_enc2d=False,
        freeze_after_load_dec192=False
    )

    train_set = TrainDataset(glob.glob(train_dir + '*.pkl'), slice_size, [10]*6)
    val_set   = TestDataset(glob.glob(val_dir + '*.pkl'))
    test_set  = TestDataset(glob.glob(test_dir + '*.pkl'))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr)
    scheduler_warm = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.2)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

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
            step_dT = dT_gt / float(L)

            step_idx = torch.arange(1, L+1, device=device).view(1, L, 1)
            T_seq = T0.unsqueeze(1) + step_idx * step_dT.unsqueeze(1)
            a_seq = step_dT.unsqueeze(1).expand(-1, L, -1)

            # ================================
            # 1) World Model Observe Rollout
            # ================================
            wm_out = model.wm_observe_rollout(
                vol    = vol,
                goal_sl= frame,
                T_seq  = T_seq,
                a_seq  = a_seq,
                h0     = None
            )

            kl_loss     = wm_out["kl_loss"]
            cur_seq     = wm_out["cur_seq"]
            sl_pred_seq = wm_out["sl_pred_seq"]
            r_pred_seq  = wm_out["r_pred_seq"]
            h_seq       = wm_out["h_seq"]
            z_seq       = wm_out["z_seq"]
            z_goal      = wm_out["z_goal"]
            zv          = wm_out["zv"]

            recon_loss = 0.0
            reward_loss = 0.0

            # ---------- 初始化 progress reward ----------
            dist_prev = losses.CornerDistLoss()(T0, dof).detach()
            ncc_prev  = losses.normalized_cross_correlation(
                cur_seq[:, 0], frame
            ).detach()
            dist_scale = 5.0

            for t in range(L):
                cur_t   = cur_seq[:, t]
                sl_dec  = sl_pred_seq[:, t]
                r_pred  = r_pred_seq[:, t]

                recon_loss_t = F.l1_loss(sl_dec, cur_t) + SSIM(sl_dec, cur_t)
                recon_loss += recon_loss_t

                dist_now = losses.CornerDistLoss()(T_seq[:, t], dof)
                ncc_now  = losses.normalized_cross_correlation(cur_t, frame)

                r_dist = (dist_prev - dist_now) / dist_scale
                r_ncc  = (ncc_now - ncc_prev)

                reward_gt_pair = torch.stack([r_dist, r_ncc], dim=-1)
                reward_loss_t = F.mse_loss(r_pred, reward_gt_pair)
                reward_loss += reward_loss_t

                dist_prev = dist_now.detach()
                ncc_prev  = ncc_now.detach()

            recon_loss  = recon_loss  / float(L)
            reward_loss = reward_loss / float(L)

            # ====================================
            # 2) Dreamer-style Actor-Critic in Latent
            # ====================================
            H = 5             # [+] imagination horizon
            discount = 0.95   # [+] discount factor

            h_start = h_seq[:, -1].detach()   # [+] detach to freeze world model
            z_start = z_seq[:, -1].detach()
            z_goal_det = z_goal.detach()
            zv_det     = zv.detach()

            h_roll = h_start
            z_roll = z_start
            r_list = []

            with torch.no_grad():            # [+] imagination: 不让 actor_loss 改 world model
                for k in range(H):
                    pol_inp = torch.cat([h_roll, z_goal_det], dim=-1)
                    a_k = model.delta_head(pol_inp)

                    step = model.wm_imagine_step(z_goal_det, zv_det, a_k, h_roll)
                    h_roll = step["h_t"]
                    z_roll = step["z_t"]
                    r_kvec = step["r_pred"]          # (B,2)
                    r_k = r_kvec[:, 0] + 1.0 * r_kvec[:, 1]
                    r_list.append(r_k)

                G = torch.zeros_like(r_list[0])
                for k in reversed(range(H)):
                    G = r_list[k] + discount * G     # (B,)

            # Value loss：V(h_start,z_start,z_goal) ≈ G
            v_inp   = torch.cat([h_start, z_goal_det, z_start], dim=-1)
            V_pred  = model.value_head(v_inp).squeeze(-1)
            value_loss = F.mse_loss(V_pred, G.detach())   # [+]

            # Actor loss：maximize V_pred
            actor_inp = torch.cat([h_start, z_goal_det], dim=-1)
            dT_actor  = model.delta_head(actor_inp)
            # 用 actor_inp 重新算 value（允许对 value_head 也有一点影响，简单起见）
            V_actor_inp = torch.cat([h_start, z_goal_det, z_start], dim=-1)
            V_actor = model.value_head(V_actor_inp).squeeze(-1)
            actor_loss = - V_actor.mean()                 # [+]

            # ================================
            # 3) Total loss (no BC act_loss)
            # ================================
            beta_kl   = 1.0
            w_recon   = 1.0
            w_reward  = 1.0
            w_value   = 0.5     # [+]
            w_actor   = 1.0     # [+]

            loss = (beta_kl * kl_loss
                    + w_recon  * recon_loss
                    + w_reward * reward_loss
                    + w_value  * value_loss
                    + w_actor  * actor_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), vol.size(0))

            print(f"[Train] Epoch {epoch} iter {it} | "
                  f"Loss {loss.item():.4f} | "
                  f"KL {kl_loss.item():.4f} | "
                  f"Recon {recon_loss.item():.4f} | "
                  f"Reward {reward_loss.item():.4f} | "
                  f"Value {value_loss.item():.4f} | "
                  f"Actor {actor_loss.item():.4f} | "
                  )

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg),
              file=f, end=' ')

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
