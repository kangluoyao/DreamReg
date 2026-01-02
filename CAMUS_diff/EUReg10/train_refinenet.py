import glob
import losses
import os
import random
import sys
import utils
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch import optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from data.datasets import TrainDataset, TestDataset
from models.refine_net import ProbeAdjustPolicy

from expert import ncc_score, expert_step_ncc
import argparse
import torch.nn.functional as F

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
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

def run_eval(loader, epoch, model, split_name='val',
             steps=4, step_scale=0.2):
    """
    model: policy 网络 (ProbeAdjustPolicy)
    steps, step_scale: 需要与你训练时一致
    """
    Lcorner = losses.CornerDistLoss()
    Lsml1   = nn.SmoothL1Loss()

    eval_DistErr = utils.AverageMeter()
    eval_NCC     = utils.AverageMeter()
    eval_ParaErr = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for data in loader:
            vol   = data[0].cuda(non_blocking=True)   # (B,1,D,H,W)
            frame = data[1].cuda(non_blocking=True)   # (B,1,H,W)
            dof   = data[2].cuda(non_blocking=True)   # GT pose (B,6)

            # ---------------------------------------------------------
            # 1) Policy rollout：根据当前 slice 迭代 K 步 refinement
            # ---------------------------------------------------------
            T0 = torch.zeros_like(dof)
            pred_T, pred_frame = model(
                vol, frame,
                T0=T0,
                steps=steps,
                step_scale=step_scale,
                return_all=False
            )
            # pred_T: (B,6)
            # pred_frame: (B,1,H,W)

            # ---------------------------------------------------------
            # 2) 指标计算（与原先保持一致）
            # ---------------------------------------------------------

            # parameter error (translation/rotation)
            param_err = (
                Lsml1(pred_T[:, :3], dof[:, :3]).item()
              + Lsml1(pred_T[:, 3:], dof[:, 3:]).item()
            )

            # corner distance
            dist_err  = Lcorner(pred_T, dof).item()

            # NCC between predicted slice & target slice
            ncc = losses.normalized_cross_correlation(frame, pred_frame).item()

            # accumulate
            bs = vol.size(0)
            eval_ParaErr.update(param_err, bs)
            eval_DistErr.update(dist_err,  bs)
            eval_NCC.update(ncc,           bs)

    print(f'[{split_name}] Epoch {epoch}  DistErr: {eval_DistErr.avg:.6f}, NCC: {eval_NCC.avg:.6f}, ParaErr: {eval_ParaErr.avg:.6f}')
    return eval_DistErr.avg, eval_NCC.avg, eval_ParaErr.avg



GPU_iden = 0

torch.cuda.init()
def main():
    args = argparse.Namespace()
    args.imit_steps = 4       # 每个样本模仿 expert 的步数
    args.step_scale = 3.0     # policy 的每一步动作尺度 (ΔT_pred * step_scale)
    args.lr = 1e-4            # policy 的学习率
    args.batch_size = 16       # 根据你显存来改
    args.max_epoch = 200      # 可改大一些，imitation 更稳定
    args.expert_trans_step = 0.05
    args.expert_rot_step   = 0.05

    train_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/training/'
    val_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/validation/'
    test_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/testing/'

    layer_nums = 4
    scaler = [10,10,10,10,10,10]
    save_dir = 'refinenet_v1/'
    img_size = (32, 192, 192)
    slice_size = (192,192)
    if not os.path.exists('experiments/CAMUS2/' + save_dir):
        os.makedirs('experiments/CAMUS2/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    best_path = os.path.join('experiments/CAMUS2/', save_dir, 'best_model.pth.tar')
    best_ParaErr = float('inf')

    sys.stdout = Logger('logs/' + save_dir)

    f = open(os.path.join('logs/' + save_dir, 'losses' + ".txt"), "a")

    test_log_path = os.path.join('logs/' + save_dir, 'test_results' + ".txt")
    os.makedirs(os.path.dirname(test_log_path), exist_ok=True)
    device = torch.device("cuda")

    epoch_start = 0
    '''
    Initialize training
    '''

    train_set = TrainDataset(glob.glob(train_dir + '*.pkl'), slice_size, scaler)
    val_set = TestDataset(glob.glob(val_dir + '*.pkl'))
    test_set = TestDataset(glob.glob(test_dir + '*.pkl'))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

  # ====== Optimizer / Scheduler / Losses ======
    Lmse  = nn.MSELoss()
    Lcorner = losses.CornerDistLoss()
    Lsml1   = nn.SmoothL1Loss()
    Llncc   = losses.LNCCLoss()
    Lflow   = losses.FlowLoss(
        img_size,
        vol_shape=[s // 2**(layer_nums-1) for s in img_size],
        slice_size=[s // 2**(layer_nums-1) for s in slice_size]
    ).cuda()

    # 只训练 policy（医生式调节网络）
    policy = ProbeAdjustPolicy(img_size, in_channel=1, first_channel=8).cuda()

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler_warm = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.2)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)

    for epoch in range(epoch_start, args.max_epoch):
        policy.train()
        loss_all = utils.AverageMeter()

        for data in train_loader:
            vol   = data[0].cuda()   # (B,1,D,H,W)
            frame = data[1].cuda()   # target slice (B,1,H,W)
            dof   = data[2].cuda()   # GT pose (B,6)


            # 1) 初始化 T_t（老师 & 学生的起点）
            # 可以先从 0 开始，如果你希望更 realistic，可以用 dof + 噪声
            T_t = torch.zeros_like(dof)  # (B,6)
            # T_t = dof + torch.randn_like(dof) * torch.tensor([5,5,5, 5,5,5], device=device)

            steps      = args.imit_steps
            step_scale = args.step_scale

            imitation_loss = 0.0

            for k in range(steps):
                # 2) expert 给出下一步 T_next_expert（基于 NCC 的经典优化器）
                T_next = expert_step_ncc(
                    vol, frame, T_t,
                    transformer=policy.transformer,   # 直接用 policy 自带的 transformer
                    trans_step=args.expert_trans_step,
                    rot_step=args.expert_rot_step,
                    search_iters=1
                )
                dT_expert = T_next - T_t      # (B,6)，老师的动作

                # 3) policy 在当前状态上给出 ΔT_pred
                dT_pred, cur = policy.forward_step(vol, frame, T_t)

                # 4) imitation loss: 让 ΔT_pred ≈ ΔT_expert
                imitation_loss = imitation_loss + F.mse_loss(dT_pred, dT_expert)

                # 5) 更新 T_t 为 expert 的结果（行为克隆轨迹）
                T_t = T_next.detach()

            imitation_loss = imitation_loss / steps

            # 6) 可选：加一个终点 pose 监督，让 policy 自己 rollout 也靠近 GT
            final_T_policy, _ = policy(
                vol, frame,
                T0=torch.zeros_like(dof),
                steps=steps,
                step_scale=step_scale,
                return_all=False
            )
            l_trans = Lsml1(final_T_policy[:, :3], dof[:, :3])
            l_rot   = Lsml1(final_T_policy[:, 3:], dof[:, 3:])  # 或者 Lsml2，看你之前的习惯

            loss = imitation_loss + 0.5 * (l_trans + l_rot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), vol.size(0))

        print(f"Epoch {epoch} | loss {loss_all.avg:.4f}")

        # ===================== Validation =====================
        val_DistErr, val_NCC, val_ParaErr = run_eval(
            val_loader, epoch, split_name='val', model=policy
        )

        try:
            print(epoch, val_DistErr, val_NCC, val_ParaErr, file=f, flush=True)
        except Exception:
            pass

        # 学习率调度
        if epoch <= 10:
            scheduler_warm.step()
        else:
            scheduler.step(val_ParaErr + val_DistErr)

        # ====== 保存 best checkpoint（保存 policy） ======
        if val_ParaErr < best_ParaErr:
            best_ParaErr = val_ParaErr
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': policy.state_dict(),
                'best_ParaErr': best_ParaErr,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/CAMUS2/' + save_dir,
            filename='best_model.pth.tar')
            print(f'>> New best @ epoch {epoch}: ParaErr={best_ParaErr:.6f}')

        # 每 10 个 epoch 存一次
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': policy.state_dict(),
                'best_ParaErr': best_ParaErr,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/CAMUS2/' + save_dir,
            filename='ParaErr{:.4f}_DistErr{:.4f}_epoch{}.pth.tar'.format(
                val_ParaErr, val_DistErr, epoch)
            )

    # 训练结束后，加载 best policy
    ckpt = torch.load(best_path, map_location='cuda')
    policy.load_state_dict(ckpt['state_dict'], strict=True)
    policy.eval()


    # 跑测试
    test_DistErr, test_NCC, test_ParaErr = run_eval(test_loader, epoch='final', split_name='test', model=model)

    # 保存测试结果到指定文件
    with open(test_log_path, 'w') as tf:
        tf.write(f'Best from epoch: {ckpt.get("epoch","unknown")}\n')
        tf.write(f'TEST  DistErr: {test_DistErr:.6f}\n')
        tf.write(f'TEST  NCC    : {test_NCC:.6f}\n')
        tf.write(f'TEST  ParaErr: {test_ParaErr:.6f}\n')
    print(f'>> Test results saved to: {test_log_path}')



def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=20):
    torch.save(state, save_dir + filename)
    # model_lists = natsorted(glob.glob(save_dir + '*'))
    # while len(model_lists) > max_model_num:
    #     os.remove(model_lists[-1])
    #     model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    torch.multiprocessing.set_start_method('spawn')
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
