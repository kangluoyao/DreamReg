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
from models.baseline_v3 import EUReg_FRT


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

def run_eval(loader, epoch, model, split_name='val'):
    Lcorner = losses.CornerDistLoss()
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

            pred_dof, sampled_frame = model(vol, frame)

            # 指标（与原先一致）
            param_err = Lsml1(pred_dof[:, :3], dof[:, :3]).item() + Lsml1(pred_dof[:, 3:], dof[:, 3:]).item()
            dist_err  = Lcorner(pred_dof, dof).item()
            ncc       = losses.normalized_cross_correlation(frame, sampled_frame).item()

            bs = vol.size(0)
            eval_ParaErr.update(param_err, bs)
            eval_DistErr.update(dist_err,  bs)
            eval_NCC.update(ncc,           bs)

    print(f'[{split_name}] Epoch {epoch}  DistErr: {eval_DistErr.avg:.6f}, NCC: {eval_NCC.avg:.6f}, ParaErr: {eval_ParaErr.avg:.6f}')
    return eval_DistErr.avg, eval_NCC.avg, eval_ParaErr.avg


GPU_iden = 0

torch.cuda.init()
def main():
    batch_size = 32

    train_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/training/'
    val_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/validation/'
    test_dir = '/Media_HDD/lykang/dataset/CAMUS_public/splits_10*10*10/testing/'
    weights = [1, 1, 1, 1]  # loss weights
    dim=16
    layer_nums = 4
    lr = 0.0005
    scaler = [5,5,5,5,5,5]
    save_dir = 'Baseline_v3/'
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
    max_epoch = 300
    cont_training = False

    '''
    Initialize model
    '''
    # model = EUReg_FRT(img_size, slice_size, dim=dim)

    model = EUReg_FRT(img_size, dim=dim)
    model.cuda()

    # reg_model_bilin = FrameRigidTransformer(img_size, [1, *img_size[1:]],  'bilinear')
    # reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        # epoch_start = 384
        model_dir = 'experiments/CAMUS2/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir+'ParaErr1.1695_DistErr2.5609_epoch585.pth.tar')['state_dict']
        model.load_state_dict(best_model)
        # print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''

    train_set = TrainDataset(glob.glob(train_dir + '*.pkl'), slice_size, scaler)
    val_set = TestDataset(glob.glob(val_dir + '*.pkl'))
    test_set = TestDataset(glob.glob(test_dir + '*.pkl'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr)
    scheduler_warm = lr_scheduler.StepLR(optimizer,step_size=1, gamma=1.2)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)


    Lmse = nn.MSELoss()
    Lcorner = losses.CornerDistLoss()
    Lsml1 = nn.SmoothL1Loss()
    Llncc = losses.LNCCLoss()
    Lflow = losses.FlowLoss(img_size,
                            vol_shape=[s//2**(layer_nums-1) for s in img_size],
                            slice_size=[s//2**(layer_nums-1) for s in slice_size]).cuda()

    it = 0
    for epoch in range(epoch_start, max_epoch):
        # print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        model.train()
        # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
        for data in train_loader:
            idx += 1
            it += 1
            # data = [t.cuda() for t in data]
            vol = data[0].cuda()
            frame = data[1].cuda()
            dof = data[2].cuda()

            # eureg
            pred_dof, sampled_frame = model(vol, frame)

            l_trans = Lsml1(pred_dof[:, :3], dof[:, :3])
            l_rot = Lsml1(pred_dof[:, 3:], dof[:, 3:])
            l_sim = Llncc(frame, sampled_frame)
            loss = l_trans * weights[0] + l_rot * weights[1] + l_sim * weights[3]
            dist_err = Lcorner(pred_dof, dof)

            loss_all.update(loss.item(), vol.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')
        loss_all.reset()


        '''        
        Validation
        '''
        val_DistErr, val_NCC, val_ParaErr = run_eval(val_loader, epoch, split_name='val', model=model)

        try:
            print(epoch, val_DistErr, val_NCC, val_ParaErr, file=f, flush=True)
        except Exception:
            pass

        if epoch <= 10:
            scheduler_warm.step() 
        else:            
            scheduler.step(val_ParaErr + val_DistErr)


        if val_ParaErr < best_ParaErr:
            best_ParaErr = val_ParaErr
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_ParaErr': best_ParaErr,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/CAMUS2/' + save_dir,
                filename='best_model.pth.tar')
            print(f'>> New best @ epoch {epoch}: ParaErr={best_ParaErr:.6f}  (saved: {best_path})')
            
        

        if (epoch+1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_DistErr': best_ParaErr,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/CAMUS2/' + save_dir,
                filename='ParaErr{:.4f}_DistErr{:.4f}_epoch{}.pth.tar'.format(val_ParaErr, val_DistErr,
                                                                                epoch))
    

    ckpt = torch.load(best_path, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

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
