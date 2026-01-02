import math
import torch
import torch.nn.functional as F
import torchgeometry as tgm
from torch import nn
from torch.distributions.normal import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentTransformer(nn.Module):
    def __init__(self, dim, nhead=8, mlp=512, layers=6):
        super().__init__()
        enc = []
        for _ in range(layers):
            enc += [
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, nhead, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))
            ]
        self.layers = nn.ModuleList(enc)
        self.mem_proj_in  = nn.Linear(dim, dim)
        self.mem_proj_out = nn.Linear(dim, dim)

    def forward(self, z, m):
        # z: [B,dim] 当前输入；m: [B,dim] 记忆
        x = torch.stack([m, z], dim=1)            # [B,2,dim] token: [mem, cur]
        for ln1, attn, ln2, ff in zip(self.layers[0::4], self.layers[1::4], self.layers[2::4], self.layers[3::4]):
            y,_ = attn(ln1(x), ln1(x), ln1(x))    # SA
            x = x + y
            x = x + ff(ln2(x))                    # FFN
        m_new = self.mem_proj_out(x[:,0])         # 取第0个token作为新记忆
        return m_new

class AttnPool2D(nn.Module):
    """Attention Pooling for 2D feature maps"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.q = nn.Parameter(torch.randn(1, hidden_dim))  # global query
        self.k_proj = nn.Linear(dim, hidden_dim)
        self.v_proj = nn.Linear(dim, hidden_dim)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, N, C], N=H*W
        k = self.k_proj(x)                    # [B, N, Hdim]
        v = self.v_proj(x)                    # [B, N, Hdim]
        q = self.q.expand(B, -1).unsqueeze(1) # [B,1,Hdim]
        attn = (q @ k.transpose(1,2)) / (k.size(-1)**0.5)  # [B,1,N]
        w = attn.softmax(-1)                  # attention weights
        out = (w @ v).squeeze(1)              # [B, Hdim]
        return out


class AttnPool3D(nn.Module):
    """Attention Pooling for 3D feature maps"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.q = nn.Parameter(torch.randn(1, hidden_dim))
        self.k_proj = nn.Linear(dim, hidden_dim)
        self.v_proj = nn.Linear(dim, hidden_dim)

    def forward(self, x):  # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, N, C], N=D*H*W
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q.expand(B, -1).unsqueeze(1)
        attn = (q @ k.transpose(1,2)) / (k.size(-1)**0.5)
        w = attn.softmax(-1)
        out = (w @ v).squeeze(1)              # [B, Hdim]
        return out



# --------------------
# Basic Conv Blocks
# --------------------

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        return self.leakyrelu(self.norm(self.conv(x)))


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = nn.Identity()
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        return self.leakyrelu(self.norm(self.conv(x)))


# --------------------
# Encoder Networks
# --------------------

class Encoder2D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super().__init__()
        c = first_channel
        self.layer1 = nn.Sequential(ConvBlock2D(in_channel, c), ConvBlock2D(c, c))
        self.layer2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock2D(c, 2*c), ConvBlock2D(2*c, 2*c))
        self.layer3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock2D(2*c, 4*c), ConvBlock2D(4*c, 4*c))
        self.layer4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock2D(4*c, 8*c), ConvBlock2D(8*c, 8*c))

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x


class Encoder3D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super().__init__()
        c = first_channel
        self.layer1 = nn.Sequential(ConvBlock3D(in_channel, c), ConvBlock3D(c, c))
        self.layer2 = nn.Sequential(nn.MaxPool3d(2), ConvBlock3D(c, 2*c), ConvBlock3D(2*c, 2*c))
        self.layer3 = nn.Sequential(nn.MaxPool3d(2), ConvBlock3D(2*c, 4*c), ConvBlock3D(4*c, 4*c))
        self.layer4 = nn.Sequential(nn.MaxPool3d(2), ConvBlock3D(4*c, 8*c), ConvBlock3D(8*c, 8*c))

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x


# --------------------
# Rigid Slice Transformer
# --------------------

class FrameRigidTransformer(nn.Module):
    def __init__(self, slice_size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.slice_size = [1] + list(slice_size)
        vectors = [torch.linspace(-0.5*(s-1), 0.5*(s-1), steps=s) for s in self.slice_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack([grids[2], grids[1], grids[0], torch.ones_like(grids[0])], dim=0)
        grid = grid.view(4, -1).float()
        self.register_buffer('grid', grid)

    def dof2mat(self, dof):
        rad = tgm.deg2rad(dof[:, 3:])
        ai,aj,ak = rad[:,0], rad[:,1], rad[:,2]
        si,sj,sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci,cj,ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc,cs = ci*ck, ci*sk; sc,ss = si*ck, si*sk

        M = torch.eye(4, device=dof.device).repeat(dof.shape[0],1,1)
        M[:,0,0] = cj*ck; M[:,0,1] = sj*sc - cs; M[:,0,2] = sj*cc + ss
        M[:,1,0] = cj*sk; M[:,1,1] = sj*ss + cc; M[:,1,2] = sj*cs - sc
        M[:,2,0] = -sj;  M[:,2,1] = cj*si;     M[:,2,2] = cj*ci
        M[:, :3,3] = dof[:, :3]
        return M

    def forward(self, vol, dof):
        M = self.dof2mat(dof)
        new = torch.matmul(M, self.grid)[:, :3]
        vol_size = vol.shape[2:]
        for i in range(len(vol_size)):
            new[:, i] = 2*((new[:, i] + 0.5*(vol_size[2-i]-1))/(vol_size[2-i]-1) - 0.5)
        new = new.permute(0,2,1).view(vol.shape[0], *self.slice_size, 3)
        return F.grid_sample(vol, new, align_corners=True, mode=self.mode)


# --------------------
# Baseline EUReg for reference
# --------------------

class EUReg(nn.Module):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super().__init__()
        c = first_channel
        self.encoder3d = Encoder3D(in_channel, c)
        self.encoder2d = Encoder2D(in_channel, c)

        self.fn1 = nn.Sequential(
            nn.Linear(184320, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, vol, sl):
        Fv = self.encoder3d(vol).view(vol.size(0), -1)
        Fs = self.encoder2d(sl).view(sl.size(0), -1)
        return self.fn1(torch.cat([Fv,Fs], dim=1))


class EUReg_FRT(EUReg):
    def __init__(self, vol_shape, in_channel=1, first_channel=8):
        super().__init__(vol_shape, in_channel, first_channel)
        self.transformer = FrameRigidTransformer(vol_shape[1:])

    def forward(self, vol, sl):
        dof = super().forward(vol, sl)
        return dof, self.transformer(vol, dof).squeeze(2)


# --------------------
# Dreamer-Style World-Model Version ✅
# --------------------

class PoseEncoder(nn.Module):
    def __init__(self, in_dim=6, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    def forward(self, T): return self.net(T)


class EUReg_WM_Belief(nn.Module):
    """
    Dreamer-style rigid US registration model.
    Adds GRU world state + slice decoder + reward head.
    """
    def __init__(self, vol_shape, in_channel=1, first_channel=8, h_dim=256, z_dim=128):
        super().__init__()
        c = first_channel
        feat_dim = 8*c

        # Encoders
        self.encoder3d = Encoder3D(in_channel, c)
        self.encoder2d = Encoder2D(in_channel, c)
        self.pose_enc  = PoseEncoder(6, 64)

        self.proj_vol   = nn.Linear(2304, z_dim)
        self.proj_slice = nn.Linear(576, z_dim)

        self.proj_slice_dir = nn.Linear(z_dim, z_dim)

        # GRU latent state
        self.gru = nn.GRUCell(3*z_dim + 64, h_dim)

        # ΔT head
        self.delta_head = nn.Sequential(nn.Linear(h_dim,256), nn.ReLU(), nn.Linear(256,6))

        # Reward head
        self.reward_head = nn.Sequential(nn.Linear(h_dim,256), nn.ReLU(), nn.Linear(256,2))

        # self.dec192 = Dec192(h_dim)

        # Physics render
        self.transformer = FrameRigidTransformer(vol_shape[1:])

        # self.pose_proj = nn.Linear(64, z_dim)

        self.pool3 = AttnPool3D(feat_dim)   # 在 __init__ 里初始化
        self.pool2 = AttnPool2D(feat_dim)

        self.rnn = RecurrentTransformer(dim=h_dim, nhead=8, layers=4)
        self.inp = nn.Linear(z_dim*64*4, h_dim)


    def init_state(self, B, device): return torch.zeros(B,256,device=device)

    def _pool3(self,f): return f.mean(dim=[2,3,4])
    def _pool2(self,f): return f.mean(dim=[2,3])

    @staticmethod
    def angle_wrap_deg(x): return (x+180)%360-180


    def forward(self, vol, sl, pose, T0, steps=6, step_scale=0.4, noise_std=0.02,
                return_all=False, return_dir_info=True):
        device = vol.device
        B = vol.size(0)
        T = T0.clone()
        h = self.init_state(B, device)

        # Fv = self.pool3(self.encoder3d(vol))
        Fv = self.encoder3d(vol)
        B, C, Z, X, Y = Fv.shape
        Fv = Fv.view(B, C, Z*X*Y)

        Ft = self.encoder2d(sl)
        B, C, X, Y = Ft.shape
        Ft = Ft.view(B, C, X * Y)

        zv = self.proj_vol(Fv)
        zt = self.proj_slice(Ft)
        cur = self.transformer(vol, T).squeeze(2)

        traj = []
        dir_info = []  # ← 新增：每步存放 (dzs, dir_tgt, dzp_in_z or dzp)
        # zs_bank = 0

        # t=0 的编码
        # Fs_prev = self.pool2(self.encoder2d(cur))
        Fs_prev = self.encoder2d(cur)
        Fs_prev = Fs_prev.view(B, C, X * Y)
        zs_prev = self.proj_slice(Fs_prev)
        # zs_bank = zs_bank * 0.3 + zs_prev * 0.7
        zp_prev = self.pose_enc(T).unsqueeze(-1).repeat(1,1,zt.size(-1))

        for _ in range(steps):
            z = torch.cat([zv, zs_prev, zt, zp_prev], dim=1).view(B,-1)

            # h = self.gru(z, h)
            z = self.inp(z)
            h = self.rnn(z, h)

            # dT = torch.tanh(self.delta_head(h))
            dT = self.delta_head(h)
            # noise = torch.randn_like(dT) * noise_std
            T = T + dT * step_scale
            T[:, 3:] = self.angle_wrap_deg(T[:, 3:])

            reward  = self.reward_head(h)
            # sl_pred = self.dec192(h)

            cur = self.transformer(vol, T).squeeze(2)

            # Fs_new = self.pool2(self.encoder2d(cur))
            # zs_new = self.proj_slice(Fs_new)
            # zp_new = self.pose_enc(T)
            # 收集方向向量
            # dzs     = self.proj_slice_dir(zs_new) - self.proj_slice_dir(zs_prev)
            # dir_tgt = self.proj_slice_dir(zt) - self.proj_slice_dir(zs_prev)
            # dzp_in_z = self.pose_proj(zp_new) - self.pose_proj(zp_prev)

            # dir_info.append((dzs, dir_tgt, dzp_in_z))  # 只把向量交给训练端
            dir_info.append((None, None, None))  # 占位符
            # 状态推进
            # zs_prev = zs_new
            # zs_bank = zs_bank * 0.3 + zs_new * 0.7
            # zp_prev = zp_new

            traj.append((T, cur, reward, None, dT))

        if return_all:
            if return_dir_info:
                return T, cur, traj, dir_info
            else:
                return T, cur, traj
        else:
            return T, cur




class Dec192(nn.Module):
    """
    Lightweight decoder to (B, 1, 192, 192) from GRU hidden state h (B, h_dim).
    Pipeline: Linear -> (64,4,4) -> deconv x3 to 32x32 -> bilinear upsample to 64/128/192 with small convs.
    """
    def __init__(self, h_dim: int):
        super().__init__()
        self.fc = nn.Linear(h_dim, 64 * 4 * 4)
        self.unflat = nn.Unflatten(1, (64, 4, 4))

        # 4 -> 8 -> 16 -> 32
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 4->8
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 8->16
        self.deconv3 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)  # 16->32

        # 32 -> 64 -> 128 -> 192 (bilinear upsample + small convs)
        self.conv32 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv64 = nn.Sequential(nn.Conv2d(16, 8, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv128 = nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv192 = nn.Conv2d(8, 1, 3, 1, 1)

        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h):
        x = self.fc(h)                         # (B, 64*4*4)
        x = self.unflat(x)                     # (B, 64, 4, 4)

        x = self.act(self.deconv1(x))          # (B, 32, 8, 8)
        x = self.act(self.deconv2(x))          # (B, 16, 16, 16)
        x = self.act(self.deconv3(x))          # (B, 16, 32, 32)

        x = self.conv32(x)                     # (B, 16, 32, 32)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 16, 64, 64)

        x = self.conv64(x)                     # (B, 8, 64, 64)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 8, 128, 128)

        x = self.conv128(x)                    # (B, 8, 128, 128)
        x = F.interpolate(x, size=(192, 192), mode='bilinear', align_corners=False) # (B, 8, 192, 192)

        x = self.conv192(x)                    # (B, 1, 192, 192)
        return x
    
class EUReg_WM_RL(nn.Module):
    """
    Dreamer-style rigid US registration world model + actor-critic for rigid US registration.

    - World model:
        h_{t+1} = f(h_t, obs_t)   （obs_t = (vol_feat, cur_slice_feat, target_feat, pose_t)）
        r_pred_t = R(h_t)
        v_pred_t = V(h_t)
    - Policy:
        a_t = dT_t ~ π(h_t)  （高斯分布）
        T_{t+1} = T_t + step_scale * a_t
    - 环境:
        frame_{t+1} = FrameRigidTransformer(vol, T_{t+1})
        r_true_t     = - pose_error(T_{t+1}, dof_gt) 或基于 NCC/几何的 reward
    """
    def __init__(self, vol_shape, in_channel=1, first_channel=8,
                 h_dim=256, z_dim=128, action_dim=6):
        super().__init__()
        c = first_channel
        feat_dim = 8 * c

        # -------- Encoders --------
        self.encoder3d = Encoder3D(in_channel, c)
        self.encoder2d = Encoder2D(in_channel, c)
        self.pose_enc  = PoseEncoder(6, 64)

        self.proj_vol   = nn.Linear(feat_dim, z_dim)
        self.proj_slice = nn.Linear(feat_dim, z_dim)

        self.pool3 = AttnPool3D(feat_dim)
        self.pool2 = AttnPool2D(feat_dim)

        # -------- World latent (recurrent) --------
        self.rnn = RecurrentTransformer(dim=h_dim, nhead=8, layers=2)
        # obs -> h 输入线性
        self.inp = nn.Linear(3 * z_dim + 64, h_dim)

        # 初始 h
        self.h_dim = h_dim

        # -------- Actor (policy) --------
        self.policy_mean = nn.Sequential(
            nn.Linear(h_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # 全局 log_std 参数（六维动作）
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))

        # -------- Reward model (world model head) --------
        # 输出 scalar reward_pred
        self.reward_head = nn.Sequential(
            nn.Linear(h_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        # -------- Value head (critic) --------
        self.value_head = nn.Sequential(
            nn.Linear(h_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        # -------- Physics render (environment) --------
        self.transformer = FrameRigidTransformer(vol_shape[1:])

        # 可选重建 decoder
        self.decoder = Dec192(h_dim)

    def init_state(self, B, device):
        return torch.zeros(B, self.h_dim, device=device)

    @staticmethod
    def angle_wrap_deg(x):
        return (x + 180) % 360 - 180

    def encode_obs(self, vol, target_sl, cur_sl, T):
        """
        编码当前观测 (vol, target slice, current slice, pose) 为 z_obs
        """
        Fv = self.pool3(self.encoder3d(vol))        # (B, feat_dim)
        Ft = self.pool2(self.encoder2d(target_sl))  # target
        Fs = self.pool2(self.encoder2d(cur_sl))     # current slice

        zv = self.proj_vol(Fv)                      # (B, z_dim)
        zt = self.proj_slice(Ft)
        zs = self.proj_slice(Fs)
        zp = self.pose_enc(T)                       # (B, 64)

        z = torch.cat([zv, zs, zt, zp], dim=-1)     # (B, 3*z_dim+64)
        return z, (zv, zs, zt, zp)

    def forward(self, vol, target_sl, T0,
                steps=6, step_scale=0.4,
                sample_actions=True):
        """
        vol:       (B, 1, D, H, W)
        target_sl: (B, 1, H, W)
        T0:        (B, 6) 初始 pose
        返回：
            traj: list of dict，每一步包含：
                {
                    'T': T_t,
                    'frame': frame_t,
                    'h': h_t,
                    'reward_pred': r_pred_t,
                    'value_pred': v_pred_t,
                    'action': a_t,
                    'logprob': logprob_t
                }
        """
        device = vol.device
        B = vol.size(0)
        T = T0.clone()
        h = self.init_state(B, device)

        # 初始观察：用 T0 采第一帧
        frame = self.transformer(vol, T).squeeze(2)   # (B,1,H,W)

        traj = []

        Fv = self.pool3(self.encoder3d(vol))        # (B, feat_dim)
        zv = self.proj_vol(Fv)
        Ft = self.pool2(self.encoder2d(target_sl))  # target
        zt = self.proj_slice(Ft)

        for t in range(steps):
            # ---- world model encode + state update ----
            Fs = self.pool2(self.encoder2d(frame))     # current slice
                                  # (B, z_dim)
            zs = self.proj_slice(Fs)
            zp = self.pose_enc(T)                       # (B, 64)
            z_obs = torch.cat([zv, zs, zt, zp], dim=-1)     # (B, 3*z_dim+64)

            z_in = self.inp(z_obs)                    # (B,h_dim)
            h = self.rnn(z_in, h)                     # (B,h_dim)

            # ---- actor: sample action dT_t ----
            mean = self.policy_mean(h)                # (B,6)
            std  = self.policy_logstd.exp().unsqueeze(0).expand_as(mean)  # (B,6)
            dist = Normal(mean, std)

            if sample_actions:
                a = dist.rsample()                    # reparameterized sample
            else:
                a = mean                              # 用于 eval 时

            logprob = dist.log_prob(a).sum(-1)        # (B,)

            # ---- environment step: update T & slice ----
            dT = a
            T = T + step_scale * dT
            T[:, 3:] = self.angle_wrap_deg(T[:, 3:])

            frame = self.transformer(vol, T).squeeze(2)

            # ---- world heads: reward prediction & value ----
            reward_pred = self.reward_head(h)         # (B,1)
            value_pred  = self.value_head(h)          # (B,1)

            # 可选重建
            # recon_sl = self.decoder(h)

            traj.append({
                'T': T.clone(),
                'frame': frame,
                'h': h,
                'reward_pred': reward_pred,
                'value_pred': value_pred,
                'action': a,
                'logprob': logprob,
            })

        return traj
