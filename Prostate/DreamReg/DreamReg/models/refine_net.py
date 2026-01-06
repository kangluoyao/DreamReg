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
    def __init__(self, dim, nhead=8, mlp=512, layers=4):
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


class PoseEncoder(nn.Module):
    def __init__(self, in_dim=6, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    def forward(self, T): return self.net(T)


    
class ProbeAdjustPolicy(nn.Module):
    """
    学习经典优化器行为的 policy：
    输入 (vol, target_slice, current_slice, current_pose) -> 预测 ΔT_pred
    """
    def __init__(self, vol_shape, in_channel=1, first_channel=8,
                 z_dim=128, h_dim=256):
        super().__init__()
        c = first_channel
        feat_dim = 8 * c

        self.encoder3d = Encoder3D(in_channel, c)
        self.encoder2d = Encoder2D(in_channel, c)
        self.pose_enc  = PoseEncoder(6, 64)
        self.transformer = FrameRigidTransformer(vol_shape[1:])

        self.pool3 = AttnPool3D(feat_dim)
        self.pool2 = AttnPool2D(feat_dim)

        self.proj_vol   = nn.Linear(feat_dim, z_dim)
        self.proj_slice = nn.Linear(feat_dim, z_dim)

        in_dim = 3*z_dim + 64  # zv, zs, zt, zp
        self.head = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, 128), nn.ReLU(),
            nn.Linear(128, 6)      # 输出 ΔT
        )

    @staticmethod
    def angle_wrap_deg(x):
        return (x + 180) % 360 - 180

    def encode_state(self, vol, target, cur, T):
        """
        编码当前状态 -> feature 向量
        vol:    (B,1,D,H,W)
        target: (B,1,H,W)
        cur:    (B,1,H,W)  当前 slice
        T:      (B,6)
        """
        Fv = self.pool3(self.encoder3d(vol))        # (B, feat_dim)
        Ft = self.pool2(self.encoder2d(target))     # (B, feat_dim)
        Fs = self.pool2(self.encoder2d(cur))        # (B, feat_dim)

        zv = self.proj_vol(Fv)                      # (B,z_dim)
        zt = self.proj_slice(Ft)
        zs = self.proj_slice(Fs)
        zp = self.pose_enc(T)                       # (B,64)

        z  = torch.cat([zv, zs, zt, zp], dim=-1)    # (B,3*z_dim+64)
        return z

    def forward_step(self, vol, target, T):
        """
        做一步：输入 vol / target / 当前 T -> ΔT_pred, cur_slice
        （方便训练/推理时调用）
        """
        cur = self.transformer(vol, T).squeeze(2)   # (B,1,H,W)
        z   = self.encode_state(vol, target, cur, T)
        dT  = self.head(z)             # 限制在 [-1,1]，再配 step_scale
        # dT  = torch.tanh(self.head(z))              # 限制在 [-1,1]，再配 step_scale
        return dT, cur

    def forward(self, vol, target, T0, steps=5, step_scale=0.2, return_all=False):
        """
        仅供 inference 使用：从 T0 出发跑多步 refinement
        """
        T = T0.clone()
        traj = []
        for _ in range(steps):
            dT, cur = self.forward_step(vol, target, T)
            T = T + dT * step_scale
            T[:, 3:] = self.angle_wrap_deg(T[:, 3:])
            traj.append((T, cur, dT))

        final_cur = self.transformer(vol, T).squeeze(2)

        if return_all:
            return traj
        else:
            return T, final_cur

