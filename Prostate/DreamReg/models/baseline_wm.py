import math
import torch
import torch.nn.functional as F
import torchgeometry as tgm
from torch import nn
from torch.distributions.normal import Normal

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
        self.layer2 = nn.Sequential(
                                    nn.AvgPool2d(2), 
                                    ConvBlock2D(c, 2*c), 
                                    ConvBlock2D(2*c, 2*c))
        self.layer3 = nn.Sequential(
                                    nn.AvgPool2d(2), 
                                    ConvBlock2D(2*c, 4*c), 
                                    ConvBlock2D(4*c, 4*c))
        self.layer4 = nn.Sequential(
                                    nn.AvgPool2d(2), 
                                    ConvBlock2D(4*c, 8*c), 
                                    ConvBlock2D(8*c, 8*c))

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x


class Encoder3D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super().__init__()
        c = first_channel
        self.layer1 = nn.Sequential(ConvBlock3D(in_channel, c), ConvBlock3D(c, c))
        self.layer2 = nn.Sequential(
                                    nn.AvgPool3d(2), 
                                    ConvBlock3D(c, 2*c),
                                    ConvBlock3D(2*c, 2*c))
        self.layer3 = nn.Sequential(
                                    nn.AvgPool3d(2), 
                                    ConvBlock3D(2*c, 4*c), 
                                    ConvBlock3D(4*c, 4*c))
        self.layer4 = nn.Sequential(
                                    nn.AvgPool3d(2), 
                                    ConvBlock3D(4*c, 8*c), 
                                    ConvBlock3D(8*c, 8*c))

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


# class Dec192(nn.Module):
#     """
#     Lightweight decoder to (B, 1, 192, 192) from GRU hidden state h (B, h_dim).
#     Pipeline: Linear -> (64,4,4) -> deconv x3 to 32x32 -> bilinear upsample to 64/128/192 with small convs.
#     """
#     def __init__(self, h_dim: int):
#         super().__init__()
#         self.fc = nn.Linear(h_dim, 64 * 4 * 4)
#         self.unflat = nn.Unflatten(1, (64, 4, 4))

#         # 4 -> 8 -> 16 -> 32
#         self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 4->8
#         self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 8->16
#         self.deconv3 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)  # 16->32

#         # 32 -> 64 -> 128 -> 192 (bilinear upsample + small convs)
#         self.conv32 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True))
#         self.conv64 = nn.Sequential(nn.Conv2d(16, 8, 3, 1, 1), nn.ReLU(inplace=True))
#         self.conv128 = nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(inplace=True))
#         self.conv192 = nn.Conv2d(8, 1, 3, 1, 1)

#         self.act = nn.ReLU(inplace=True)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, h):
#         x = self.fc(h)                         # (B, 64*4*4)
#         x = self.unflat(x)                     # (B, 64, 4, 4)

#         x = self.act(self.deconv1(x))          # (B, 32, 8, 8)
#         x = self.act(self.deconv2(x))          # (B, 16, 16, 16)
#         x = self.act(self.deconv3(x))          # (B, 16, 32, 32)

#         x = self.conv32(x)                     # (B, 16, 32, 32)
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 16, 64, 64)

#         x = self.conv64(x)                     # (B, 8, 64, 64)
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 8, 128, 128)

#         x = self.conv128(x)                    # (B, 8, 128, 128)
#         x = F.interpolate(x, size=(192, 192), mode='bilinear', align_corners=False) # (B, 8, 192, 192)

#         x = self.conv192(x)                    # (B, 1, 192, 192)
#         return x


class Dec192(nn.Module):
    def __init__(self, h_dim, base_ch=64):
        super().__init__()
        # 先把 h 映射到一个较粗的 feature map，比如 48×48
        self.fc = nn.Linear(h_dim, base_ch * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch//2, 4, 2, 1),  # 6→12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch//2, base_ch//4, 4, 2, 1),  # 12→24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch//4, base_ch//4, 4, 2, 1),  # 24→48
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch//4, 16, 4, 2, 1),  # 48→96
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(16, 8, 4, 2, 1),  # 96→192
            # nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1)  # 输出单通道
        )

    def forward(self, h):
        B = h.size(0)
        x = self.fc(h)                      # (B, C*6*6)
        x = x.view(B, -1, 4, 4)             # (B, C, 6, 6)
        x = self.net(x)                     # (B,1,192,192)
        return x



class EUReg_WM_Belief(nn.Module):
    """
    Dreamer-style world model for rigid US registration.
    Components:
      - Stochastic latent z_t with prior/posterior + KL
      - Deterministic belief h_t (GRU)
      - Observation decoder (2D slice)
      - Reward model head
      - Policy head (ΔT) for registration
    依赖模块（你已有）:
      Encoder3D, Encoder2D, PoseEncoder, Dec192, FrameRigidTransformer
    """

    def __init__(self, vol_shape, in_channel=1,
                 first_channel=8, h_dim=512, z_dim=1024):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        c = first_channel

        # ------------------------------
        # Encoders
        # ------------------------------
        self.encoder3d = Encoder3D(in_channel, c)
        self.encoder2d = Encoder2D(in_channel, c)
        self.pose_enc  = PoseEncoder(6, 64)  # 目前先不用在 world model 里强依赖 pose

        # Volume & slice projection to z_dim
        # 这里假设 encoder3d/2d 之后我们做 global average pooling 得到 (B,C')
        # self.vol_proj   = nn.Linear(c, z_dim)   # Fv_pool -> z_vol
        # self.slice_proj = nn.Linear(c, z_dim)   # Fs_pool -> z_slice / z_goal

        self.vol_proj   = nn.Linear(512*64, z_dim)
        self.slice_proj = nn.Linear(64*64, z_dim)


        # ------------------------------
        # Deterministic state transition (GRU over [z_t, a_t, z_goal, z_vol])
        # ------------------------------
        # self.gru = nn.GRUCell(z_dim + 6 + z_dim + z_dim, h_dim)

        # ------------------------------
        # Stochastic latent prior / posterior
        # prior: p(z_t | h_{t-1}, a_t, z_goal, z_vol)
        # post : q(z_t | h_{t-1}, a_t, z_goal, z_vol, z_obs)
        # ------------------------------
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim + 6 + 6 + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2*z_dim)  # [mu_p, logvar_p]
        )
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + 6 + 6 + z_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2*z_dim)  # [mu_q, logvar_q]
        )

        # ------------------------------
        # Decoder: reconstruct current slice from h_t
        # ------------------------------
        # self.dec192 = Dec192(256)  # 你原来的 Decoder，输出 (B,1,192,192)
        self.dec192 = Dec192(h_dim + 2*z_dim)  # 你原来的 Decoder，输出 (B,1,192,192)

        # ------------------------------
        # Reward head: from [h_t, z_t, z_goal] -> R (e.g. [cos_trans, cos_rot])
        # ------------------------------
        self.reward_head = nn.Sequential(
            nn.Linear(h_dim + z_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # ------------------------------
        # Policy head: ΔT from [h_t, z_goal]
        # ------------------------------
        self.delta_head = nn.Sequential(
            nn.Linear(h_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

        self.pretrain_proj = nn.Sequential(
            nn.Linear(h_dim + z_dim + z_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 256))
        
        self.value_head = nn.Sequential(
            nn.Linear(h_dim + z_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        # ------------------------------
        # Physics renderer: only used in observe mode (训练时获取真实 slice)
        # vol_shape: (B, C, D, H, W) -> 传进来的是整个体数据的shape
        # ------------------------------
        self.transformer = FrameRigidTransformer(vol_shape[1:])

        self.rnn = RecurrentTransformer(dim=h_dim, nhead=8, layers=4)
        self.inp = nn.Linear(z_dim + 6 + 6+ z_dim, h_dim)

    # ==================================================================
    # Utility
    # ==================================================================
    def init_state(self, B, device):
        """初始 belief h_0"""
        return torch.zeros(B, self.h_dim, device=device)

    @staticmethod
    def angle_wrap_deg(x):
        """把角度 wrap 到 [-180,180)"""
        return (x + 180.0) % 360.0 - 180.0

    def _encode_volume(self, vol):
        """
        vol: (B, C, D, H, W)
        返回 zv: (B, z_dim)
        """
        Fv = self.encoder3d(vol)          # (B, C', D', H', W')
        # Fv = Fv.mean(dim=[2, 3, 4])       # (B, C')
        B, C, Z, X, Y = Fv.shape
        Fv = Fv.view(B, C*Z*X*Y)
        zv = self.vol_proj(Fv)            # (B, z_dim)
        return zv

    def _encode_slice(self, sl):
        """
        sl: (B, 1, H, W)
        返回 zs: (B, z_dim)
        """
        Fs = self.encoder2d(sl)           # (B, C', H', W')
        # Fs = Fs.mean(dim=[2, 3])          # (B, C')
        B, C, X, Y = Fs.shape
        Fs = Fs.view(B, C*X*Y)
        zs = self.slice_proj(Fs)          # (B, z_dim)
        return zs

    def _split_mu_logvar(self, stats):
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def _sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        KL(q||p) for diagonal Gaussians, summed over latent dim.
        返回: (B,)
        """
        return 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p)
            - 1.0
        ).sum(dim=-1)

    # ==================================================================
    # World model: OBSERVE MODE (训练时，用真实环境)
    # ==================================================================
    def wm_observe_step(self, vol, goal_z, zv, T_t, a_t, h_prev):
        """
        单步 Dreamer-style 观察（train-time） - Version 1.

        Inputs:
            vol:    (B,1,D,H,W)   3D US 体数据
            goal_z: (B,z_dim)     目标 slice 的 embedding（仅用于 decoder/reward 的任务条件）
            zv:     (B,z_dim)     volume 的 embedding（环境/患者固定上下文，time-invariant）
            T_t:    (B,6)         当前真实 pose
            a_t:    (B,6)         当前真实 action (ΔT_t = T_{t+1}-T_t)
            h_prev: (B,h_dim)     上一时刻 deterministic state

        Returns dict:
            h_t, z_t, cur, sl_pred, r_pred, kl
        """
        # B = vol.size(0)

        # 1) 用真实 physics 获得当前 slice（只在 observe 模式用）
        with torch.no_grad():
            cur = self.transformer(vol, T_t).squeeze(2)  # (B,1,H,W)

        # 2) encode 当前观测 slice
        z_obs = self._encode_slice(cur)                  # (B,z_dim)

        # ------------------------------------------------------------
        # Version 1: world dynamics conditioning uses (h_prev, a_t, T_t, zv)
        # - remove goal_z from prior/post and RNN update to avoid goal shortcut
        # - include explicit pose T_t to make dynamics identifiable
        # ------------------------------------------------------------

        # 3) prior: p(z_t | h_prev, a_t, T_t, zv)
        prior_inp = torch.cat([h_prev, a_t, T_t, zv], dim=-1)
        prior_stats = self.prior_net(prior_inp)
        mu_p, logvar_p = self._split_mu_logvar(prior_stats)

        # 4) posterior: q(z_t | h_prev, a_t, T_t, zv, z_obs)
        post_inp = torch.cat([h_prev, a_t, T_t, zv, z_obs], dim=-1)
        post_stats = self.post_net(post_inp)
        mu_q, logvar_q = self._split_mu_logvar(post_stats)

        # 5) reparam sample z_t
        z_t = self._sample(mu_q, logvar_q)               # (B,z_dim)

        # 6) 更新 deterministic belief h_t（同样不喂 goal_z）
        gru_inp = torch.cat([z_t, a_t, T_t, zv], dim=-1)
        gru_inp = self.inp(gru_inp)
        h_t = self.rnn(gru_inp, h_prev)                  # (B,h_dim)

        # 7) decoder & reward（这里仍然允许用 goal_z 作为任务条件）
        dec_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        # print(dec_inp.shape)
        sl_pred = self.dec192(dec_inp)                   # (B,1,H,W)

        rew_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        r_pred  = self.reward_head(rew_inp)              # (B,2)

        # 8) KL(q||p)
        kl_t = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)  # (B,)

        return {
            "h_t": h_t,
            "z_t": z_t,
            "cur": cur,
            "sl_pred": sl_pred,
            "r_pred": r_pred,
            "kl": kl_t
        }


    def wm_observe_rollout(self, vol, goal_sl, T_seq, a_seq, h0=None):
        """
        训练 world model 时，沿真实轨迹 rollout 多步。

        Args:
            vol:     (B,1,D,H,W)
            goal_sl: (B,1,H,W)       目标 slice
            T_seq:   (B,L,6)         真实 pose 序列
            a_seq:   (B,L,6)         真实动作序列 ΔT
            h0:      (B,h_dim) 或 None  初始 belief

        Returns dict:
            {
                "kl_loss":      scalar
                "recon_loss":   scalar
                "cur_seq":      (B,L,1,H,W) 真slice
                "sl_pred_seq":  (B,L,1,H,W) 解码器重建的slice
                "r_pred_seq":   (B,L,2)
                "h_seq":        (B,L,h_dim)
                "z_seq":        (B,L,z_dim)
                "z_goal":       (B,z_dim)
                "zv":           (B,z_dim)
                "h_last":       (B,h_dim)
            }
        reward 的监督建议你在外部计算（拿 r_pred_seq 和你算好的 reward_gt 做 MSE）
        """
        B, L, _ = T_seq.shape
        device = vol.device

        if h0 is None:
            h = self.init_state(B, device)
        else:
            h = h0

        # encode 静态信息：volume + goal slice
        zv     = self._encode_volume(vol)       # (B,z_dim)
        z_goal = self._encode_slice(goal_sl)    # (B,z_dim)

        kls = []
        recons = []
        cur_list = []
        sl_pred_list = []
        r_pred_list = []
        h_list = []
        z_list = []

        for t in range(L):
            step = self.wm_observe_step(
                vol, z_goal, zv,
                T_seq[:, t], a_seq[:, t], h
            )
            h = step["h_t"]

            cur   = step["cur"]       # 真 slice
            sl_p  = step["sl_pred"]   # 重建 slice
            r_p   = step["r_pred"]

            # per-step reconstruction loss（先用 MSE，后续你可以换 MSSSIM+L1）
            recon_t = F.mse_loss(
                sl_p, cur,
                reduction='none'
            ).mean(dim=[1, 2, 3])     # (B,)

            kls.append(step["kl"])
            recons.append(recon_t)
            cur_list.append(cur)
            sl_pred_list.append(sl_p)
            r_pred_list.append(r_p)
            h_list.append(h)
            z_list.append(step["z_t"])

        kl_loss    = torch.stack(kls, dim=1).mean()
        recon_loss = torch.stack(recons, dim=1).mean()

        cur_seq     = torch.stack(cur_list, dim=1)      # (B,L,1,H,W)
        sl_pred_seq = torch.stack(sl_pred_list, dim=1)  # (B,L,1,H,W)
        r_pred_seq  = torch.stack(r_pred_list, dim=1)   # (B,L,2)
        h_seq       = torch.stack(h_list, dim=1)        # (B,L,h_dim)
        z_seq       = torch.stack(z_list, dim=1)        # (B,L,z_dim)

        return {
            "kl_loss": kl_loss,
            "recon_loss": recon_loss,
            "cur_seq": cur_seq,
            "sl_pred_seq": sl_pred_seq,
            "r_pred_seq": r_pred_seq,
            "h_seq": h_seq,
            "z_seq": z_seq,
            "z_goal": z_goal,
            "zv": zv,
            "h_last": h,
        }

    # ==================================================================
    # World model: IMAGINE MODE (不调用真实 transformer，只用 prior)
    # ==================================================================
    def wm_imagine_step(self, goal_z, zv, T_t, a_t, h_prev):
        """
        想象步（imagination） - Version 1: dynamics conditioned on (h_prev, a_t, T_t, zv).

        Inputs:
            goal_z: (B,z_dim)  目标 slice embedding（仅用于 decoder/reward 的任务条件）
            zv:     (B,z_dim)  volume embedding（环境上下文）
            T_t:    (B,6)      当前 pose（imagine 时由外部维护：T_{t+1}=T_t+a_t）
            a_t:    (B,6)      动作
            h_prev: (B,h_dim)  上一时刻 deterministic state

        Returns dict:
            h_t, z_t, sl_pred, r_pred
        """
        # 1) prior p(z_t | h_prev, a_t, T_t, zv)
        prior_inp = torch.cat([h_prev, a_t, T_t, zv], dim=-1)
        prior_stats = self.prior_net(prior_inp)
        mu_p, logvar_p = self._split_mu_logvar(prior_stats)

        # reparam sample from prior
        z_t = self._sample(mu_p, logvar_p)

        # 2) update deterministic belief h_t (do NOT feed goal_z)
        gru_inp = torch.cat([z_t, a_t, T_t, zv], dim=-1)
        gru_inp = self.inp(gru_inp)
        h_t = self.rnn(gru_inp, h_prev)

        # 3) decoder & reward (can use goal_z as task conditioning)
        dec_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        # print(dec_inp.shape)
        sl_pred = self.dec192(dec_inp)

        rew_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        r_pred  = self.reward_head(rew_inp)

        return {
            "h_t": h_t,
            "z_t": z_t,
            "sl_pred": sl_pred,
            "r_pred": r_pred
        }


    # ==================================================================
    # Policy / Registration Forward (测试/推理)
    # ==================================================================
    def forward(self, vol, goal_sl, T0,
                steps=6, step_scale=0.4,
                return_all=False):
        """
        推理阶段的 registration loop：
        - 用 world model belief (h_t) + 目标 embedding z_goal
        - 通过 delta_head(h_t, z_goal) 产生动作 ΔT
        - 用真实 transformer(vol, T) 执行动作（更新 pose，对应真正的环境）

        Args:
            vol:     (B,1,D,H,W)
            goal_sl: (B,1,H,W) 目标 slice
            T0:      (B,6) 初始 pose
        Returns:
            若 return_all=False: (T_final, cur_final)
            若 return_all=True : (T_final, cur_final, traj)
                traj: list of (T_t, cur_t, dT_t, r_pred_t, sl_pred_t)
        """
        device = vol.device
        B = vol.size(0)
        h = self.init_state(B, device)

        # encode 静态上下文
        zv     = self._encode_volume(vol)
        z_goal = self._encode_slice(goal_sl)

        T = T0.clone()
        traj = []

        for _ in range(steps):
            # ---- policy: ΔT = π(h_t, z_goal) ----
            pol_inp = torch.cat([h, z_goal], dim=-1)
            dT = self.delta_head(pol_inp)          # (B,6)

            # 更新真实 pose
            T_prev = T
            T = T + dT * step_scale
            T[:, 3:] = self.angle_wrap_deg(T[:, 3:])

            # ---- 用 world model 想象下一个 belief ----
            step = self.wm_imagine_step(z_goal, zv, T_prev, dT, h)
            h = step["h_t"]

            # ---- 真实环境的当前 slice（仅用于输出/可视化） ----
            cur = self.transformer(vol, T).squeeze(2)

            traj.append((T.clone(), cur, dT, step["r_pred"], step["sl_pred"]))

        if return_all:
            return T, cur, traj
        else:
            return T, cur
