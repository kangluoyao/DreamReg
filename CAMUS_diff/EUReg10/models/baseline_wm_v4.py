import math
import torch
import torch.nn.functional as F
import torchgeometry as tgm
from torch import nn

# --------------------
# Small utils
# --------------------

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
        x = torch.stack([m, z], dim=1)
        for ln1, attn, ln2, ff in zip(self.layers[0::4], self.layers[1::4],
                                      self.layers[2::4], self.layers[3::4]):
            y, _ = attn(ln1(x), ln1(x), ln1(x))
            x = x + y
            x = x + ff(ln2(x))
        m_new = self.mem_proj_out(x[:, 0])
        return m_new


class AttnPool2D(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.q = nn.Parameter(torch.randn(1, hidden_dim))
        self.k_proj = nn.Linear(dim, hidden_dim)
        self.v_proj = nn.Linear(dim, hidden_dim)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, N, C]
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q.expand(B, -1).unsqueeze(1)     # [B,1,Hdim]
        attn = (q @ k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        w = attn.softmax(-1)
        out = (w @ v).squeeze(1)
        return out


class AttnPool3D(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.q = nn.Parameter(torch.randn(1, hidden_dim))
        self.k_proj = nn.Linear(dim, hidden_dim)
        self.v_proj = nn.Linear(dim, hidden_dim)

    def forward(self, x):  # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, N, C]
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q.expand(B, -1).unsqueeze(1)
        attn = (q @ k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        w = attn.softmax(-1)
        out = (w @ v).squeeze(1)
        return out


# --------------------
# Basic Conv Blocks
# --------------------

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)
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
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)
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
        self.layer2 = nn.Sequential(nn.AvgPool2d(2), ConvBlock2D(c, 2*c), ConvBlock2D(2*c, 2*c))
        self.layer3 = nn.Sequential(nn.AvgPool2d(2), ConvBlock2D(2*c, 4*c), ConvBlock2D(4*c, 4*c))
        self.layer4 = nn.Sequential(nn.AvgPool2d(2), ConvBlock2D(4*c, 8*c), ConvBlock2D(8*c, 8*c))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Encoder3D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super().__init__()
        c = first_channel
        self.layer1 = nn.Sequential(ConvBlock3D(in_channel, c), ConvBlock3D(c, c))
        self.layer2 = nn.Sequential(nn.AvgPool3d(2), ConvBlock3D(c, 2*c), ConvBlock3D(2*c, 2*c))
        self.layer3 = nn.Sequential(nn.AvgPool3d(2), ConvBlock3D(2*c, 4*c), ConvBlock3D(4*c, 4*c))
        self.layer4 = nn.Sequential(nn.AvgPool3d(2), ConvBlock3D(4*c, 8*c), ConvBlock3D(8*c, 8*c))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack([grids[2], grids[1], grids[0], torch.ones_like(grids[0])], dim=0)
        grid = grid.view(4, -1).float()
        self.register_buffer('grid', grid)

    def dof2mat(self, dof):
        rad = tgm.deg2rad(dof[:, 3:])
        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        M = torch.eye(4, device=dof.device, dtype=dof.dtype).repeat(dof.shape[0], 1, 1)
        M[:, 0, 0] = cj * ck
        M[:, 0, 1] = sj * sc - cs
        M[:, 0, 2] = sj * cc + ss
        M[:, 1, 0] = cj * sk
        M[:, 1, 1] = sj * ss + cc
        M[:, 1, 2] = sj * cs - sc
        M[:, 2, 0] = -sj
        M[:, 2, 1] = cj * si
        M[:, 2, 2] = cj * ci
        M[:, :3, 3] = dof[:, :3]
        return M

    def forward(self, vol, dof):
        M = self.dof2mat(dof)                                # (B,4,4)
        new = torch.matmul(M, self.grid)[:, :3]             # (B,3,N)
        vol_size = vol.shape[2:]
        for i in range(len(vol_size)):
            new[:, i] = 2 * ((new[:, i] + 0.5*(vol_size[2-i]-1)) /
                              (vol_size[2-i]-1) - 0.5)
        new = new.permute(0, 2, 1).view(vol.shape[0], *self.slice_size, 3)
        return F.grid_sample(vol, new, align_corners=True, mode=self.mode)


# --------------------
# Pose encoder
# --------------------

class PoseEncoder(nn.Module):
    def __init__(self, in_dim=6, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, T):
        return self.net(T)


class Dec192(nn.Module):
    def __init__(self, h_dim, base_ch=64):
        super().__init__()
        self.fc = nn.Linear(h_dim, base_ch * 6 * 6)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch//2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch//2, base_ch//4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch//4, base_ch//4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch//4, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, 1, 1)
        )

    def forward(self, h):
        B = h.size(0)
        x = self.fc(h)
        x = x.view(B, -1, 6, 6)
        x = self.net(x)
        return x


# --------------------
# Dreamer-Style World Model
# --------------------

class EUReg_WM_Belief(nn.Module):
    """
    Dreamer-style world model for rigid US registration.

    Components:
      - Stochastic latent z_t with prior/posterior + KL
      - Deterministic belief h_t (GRU)
      - Observation decoder (2D slice)
      - Reward model head
      - Policy head (ΔT) for registration
      - [+] Value head for Dreamer-style actor-critic
    """

    def __init__(self, vol_shape, in_channel=1,
                 first_channel=8, h_dim=512, z_dim=1024):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        c = first_channel

        self.encoder3d = Encoder3D(in_channel, c)
        self.encoder2d = Encoder2D(in_channel, c)
        self.pose_enc  = PoseEncoder(6, 64)

        # 这里直接 flatten 后映射到 z_dim（根据你之前的 shape 写死）
        self.vol_proj   = nn.Linear(2304 * 64, z_dim)
        self.slice_proj = nn.Linear(576 * 64, z_dim)

        # GRU over [z_t, a_t, z_goal, z_vol]
        self.gru = nn.GRUCell(z_dim + 6 + z_dim + z_dim, h_dim)

        # Prior / Posterior
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim + 6 + z_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim)
        )
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + 6 + z_dim + z_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim)
        )

        # Decoder
        self.dec192 = Dec192(256)  # 输入为 pretrain_proj 输出 256D

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(h_dim + z_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # Policy head
        self.delta_head = nn.Sequential(
            nn.Linear(h_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

        # [+] pretrain_proj: 用于降维到 decoder 输入
        self.pretrain_proj = nn.Sequential(
            nn.Linear(h_dim + z_dim + z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # [+] Value head for Dreamer-RL
        self.value_head = nn.Sequential(                 # [+]
            nn.Linear(h_dim + z_dim + z_dim, 256),       # [+]
            nn.ReLU(),                                   # [+]
            nn.Linear(256, 1)                            # [+]
        )                                                # [+]

        self.transformer = FrameRigidTransformer(vol_shape[1:])

    # ==================================================================
    # Utility
    # ==================================================================
    def init_state(self, B, device):
        return torch.zeros(B, self.h_dim, device=device)

    @staticmethod
    def angle_wrap_deg(x):
        return (x + 180.0) % 360.0 - 180.0

    def _encode_volume(self, vol):
        Fv = self.encoder3d(vol)         # (B,C,D',H',W')
        B, C, D, H, W = Fv.shape
        Fv = Fv.view(B, C * D * H * W)
        zv = self.vol_proj(Fv)
        return zv

    def _encode_slice(self, sl):
        Fs = self.encoder2d(sl)
        B, C, H, W = Fs.shape
        Fs = Fs.view(B, C * H * W)
        zs = self.slice_proj(Fs)
        return zs

    def _split_mu_logvar(self, stats):
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def _sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        return 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1.0
        ).sum(dim=-1)

    # ==================================================================
    # OBSERVE MODE
    # ==================================================================
    def wm_observe_step(self, vol, goal_z, zv, T_t, a_t, h_prev):
        with torch.no_grad():
            cur = self.transformer(vol, T_t).squeeze(2)

        z_obs = self._encode_slice(cur)

        prior_inp = torch.cat([h_prev, a_t, goal_z, zv], dim=-1)
        prior_stats = self.prior_net(prior_inp)
        mu_p, logvar_p = self._split_mu_logvar(prior_stats)

        post_inp = torch.cat([h_prev, a_t, goal_z, zv, z_obs], dim=-1)
        post_stats = self.post_net(post_inp)
        mu_q, logvar_q = self._split_mu_logvar(post_stats)

        z_t = self._sample(mu_q, logvar_q)

        gru_inp = torch.cat([z_t, a_t, goal_z, zv], dim=-1)
        h_t = self.gru(gru_inp, h_prev)

        dec_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        dec_inp = self.pretrain_proj(dec_inp)
        sl_pred = self.dec192(dec_inp)

        rew_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        r_pred = self.reward_head(rew_inp)

        kl_t = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

        return {
            "h_t": h_t,
            "z_t": z_t,
            "cur": cur,
            "sl_pred": sl_pred,
            "r_pred": r_pred,
            "kl": kl_t
        }

    def wm_observe_rollout(self, vol, goal_sl, T_seq, a_seq, h0=None):
        B, L, _ = T_seq.shape
        device = vol.device

        if h0 is None:
            h = self.init_state(B, device)
        else:
            h = h0

        zv = self._encode_volume(vol)
        z_goal = self._encode_slice(goal_sl)

        kls, recons = [], []
        cur_list, sl_pred_list, r_pred_list = [], [], []
        h_list, z_list = [], []

        for t in range(L):
            step = self.wm_observe_step(
                vol, z_goal, zv,
                T_seq[:, t], a_seq[:, t], h
            )
            h = step["h_t"]
            cur  = step["cur"]
            sl_p = step["sl_pred"]
            r_p  = step["r_pred"]

            recon_t = F.mse_loss(sl_p, cur, reduction='none').mean(dim=[1, 2, 3])

            kls.append(step["kl"])
            recons.append(recon_t)
            cur_list.append(cur)
            sl_pred_list.append(sl_p)
            r_pred_list.append(r_p)
            h_list.append(h)
            z_list.append(step["z_t"])

        kl_loss = torch.stack(kls, dim=1).mean()
        recon_loss = torch.stack(recons, dim=1).mean()

        cur_seq     = torch.stack(cur_list, dim=1)
        sl_pred_seq = torch.stack(sl_pred_list, dim=1)
        r_pred_seq  = torch.stack(r_pred_list, dim=1)
        h_seq       = torch.stack(h_list, dim=1)
        z_seq       = torch.stack(z_list, dim=1)

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
            "h_last": h
        }

    # ==================================================================
    # IMAGINE MODE
    # ==================================================================
    def wm_imagine_step(self, goal_z, zv, a_t, h_prev):
        prior_inp = torch.cat([h_prev, a_t, goal_z, zv], dim=-1)
        prior_stats = self.prior_net(prior_inp)
        mu_p, logvar_p = self._split_mu_logvar(prior_stats)
        z_t = self._sample(mu_p, logvar_p)

        gru_inp = torch.cat([z_t, a_t, goal_z, zv], dim=-1)
        h_t = self.gru(gru_inp, h_prev)

        dec_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        dec_inp = self.pretrain_proj(dec_inp)
        sl_pred = self.dec192(dec_inp)

        rew_inp = torch.cat([h_t, z_t, goal_z], dim=-1)
        r_pred = self.reward_head(rew_inp)

        return {
            "h_t": h_t,
            "z_t": z_t,
            "sl_pred": sl_pred,
            "r_pred": r_pred
        }

    # ==================================================================
    # Policy / Inference
    # ==================================================================
    def forward(self, vol, goal_sl, T0,
                steps=6, step_scale=0.4,
                return_all=False):
        device = vol.device
        B = vol.size(0)
        h = self.init_state(B, device)

        zv = self._encode_volume(vol)
        z_goal = self._encode_slice(goal_sl)

        T = T0.clone()
        traj = []

        for _ in range(steps):
            pol_inp = torch.cat([h, z_goal], dim=-1)
            dT = self.delta_head(pol_inp)

            T = T + dT * step_scale
            T[:, 3:] = self.angle_wrap_deg(T[:, 3:])

            step = self.wm_imagine_step(z_goal, zv, dT, h)
            h = step["h_t"]

            cur = self.transformer(vol, T).squeeze(2)
            traj.append((T.clone(), cur, dT, step["r_pred"], step["sl_pred"]))

        if return_all:
            return T, cur, traj
        else:
            return T, cur
