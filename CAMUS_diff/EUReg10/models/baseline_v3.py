import math
import torchgeometry as tgm
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal
from torch import nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leakyrelu(x)
        return x


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = nn.Identity()
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leakyrelu(x)
        return x


class ProjectLN3D(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat


class ProjectLN2D(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 1)
        feat = self.norm(self.proj(feat))
        return feat


class FrameRigidTransformer(nn.Module):
    def __init__(self, slice_size, mode='bilinear'):
        super(FrameRigidTransformer, self).__init__()
        '''
        slice_size:(h, w) (default vol size (T, H, W)) 
        mode: 'bilinear'(default), 'neareast'
        '''

        self.mode = mode
        self.slice_size = [1] + list(slice_size)
        vectors = [torch.linspace(-0.5 * (s - 1), 0.5 * (s - 1), steps=s) for s in self.slice_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack([grids[2], grids[1], grids[0], torch.ones_like(grids[0])], dim=0) # G[x, y, z] = (z, y, x, 1)
        grid = grid.view(4, -1).type(torch.FloatTensor).contiguous()

        self.register_buffer('grid', grid)

    def dof2mat(self, input_dof):
        rad = tgm.deg2rad(input_dof[:, 3:])

        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk
        M = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
        M[:, 0, 0] = cj * ck
        M[:, 0, 1] = sj * sc - cs
        M[:, 0, 2] = sj * cc + ss
        M[:, 1, 0] = cj * sk
        M[:, 1, 1] = sj * ss + cc
        M[:, 1, 2] = sj * cs - sc
        M[:, 2, 0] = -sj
        M[:, 2, 1] = cj * si
        M[:, 2, 2] = cj * ci
        M[:, :3, 3] = input_dof[:, :3]  # 平移分量

        return M

    def forward(self, vol, dof):  # DOF已经翻转的，比如volsize是H,W,D, 那么dof是[Td, Tw, Th, Rd, Rw, Rh]

        mat = self.dof2mat(dof)  # 已经是z, y, x顺序的
        new_locs = torch.matmul(mat, self.grid)[:, :3]  # (z, y, x, 1)->(z, y, x)

        vol_size = vol.shape[2:]
        for i in range(len(vol_size)): # to [-1, 1]
            new_locs[:, i] = 2 * ((new_locs[:, i] + 0.5 * (vol_size[2 - i] - 1)) / (vol_size[2 - i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 1).contiguous().view(vol.shape[0], *self.slice_size, 3)

        return F.grid_sample(vol, new_locs, align_corners=True, mode=self.mode)

class Encoder2D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super(Encoder2D, self).__init__()

        c = first_channel
        self.layer1 = nn.Sequential(
            ConvBlock2D(in_channel, c),
            ConvBlock2D(c, c),
        )

        self.layer2 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock2D(c, 2 * c),
            ConvBlock2D(2 * c, 2 * c),
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock2D(2 * c, 4 * c),
            ConvBlock2D(4 * c, 4 * c),
        )

        self.layer4 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock2D(4 * c, 8 * c),
            ConvBlock2D(8 * c, 8 * c),
        )

        # self.layer5 = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     ConvBlock2D(8 * c, 8 * c),
        #     ConvBlock2D(8 * c, 8 * c),
        # )
        #

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        # out5 = self.layer5(out4)
        return out4  # , out2, out3, out4


class Encoder3D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super(Encoder3D, self).__init__()

        c = first_channel
        self.layer1 = nn.Sequential(
            ConvBlock3D(in_channel, c),
            ConvBlock3D(c, c),
        )

        self.layer2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(c, 2 * c),
            ConvBlock3D(2 * c, 2 * c),
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(2 * c, 4 * c),
            ConvBlock3D(4 * c, 4 * c),
        )

        self.layer4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(4 * c, 8 * c),
            ConvBlock3D(8 * c, 8 * c),
        )


    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        # out5 = self.layer5(out4)
        return out4  # , out2, out3, out4

class EUReg(nn.Module):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super(EUReg, self).__init__()
        c = first_channel
        self.encoder3d = Encoder3D(in_channel=in_channel, first_channel=c)
        self.encoder2d = Encoder2D(in_channel=in_channel, first_channel=c)

        L = 4
        deep_vol_shape = [s // 2 ** (L - 1) for s in vol_shape]
        self.deep_vol_shape = deep_vol_shape

        self.fn1 = nn.Sequential(
            # nn.Linear(1327104, 128),
            nn.Linear(184320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )


    def forward(self, vol, slice):
        Fv = self.encoder3d(vol)
        B, C, T, H, W = Fv.shape
        Fv = Fv.view(B, C, T * H * W)

        Fs = self.encoder2d(slice)
        B, C, H, W = Fs.shape
        Fs = Fs.view(B, C, H * W)

        #flatten and project to 6 dof
        feat = torch.concat([Fv, Fs], dim=2)  # B, C, THW+HW
        attn_flat = feat.view(B, -1)  # B, C*(TH

        dof_params = self.fn1(attn_flat)


        return dof_params

class EUReg_FRT(EUReg):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super().__init__(vol_shape, in_channel, first_channel, dim)

        self.transformer = FrameRigidTransformer(vol_shape[1:])

    def forward(self, vol, slice):
        pred_dof= super().forward(vol, slice)
        moved = self.transformer(vol, pred_dof).squeeze(2)

        return pred_dof, moved

class EUReg_FRT_Test(EUReg):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super().__init__(vol_shape, in_channel, first_channel, dim)

        self.transformer = FrameRigidTransformer(vol_shape[1:])

    def forward(self, vol, slice):
        pred_dof= super().forward(vol, slice)
        moved = self.transformer(vol, pred_dof).squeeze(2)

        return pred_dof, moved

if __name__ == '__main__':
    model = EUReg((32, 128, 128)).cuda()
    model2 = EUReg_FRT((32, 128, 128)).cuda()
    f = torch.ones(2, 1, 128, 128).cuda()
    v = torch.ones(2, 1, 32, 128, 128).cuda()
    output = model(v, f)
    for a in output:
        print(a.shape)

    output = model2(v, f)
    for a in output:
        print(a.shape)