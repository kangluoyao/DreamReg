import math
import torch
import torch.nn.functional as F

def sobel_grad_xy(img):
    # img: (B,1,H,W) in [-1,1] or [0,1]
    gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)/4.0
    gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)/4.0
    fx = F.conv2d(img, gx, padding=1)
    fy = F.conv2d(img, gy, padding=1)
    return fx, fy

def soft_hog(img, cell=8, bins=9, eps=1e-6):
    """
    img: (B,1,H,W), returns HOG descriptor per image as (B, C) where C = (#cells_y * #cells_x * bins)
    Soft assignment to orientation bins; L2-Hys block norm可省略，先做简单版。
    """
    B,_,H,W = img.shape
    fx, fy = sobel_grad_xy(img)
    mag = torch.sqrt(fx*fx + fy*fy + eps)                      # (B,1,H,W)
    ang = torch.atan2(fy, fx)                                  # [-pi, pi]
    ang = (ang + math.pi) / (2*math.pi)                        # -> [0,1)
    ang = ang * bins                                           # -> [0,bins)

    # soft bin assignment (triangle kernel)
    bin_floor = torch.floor(ang).long().clamp(max=bins-1)
    bin_ceil  = (bin_floor + 1) % bins
    w_ceil = ang - bin_floor.float()
    w_floor = 1.0 - w_ceil

    # per-bin maps
    maps = []
    for bidx in range(bins):
        w = torch.where(bin_floor==bidx, w_floor, torch.zeros_like(w_floor)) + \
            torch.where(bin_ceil==bidx,  w_ceil,  torch.zeros_like(w_ceil))
        maps.append((mag * w).squeeze(1))  # (B,H,W)
    maps = torch.stack(maps, dim=1)        # (B,bins,H,W)

    # cell histogram by average pooling
    pool = torch.nn.AvgPool2d(kernel_size=cell, stride=cell, ceil_mode=False)
    hog_cells = pool(maps)                 # (B,bins,Hc,Wc)
    B, bins, Hc, Wc = hog_cells.shape
    hog_vec = hog_cells.reshape(B, bins*Hc*Wc)
    hog_vec = hog_vec / (hog_vec.norm(dim=1, keepdim=True) + eps)  # L2 normalize
    return hog_vec  # (B, C)

def hog_cosine_reward(img_a, img_b):
    ha = soft_hog(img_a)
    hb = soft_hog(img_b)
    cos = F.cosine_similarity(ha, hb, dim=1, eps=1e-6)         # (B,)
    return cos.unsqueeze(-1)                                    # (B,1)


def sobel_grad(img):
    # img: (B,1,H,W), [-1,1] ok; 返回梯度幅值
    gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)/4.0
    gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)/4.0
    fx = F.conv2d(img, gx, padding=1)
    fy = F.conv2d(img, gy, padding=1)
    return torch.sqrt(fx*fx + fy*fy + 1e-6)

def safe_ncc(a, b, eps=1e-6):
    # a,b: (B,1,H,W)
    a2 = a - a.mean(dim=[2,3], keepdim=True)
    b2 = b - b.mean(dim=[2,3], keepdim=True)
    s1 = (a2*a2).mean(dim=[2,3], keepdim=True).sqrt()
    s2 = (b2*b2).mean(dim=[2,3], keepdim=True).sqrt()
    return ((a2*b2).mean(dim=[2,3], keepdim=True) / (s1*s2 + eps)).squeeze()


def normalize_vec(x, eps=1e-8):
    # x: (B,3)
    return x / (x.norm(dim=-1, keepdim=True) + eps)
