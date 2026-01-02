import torch

# ----------------------------------------------------------------------
# Hat operator
# ----------------------------------------------------------------------
def hat(w):
    B = w.size(0)
    wx, wy, wz = w[:,0], w[:,1], w[:,2]
    O = torch.zeros(B, 3, 3, device=w.device, dtype=w.dtype)
    O[:,0,1] = -wz
    O[:,0,2] =  wy
    O[:,1,0] =  wz
    O[:,1,2] = -wx
    O[:,2,0] = -wy
    O[:,2,1] =  wx
    return O


# ----------------------------------------------------------------------
# SO(3) exponential
# ----------------------------------------------------------------------
def so3_exp(w):
    B = w.size(0)
    theta = torch.norm(w, dim=1, keepdim=True)   # (B,1)
    theta_safe = theta + 1e-8

    W = hat(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B,3,3)

    A = (torch.sin(theta_safe) / theta_safe).unsqueeze(-1)
    Bcoef = ((1 - torch.cos(theta_safe)) / (theta_safe**2)).unsqueeze(-1)

    R = I + A * W + Bcoef * (W @ W)
    return R


# ----------------------------------------------------------------------
# SO(3) logarithm
# ----------------------------------------------------------------------
def so3_log(R):
    B = R.size(0)
    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-8, 1 - 1e-8)

    theta = torch.acos(cos_theta).view(B,1)

    w = torch.stack([
        R[:,2,1] - R[:,1,2],
        R[:,0,2] - R[:,2,0],
        R[:,1,0] - R[:,0,1],
    ], dim=1)

    small = 1e-8
    w = theta / (2*torch.sin(theta) + small) * w
    return w


# ----------------------------------------------------------------------
# SE(3) exponential
# ----------------------------------------------------------------------
def se3_exp(xi):
    B = xi.size(0)
    w = xi[:, :3]
    v = xi[:, 3:]

    R = so3_exp(w)

    theta = torch.norm(w, dim=1, keepdim=True)
    theta_safe = theta + 1e-8

    W = hat(w)

    A = (torch.sin(theta_safe)/theta_safe).unsqueeze(-1)
    Bcoef = ((1 - torch.cos(theta_safe))/(theta_safe**2)).unsqueeze(-1)
    Ccoef = ((theta_safe - torch.sin(theta_safe))/(theta_safe**3)).unsqueeze(-1)

    I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(B,3,3)

    V = I + Bcoef * W + Ccoef * (W @ W)

    t = (V @ v.unsqueeze(-1)).squeeze(-1)

    T = torch.zeros(B,4,4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1
    return T


# ----------------------------------------------------------------------
# SE(3) logarithm (稳定版)
# ----------------------------------------------------------------------
def se3_log(T):
    B = T.size(0)
    device = T.device

    R = T[:, :3, :3]
    t = T[:, :3, 3]

    w = so3_log(R)
    theta = torch.norm(w, dim=1, keepdim=True)
    theta_safe = theta + 1e-8

    W = hat(w)

    half = 0.5
    alpha = (1 - (theta_safe * torch.sin(theta_safe)) / (2*(1 - torch.cos(theta_safe)))) / (theta_safe**2)
    alpha = alpha.unsqueeze(-1)  # (B,1,1)

    I = torch.eye(3, device=device, dtype=T.dtype).unsqueeze(0).expand(B,3,3)

    Vinv = I - half * W + alpha * (W @ W)

    v = (Vinv @ t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([w, v], dim=1)


# ----------------------------------------------------------------------
# SE(3) composition
# ----------------------------------------------------------------------
def se3_compose(T, xi):
    return T @ se3_exp(xi)


# ----------------------------------------------------------------------
# Euler → rotation matrix
# ----------------------------------------------------------------------
def euler_to_matrix(euler_rad):
    ai, aj, ak = euler_rad[:,0], euler_rad[:,1], euler_rad[:,2]
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)

    B = euler_rad.size(0)
    R = torch.zeros(B,3,3, device=euler_rad.device, dtype=euler_rad.dtype)

    R[:,0,0] = cj*ck
    R[:,0,1] = sj*sk - si*ck
    R[:,0,2] = sj*ck + si*sk
    R[:,1,0] = cj*sk
    R[:,1,1] = sj*sk + ci*ck
    R[:,1,2] = sj*ck - ci*sk
    R[:,2,0] = -sj
    R[:,2,1] = cj*si
    R[:,2,2] = cj*ci

    return R


# ----------------------------------------------------------------------
# Build SE(3) geodesic trajectory
# ----------------------------------------------------------------------
def build_se3_trajectory(T0_6d, dof_6d, L, noise_std=0.0):
    B = T0_6d.size(0)
    device = T0_6d.device

    R_gt = euler_to_matrix(torch.deg2rad(dof_6d[:, 3:]))
    t_gt = dof_6d[:, :3]

    T_gt = torch.zeros(B,4,4, device=device)
    T_gt[:, :3, :3] = R_gt
    T_gt[:, :3, 3] = t_gt
    T_gt[:, 3, 3] = 1

    T0 = torch.eye(4, device=device).unsqueeze(0).expand(B,4,4)

    xi_gt = se3_log(T_gt)

    if noise_std > 0:
        xi_gt = xi_gt + noise_std * torch.randn_like(xi_gt)

    dxi = xi_gt / L

    T_seq=[]
    a_seq=[]

    Tcur = T0.clone()

    for _ in range(L):
        Tcur = se3_compose(Tcur, dxi)
        T_seq.append(Tcur)
        a_seq.append(dxi)

    T_seq = torch.stack(T_seq, dim=1)   # (B,L,4,4)
    a_seq = torch.stack(a_seq, dim=1)   # (B,L,6)
    return T_seq, a_seq


def se3_delta_from_pair(T_prev, T_next):
    """
    从两帧位姿 T_prev, T_next 求相对 twist Δξ

    Args:
        T_prev: (B,4,4)
        T_next: (B,4,4)

    Returns:
        xi: (B,6) = [omega, v]
    """
    assert T_prev.shape == T_next.shape
    B = T_prev.size(0)
    device = T_prev.device
    dtype  = T_prev.dtype

    R_prev = T_prev[:, :3, :3]      # (B,3,3)
    t_prev = T_prev[:, :3, 3]       # (B,3)
    R_next = T_next[:, :3, :3]
    t_next = T_next[:, :3, 3]

    # T_rel = T_prev^{-1} @ T_next
    # R_rel = R_prev^T R_next
    R_rel = R_prev.transpose(1, 2) @ R_next                    # (B,3,3)

    # t_rel = R_prev^T (t_next - t_prev)
    t_diff = (t_next - t_prev).unsqueeze(-1)                   # (B,3,1)
    t_rel  = (R_prev.transpose(1, 2) @ t_diff).squeeze(-1)     # (B,3)

    T_rel = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    T_rel[:, :3, :3] = R_rel
    T_rel[:, :3, 3]  = t_rel
    T_rel[:, 3, 3]   = 1.0

    xi = se3_log(T_rel)      # (B,6)  [omega, v]
    return xi

