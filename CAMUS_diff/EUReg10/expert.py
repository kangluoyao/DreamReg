import torch
from models.refine_net import FrameRigidTransformer

@torch.no_grad()
def ncc_score(a, b):
    """
    简单 NCC 得分（越大越好）。
    a, b: (B,1,H,W)
    这里给个简化版，你可以直接用你现有的 safe_ncc 实现。
    返回: (B,1)
    """
    B = a.size(0)
    a_flat = a.view(B, -1)
    b_flat = b.view(B, -1)

    a_mean = a_flat.mean(dim=1, keepdim=True)
    b_mean = b_flat.mean(dim=1, keepdim=True)

    a_c = a_flat - a_mean
    b_c = b_flat - b_mean

    num = (a_c * b_c).sum(dim=1, keepdim=True)
    den = (a_c.pow(2).sum(dim=1, keepdim=True).sqrt() *
           b_c.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-6)

    return num / den   # (B,1)


@torch.no_grad()
def expert_step_ncc(
    vol, target, T,
    transformer: FrameRigidTransformer,
    trans_step=2.0,
    rot_step=2.0,
    search_iters=1
):
    """
    NCC-based 经典优化器的一步或多步：
    给定当前 T，基于 NCC 在一个小邻域里搜索更好的 T_next（更高 NCC）。

    简单实现：coordinate-wise search，每个维度试 [-δ,0,+δ]。
    为了速度，search_iters 一般设 1-2 就够了。
    """
    device = vol.device
    B = vol.size(0)
    T_cur = T.clone()

    for _ in range(search_iters):
        # 当前 score
        cur_slice = transformer(vol, T_cur).squeeze(2)
        best_score = ncc_score(target, cur_slice)      # (B,1)
        best_T     = T_cur.clone()

        # 6 个 DOF 维度，依次搜索
        for dim in range(6):
            # 为该维度构造三个候选：-step, 0, +step
            step_size = trans_step if dim < 3 else rot_step   # 前三维是平移，后三维是角度
            offsets = torch.tensor([-step_size, 0.0, step_size],
                                   device=device)

            # 对三个 offset 一次性算（广播）
            # T_cur: (B,6) -> (3,B,6)
            T_candidates = T_cur.unsqueeze(0).repeat(3, 1, 1)   # (3,B,6)
            T_candidates[:, :, dim] += offsets.view(3, 1)

            # wrap 角度
            if dim >= 3:
                T_candidates[:, :, 3:] = (T_candidates[:, :, 3:] + 180) % 360 - 180

            # 展平 batch 维： (3*B,6)
            T_flat = T_candidates.view(-1, 6)
            # 采样对应 slice
            vol_rep = vol.repeat(3, 1, 1, 1, 1)       # (3B,1,D,H,W)
            cand_slice = transformer(vol_rep, T_flat).squeeze(2)  # (3B,1,H,W)
            target_rep = target.repeat(3, 1, 1, 1)    # (3B,1,H,W)

            scores = ncc_score(target_rep, cand_slice)   # (3B,1)
            scores = scores.view(3, B, 1)               # (3,B,1)

            # 对每个 batch，选出 score 更大的那个
            # best_score: (B,1); scores: (3,B,1)
            # 把 best_score 扩成 (3,B,1) 跟 scores 比
            better_mask = scores > best_score.unsqueeze(0)  # (3,B,1) bool

            # 如果某个 offset 比当前好，就更新对应的 best_T / best_score
            for i in range(3):
                mask_i = better_mask[i,:,0]  # (B,)
                if mask_i.any():
                    best_score[mask_i] = scores[i, mask_i]
                    best_T[mask_i]     = T_candidates[i, mask_i]

        T_cur = best_T

    # T_cur 此时是搜索后得到的 T_next
    return T_cur
