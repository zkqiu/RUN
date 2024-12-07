import torch
import math

def apply_rope(q, k, seq_len, d_model):
    # 假设 d_model 是偶数，分成 [cos, sin] 两部分
    theta = torch.arange(0, d_model, step=2) / d_model
    theta = 1.0 / (10000 ** theta)  # 类似 sinusoidal encoding 的频率
    positions = torch.arange(0, seq_len).unsqueeze(1)  # 位置 [seq_len, 1]
    
    # 计算 [cos, sin] 的值
    cos_theta = torch.cos(positions * theta)
    sin_theta = torch.sin(positions * theta)
    
    # 对 Q 和 K 进行变换
    q1, q2 = q[..., ::2], q[..., 1::2]  # 偶数维和奇数维
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    q_rope = torch.cat([q1 * cos_theta - q2 * sin_theta, 
                        q1 * sin_theta + q2 * cos_theta], dim=-1)
    k_rope = torch.cat([k1 * cos_theta - k2 * sin_theta, 
                        k1 * sin_theta + k2 * cos_theta], dim=-1)
    
    return q_rope, k_rope
