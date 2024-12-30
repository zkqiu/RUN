import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-8):
        """
        RMSNorm 初始化。
        :param normalized_dim: 要归一化的特征维度大小
        :param eps: 防止分母为零的小值
        """
        super(RMSNorm, self).__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_dim))  # 可学习的缩放因子

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 (..., normalized_dim)
        :return: 归一化后的张量
        """
        # 计算均方根
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)  # Shape: (..., 1)
        # 归一化并缩放
        x_normalized = x / rms  # Shape: (..., normalized_dim)
        return x_normalized * self.scale  # Shape: (..., normalized_dim)

# 示例使用
if __name__ == "__main__":
    # 定义输入参数
    batch_size = 4
    seq_length = 10
    hidden_dim = 16

    # 创建RMSNorm层
    rms_norm = RMSNorm(normalized_dim=hidden_dim)

    # 创建示例输入
    x = torch.randn(batch_size, seq_length, hidden_dim)
    
    # 前向传播
    output = rms_norm(x)
    print("输出形状:", output.shape)
