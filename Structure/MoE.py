import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 定义专家网络
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        
        # 门控网络
        self.gating_network = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # 计算专家的得分
        gate_scores = self.gating_network(x)  # Shape: (batch_size, num_experts)
        gate_probs = F.softmax(gate_scores, dim=-1)  # Shape: (batch_size, num_experts)
        
        # 获取 Top-k 的专家索引和对应权重
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # Shape: (batch_size, top_k)
        
        # 初始化输出
        output = torch.zeros(x.size(0), self.experts[0].out_features, device=x.device)
        
        # 动态选择专家并加权输出
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # 当前选中的专家索引
            expert_weight = topk_weights[:, i].unsqueeze(1)  # 当前选中专家的权重
            
            # 获取专家输出
            expert_output = torch.stack([self.experts[idx](x[j].unsqueeze(0)) for j, idx in enumerate(expert_idx)])
            
            # 加权求和
            output += expert_weight * expert_output.squeeze(1)
        
        return output

# 示例使用
if __name__ == "__main__":
    # 定义输入参数
    batch_size = 4
    input_dim = 16
    output_dim = 8
    num_experts = 3
    top_k = 2

    # 初始化MoE层
    moe_layer = MoELayer(input_dim=input_dim, output_dim=output_dim, num_experts=num_experts, top_k=top_k)
    
    # 创建示例输入
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    output = moe_layer(x)
    print("输出形状:", output.shape)
