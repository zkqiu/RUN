## 多查询注意力
import torch
from torch import nn

class FFN(nn.Module):
    def __init__(self, hidden_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        return self.ln(x+self.fc2(self.relu(self.fc1(x))))

class MultiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        ## 初始化Q、K、V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim) ###
        self.v_linear = nn.Linear(hidden_size, self.head_dim) ###
        
        ## 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

        self.ffn = FFN(hidden_size)
        
    def forward(self, x, attention_mask=None):
        batch_size = x.size()[0]
        
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)
        
        query = self.split_head(query)
        key = self.split_head(key, 1)
        value = self.split_head(value, 1)
        
        ## 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        
        ## 对注意力分数进行归一化
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_probs, value)
        
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        
        output = self.ln(self.o_linear(output) + x)

        output = self.ffn(x)
        
        return output
        
    def split_head(self, x, head_num=None):
        
        batch_size = x.size()[0]
        
        if head_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        else:
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1,2)
        
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    hidden_size = 16
    num_heads = 4

    mqa = MultiQueryAttention(hidden_size, num_heads)
    x = torch.rand(batch_size, seq_len, hidden_size)
    # key = torch.rand(batch_size, seq_len, hidden_size)
    # value = torch.rand(batch_size, seq_len, hidden_size)
    mask = None

    output, attention = mqa(x, mask)
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, hidden_size)
    print("Attention shape:", attention.shape)  # Expected: (batch_size, num_heads, seq_len, seq_len)