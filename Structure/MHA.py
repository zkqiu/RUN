import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, hidden_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        return self.ln(x+self.fc2(self.relu(self.fc1(x))))
    
class MultiHeadAttention(nn.Module):
    # input: hidden_size, num_head, dropout_rate
    def __init__(self, hidden_size, num_head, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_head == 0, "hidden_size must be divided by num_head"

        self.hidden_size = hidden_size
        self.num_head = num_head
        self.head_dim = hidden_size // num_head

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.scale = self.head_dim ** 0.5

        self.ffn = FFN(hidden_size)
    
    def forward(self, x, mask=None):
        # dim of query, key, val: (batch_size, seq_len, hidden_size)
        batch_size = x.size(0)

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        print(f"Original Q size: {Q.shape}, K size: {K.shape}, V size: {V.shape}")

        Q = Q.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        print(f"Resized Q size: {Q.shape}, K size: {K.shape}, V size: {V.shape}")

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        print(f"scores size: {scores.shape}")
        if mask:
            scores = scores.masked_fill(mask==0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        print(f"Attention size: {attention.shape}")
        attention = self.dropout(attention)

        context = torch.matmul(attention, V) # (batch_size, num_head, seq_len, head_dim)
        print(f"Original context size: {context.shape}")
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        print(f"Resized context size: {context.shape}")
        context = self.fc(context)
        context = self.ln(context+x)

        context = self.ffn(context)

        return context


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    hidden_size = 16
    num_heads = 4

    mha = MultiHeadAttention(hidden_size, num_heads)
    x = torch.rand(batch_size, seq_len, hidden_size)
    # key = torch.rand(batch_size, seq_len, hidden_size)
    # value = torch.rand(batch_size, seq_len, hidden_size)
    mask = None

    output = mha(x, mask)
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, hidden_size)


        