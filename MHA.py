import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    # input: embed_dim, num_head, dropout_rate
    def __init__(self, embed_dim, num_head, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_head == 0, "embed_dim must be divided by num_head"

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.ln = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU()

        self.scale = self.head_dim ** 0.5
    
    def forward(self, x, mask=None):
        # dim of query, key, val: (batch_size, seq_len, embed_dim)
        batch_size = x.size(0)

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask:
            scores = scores.masked_fill(mask==0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V) # (batch_size, num_head, seq_len, head_dim)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        context = self.ln(context+x)

        context = self.ln(self.linear2(self.relu(self.linear1(context)))+context)

        return context


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4

    mha = MultiHeadAttention(embed_dim, num_heads)
    x = torch.rand(batch_size, seq_len, embed_dim)
    # key = torch.rand(batch_size, seq_len, embed_dim)
    # value = torch.rand(batch_size, seq_len, embed_dim)
    mask = None

    output, attention = mha(x, mask)
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, embed_dim)
    print("Attention shape:", attention.shape)  # Expected: (batch_size, num_heads, seq_len, seq_len)


        