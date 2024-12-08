import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x*self.sigmoid(x)

class SwishGLU(nn.Module):
    def __init__(self, hidden_size):
        super(SwishGLU, self).__init__()

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.swish = Swish()
    
    def forward(self, x):
        return self.swish(self.fc1(x))*self.fc2(x)


if __name__=="__main__":
    batch_size = 2
    seq_len = 16
    hidden_size = 8

    layer = SwishGLU(hidden_size)

    x = torch.rand([batch_size, seq_len, hidden_size])
    y = layer(x)

    print(f"x.shape is: {x.shape}")
    print(f"y.shape is: {y.shape}")