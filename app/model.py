import torch
import torch.nn as nn
import torch.nn.functional as F

class Shallow300(nn.Module):
    def __init__(self, random_state=47):
        super().__init__()
        torch.manual_seed(random_state)
        self.W0 = nn.Parameter(torch.randn(28*28, 300).mul_(2/pow(28*28, 0.5)))
        self.b0 = nn.Parameter(torch.zeros(300))

        self.W1 = nn.Parameter(torch.randn(300, 10).mul_(1/pow(300, 0.5)))
        self.b1 = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = torch.matmul(x, self.W0) + self.b0
        x = F.relu(x)
        x = torch.matmul(x, self.W1) + self.b1
        return x