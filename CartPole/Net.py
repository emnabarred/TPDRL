import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, actionSize):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, actionSize)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

