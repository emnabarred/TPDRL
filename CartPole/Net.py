import torch

class Net(torch.nn.Module):
    def __init__(self, actionSize):
        super(Net, self).__init__()
        self.entry = torch.nn.Linear(4, 37)
        self.middle_1 = torch.nn.Linear(37, 37)
        self.output = torch.nn.Linear(37, actionSize)

        torch.nn.init.xavier_uniform_(self.entry.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.xavier_uniform_(self.middle_1.weight)


    def forward(self, x):
        model = torch.nn.Sequential(self.entry, torch.nn.ReLU(), self.middle_1, torch.nn.ReLU(), self.output)
        return model(x)