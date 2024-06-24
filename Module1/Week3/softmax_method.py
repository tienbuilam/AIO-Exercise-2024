import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        exps = torch.exp(x)
        return exps / torch.sum(exps)


class softmax_stable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = torch.max(x)
        exps = torch.exp(x - c)
        return exps / torch.sum(exps)


data = torch.Tensor([1, 2, 300000000])

my_softmax = Softmax()
output = my_softmax(data)
print(output)

my_softmax_stable = softmax_stable()
output = my_softmax_stable(data)
print(output)
