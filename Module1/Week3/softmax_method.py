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
        # x_max = torch.max(x, dim=0, keepdims=True)
        # x_exp = torch.exp(x - x_max . values)
        # partition = x_exp.sum(0, keepdims=True)
        # return x_exp / partition


data = torch.Tensor([1, 2, 3])
softmax_function = nn.Softmax(dim=0)
output = softmax_function(data)
assert round(output[0].item(), 2) == 0.09

data = torch.Tensor([5, 2, 4])
my_softmax = Softmax()
output = my_softmax(data)
assert round(output[-1].item(), 2) == 0.26


data = torch.Tensor([1, 2, 300000000])
my_softmax = Softmax()
output = my_softmax(data)
assert round(output[0].item(), 2) == 0.0

data = torch . Tensor([1, 2, 3])
softmax_stable = softmax_stable()
output = softmax_stable(data)
assert round(output[-1].item(), 2) == 0.67
