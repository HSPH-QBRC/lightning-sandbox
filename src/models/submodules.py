import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(
                            x.clamp(min=eps).pow(p), # such that the minimum is not too small
                            (x.size(-2), x.size(-1)) # the full img size
                            ).pow(1.0 / p)


class SEBlock(nn.Module):
    '''
    Variant on the Squeeze-Excitation block to match 
    https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution/blob/master/src/models/layer.py#L88
    '''
    def __init__(self, in_ch, r=8):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = input_x * x
        return x