import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchvision.models import efficientnet_b1


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


class EfficientNetModel(nn.Module):

    NAME = 'efficientnet_b1'

    def __init__(self,
                 model_cfg,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        dropout = model_cfg.params.dropout
        out_features = model_cfg.params.output_channels

        # instantiate the base EfficientNet-B1
        self.net = efficientnet_b1(weights='IMAGENET1K_V2', dropout=dropout)

        # in the Kaggle code, they override the `nn.AdaptiveAvgPool2d(1)`
        # to be a GeM:
        self.net.avgpool = GeM()

        # they also override the fully-connected layer. In the `efficientnet_b1`
        # model they have self.classifier as a nn.Sequential of a nn.Dropout and
        # a nn.Linear. Here, we keep that nn.Dropout, but replace the nn.Linear
        # by a bunch of things:

        # First get the dimensions of the linear classifier
        # from the default `classifier` component. 
        # Index 1 is the nn.Linear layer
        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            SEBlock(in_features),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, out_features),
        )