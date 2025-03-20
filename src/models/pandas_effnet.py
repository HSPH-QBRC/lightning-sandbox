import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchvision.models import efficientnet_b1

from models.submodules import GeM, SEBlock


class EfficientNetB1Model(nn.Module):
    '''
    This model class extends the base EfficientNet-B1 model
    and replaces various components (e.g. avg pool - GeM)
    '''

    NAME = 'pandas_efficientnet_b1'

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

    def forward(self, x):
        '''
        The forward pass through the network.

        `x` is a batch of images,
        e.g. x.shape = [batch_size. C, H, W] where H,W are
        image height and width.
        '''
        return self.net(x)