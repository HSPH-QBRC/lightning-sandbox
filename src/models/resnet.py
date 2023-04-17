from torch import nn
from torchvision.models import resnet18


class ResnetModel(nn.Module):
    '''
    This model uses Resnet18 without weights, but
    modifies the initial convolutional layer to
    accommodate the smaller input sizes of CIFAR10
    '''

    NAME = 'resnet18_cifar10'

    def __init__(self,
                 model_cfg,
                 *args,
                 **kwargs):
        '''
        No params are passed to this model
        '''
        super().__init__(*args, **kwargs)

        self.num_classes = model_cfg.params.num_classes

        # load the architecture (no weights)
        self.net = resnet18(weights=None,
                            num_classes=model_cfg.params.num_classes)

        # given the smaller size of the input images (32x32 instead of 224x224)
        # make the initial convolutional layer to be smaller.
        self.net.conv1 = nn.Conv2d(3, 64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        '''
        The forward pass through the network.

        `x` is a batch of images,
        e.g. x.shape = [batch_size. C, H, W] where H,W are
        image height and width.
        '''
        return self.net(x)
