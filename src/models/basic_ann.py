from torch import nn


class BasicAnnModel(nn.Module):
    '''
    This is a simple implementation of a basic artificial
    neural net.

    This can be used as a simple/quick dummy model to ensure
    that the various mechanics of training/validating/testing
    a model are working properly.

    For simplicity, this model assumes we are training a simple
    classifier (finite number of distinct, labeled classes) on greyscale
    images (e.g. channel depth of 1)
    '''

    NAME = 'basic_ann'

    def __init__(self,
                 model_cfg,
                 *args,
                 **kwargs):
        '''
        Required params:
        `img_height`: The image height in pixels
        `img_width`: The image width in pixels
        `num_classes`: The number of classes to map to for this classifier
        '''
        super().__init__(*args, **kwargs)

        self.num_classes = model_cfg.params.num_classes
        self.net = nn.Sequential(
            nn.Linear(
                model_cfg.params.img_height * model_cfg.params.img_width,
                64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        '''
        The forward pass through the network.

        `x` is a batch of images,
        e.g. x.shape = [batch_size, H, W] where H,W are
        image height and width.

        The net expects the images to be flattened, so we
        apply that below
        '''
        # flatten the input tensor from (B,H,W) to (B,H*W)
        x = x.view(x.size(0), -1)
        return self.net(x)
