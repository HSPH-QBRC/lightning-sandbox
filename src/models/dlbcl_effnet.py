import torch
from torch import nn
from torchvision.models import efficientnet_b1

from models.submodules import GeM, SEBlock


class EfficientNetB1Model(nn.Module):
    '''
    This model class extends the base EfficientNet-B1 model
    and replaces various components (e.g. avg pool - GeM)
    '''

    NAME = 'dlbcl_efficientnet_b1'

    def __init__(self,
                 model_cfg,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        dropout = model_cfg.params.dropout
        out_features = model_cfg.params.output_channels

        # instantiate the base EfficientNet-B1. Initlally load the imagenet weights
        # and later we can replace those if passed the proper key in the model config
        # (see below)
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

        # if model weights were provided in the config (e.g. from a prior training),
        # then we overwrite here:
        if model_cfg.checkpoint is not None:
            self.load_model_weights(model_cfg.checkpoint)

    def forward(self, x):
        '''
        The forward pass through the network.

        `x` is a batch of images,
        e.g. x.shape = [batch_size. C, H, W] where H,W are
        image height and width.
        '''
        return self.net(x)
    
    def load_model_weights(self, checkpoint_path, prefix='model.net'):
        '''
        Loads the weights from training into the provided model. 

        Note that as written, as expect this to be an efficientnet-b1 model
        that has a different size classifier head on it.

        The `prefix` kwarg allows us to grab the weights from the pre-trained
        model and put them into this model: based on how we embed a vanilla
        nn.Module inside a lightning module, it ends up putting a prefix
        on the various module names
        '''
        checkpoint_data = torch.load(checkpoint_path)

        # The model weights have a 'prefix' which we need to strip
        weights = {k[len(prefix)+1:]: v for k, v in checkpoint_data["state_dict"].items() if k.startswith(f"{prefix}.")}

        # get the random weights that were populated into our current model and use them to overwrite
        # those in the `weights` dict.
        current_state_dict = self.net.state_dict()
        weights['classifier.2.weight']  = current_state_dict.pop('classifier.2.weight')
        weights['classifier.2.bias']  = current_state_dict.pop('classifier.2.bias')

        self.net.load_state_dict(weights)
