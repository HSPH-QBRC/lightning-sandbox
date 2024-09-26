from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy



class PandasModule(LightningModule):

    NAME = 'pandas'

    def __init__(self, model, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = cfg
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        training.

        `batch` is a tuple of inputs and targets.
        '''
        # need to unpack the batch
        x, y = batch

    def validation_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        validation.

        `batch` is a tuple of inputs and targets.
        '''

        # unpack the batch and run the forward pass through the model
        x, y = batch

    def configure_optimizers(self):
        pass