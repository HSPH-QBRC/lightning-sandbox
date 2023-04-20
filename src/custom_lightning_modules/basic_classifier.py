from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from checkpoints.basic_classifier import BasicClassifierCheckpoint
from optimizers import load_optimizer_and_lr_scheduler


class BasicClassifierModule(LightningModule):
    '''
    This LightningModule can be used with a simple classifier model
    and uses a cross-entropy loss function.
    '''

    NAME = 'basic_classifier'

    def __init__(self, model, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = cfg
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass",
                                  num_classes=self.model.num_classes)
        self.valid_acc = Accuracy(task="multiclass",
                                  num_classes=self.model.num_classes)

    def configure_optimizers(self):
        optimizer, lr_scheduler = load_optimizer_and_lr_scheduler(
            self.parameters(), self.config, self.trainer)

        if lr_scheduler is not None:
            additional_params = {}
            if 'scheduler_config' in self.config.lr_scheduler:
                additional_params = self.config.lr_scheduler.scheduler_config
            scheduler_dict = {
                "scheduler": lr_scheduler,
                **additional_params
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return optimizer

    def _batchstep(self, x, y, batch_idx):
        '''
        A "private" method which performs a forward pass and
        returns the logits and loss. Used by the training/validation/test
        step method overrides

        `x` is a batch of images (size B)
        `y` is a vector of targets/outputs (e.g. the class we are
            trying to predict)
        '''
        logits = self.model(x)
        return logits, self.loss_fn(logits, y)

    def _get_predictions(self, logits):
        '''
        A bit of a trivial function, but keeps the prediction
        logic in a single location.

        `logits` is a tensor of size (B, num_classes)
            so finding the largest component along the dim=1
            axis will give an array of length B providing the
            predicted class identities
        '''
        return logits.argmax(dim=1)

    def _calculate_accuracy(self, logits, targets, metric_key, metric):
        '''
        Since the train/validation/test steps all collect
        accuracy metrics, we create a single method to
        handle this.

        `logits` is the direct output of self.model (i.e. no softmax, etc.)
        `metric_key` is the name of the metric that we are
            collecting via LightningModule's self.log method
        `metric` is the actual metric we are computing.
        '''
        # We track the performance by measuring the class
        # prediction accuracy.
        predictions = self._get_predictions(logits)
        metric(predictions, targets)
        self.log(metric_key, metric)

    def training_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        training.

        `batch` is a tuple of inputs and targets.
        '''
        # need to unpack the batch
        x, y = batch

        # perform the forward pass:
        logits, loss = self._batchstep(x, y, batch_idx)

        # track our training accuracy:
        self._calculate_accuracy(logits, y, 'train_acc', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        validation.

        `batch` is a tuple of inputs and targets.
        '''

        # unpack the batch and run the forward pass through the model
        x, y = batch
        logits, validation_loss = self._batchstep(x, y, batch_idx)

        # track our training accuracy:
        self._calculate_accuracy(logits, y, 'val_acc', self.valid_acc)

        # specific to validation, track the loss so we can determine
        # if overfitting
        self.log('val_loss', validation_loss)

    def predict_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        prediction.

        `batch` is a tuple of inputs and targets.
        '''
        x, y = batch
        logits, _ = self._batchstep(x, y, batch_idx)
        predictions = self._get_predictions(logits)
        return predictions, x, y

    def configure_callbacks(self):
        '''
        Since model-specific callbacks are somewhat tied to
        the model (e.g. monitoring 'val_acc'), define the
        checkpoint(s) here.

        Callbacks defined here will be merged with those passed to
        the Trainer
        See https://lightning.ai/docs/pytorch/stable/\
            common/lightning_module.html#configure-callbacks
        '''
        cb_list = super().configure_callbacks()
        chkpt = BasicClassifierCheckpoint()
        cb_list.append(chkpt)
        return cb_list
