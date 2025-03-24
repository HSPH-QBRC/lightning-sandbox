import numpy as np
from pytorch_lightning import LightningModule
import torch
from torchmetrics import Accuracy
from torch.nn.functional import one_hot

from optimizers import load_optimizer_and_lr_scheduler
from checkpoints.basic_classifier import BasicClassifierCheckpoint


class TCIADLBCLModule(LightningModule):

    NAME = 'tcia_dlbcl'

    def __init__(self, model, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = cfg
        self.debug = self.config.debug
        self.model = model
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.output_channels = self.config.model.params.output_channels

        self.train_acc = Accuracy(task="multiclass",
                                  num_classes=self.config.dataset.num_stages)
        self.valid_acc = Accuracy(task="multiclass",
                                  num_classes=self.config.dataset.num_stages)
        self.test_acc = Accuracy(task="multiclass",
                                  num_classes=self.config.dataset.num_stages)
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        training.

        `batch` is a tuple of inputs and targets.
        '''
        imgs, target_meta = batch
        logits, loss = self._batchstep(imgs, target_meta, batch_idx)

        # track our training accuracy:
        self._calculate_accuracy(logits, target_meta, 'train_acc', self.train_acc)

        return loss
        
    def validation_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        validation.

        `batch` is a tuple of inputs and targets.
        '''
        imgs, target_meta = batch
        logits, validation_loss = self._batchstep(imgs, target_meta, batch_idx)

        # track our validation accuracy:
        self._calculate_accuracy(logits, target_meta, 'val_acc', self.valid_acc)

        # specific to validation, track the loss so we can determine
        # if overfitting
        self.log('val_loss', validation_loss)

    def test_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        test.
        '''
        imgs, target_meta = batch
        logits, test_loss = self._batchstep(imgs, target_meta, batch_idx)
                
        self._calculate_accuracy(logits, target_meta, 'test_acc', self.test_acc)

        self.log('test_loss', test_loss)
        
    def predict_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        prediction.
        '''
        imgs, _ = batch
        logits = self.model(imgs)
        predictions = self._make_prediction(logits)
        return predictions

    def _create_target(self, y):
        '''
        Depending on the model used, we can alter how the 
        target is represented. 
        '''
        # each of those is some iterable with batch-size length:
        stages, ipi_scores, ipi_risk_groups, img_ids = y

        if self.output_channels == 4:
            return one_hot(stages.int(), 4)
        else:
            raise NotImplementedError('Labels are not available for'
                                        ' this number of output channels')

    def _batchstep(self, imgs, target_meta, batch_idx):
        '''
        A "private" method which performs a forward pass and
        returns the logits and loss. Used by the training/validation/test
        step method overrides

        `x` is a batch of images (size B)
        `y` is a vector of targets/outputs (e.g. the class we are
            trying to predict).

        Note that given the DataSet we have, `y` is a tuple of
        (stages, ipi_scores, ipi_risk_groups, img_ids) 
        '''
        logits = self.model(imgs)
        targets = self._create_target(target_meta)
        return logits, self.loss_fn(logits, targets)

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

    def _make_prediction(self, logits):
        '''
        Method used for making a grade prediction using the passed logits.

        Note that in the model, the final layer has some specified number
        of channels (e.g. 5 or 10). In the loss function
        (`nn.BCEWithLogitsLoss`), it takes those logits and performs a 
        sigmoid operation (so all entries are in the interval of [0,1]).
        It then calculates the BCE loss compared to the targets 
        (where we have encoded grade 3 as [1,1,1,0,0]).

        Hence, to make the predictions, we have to take the logits
        and run through a sigmoid. Then, that is summed to produce
        some number in the range [0,5]. Those are then rounded
        '''
        if self.output_channels == 4:
            return logits.sigmoid().sum(axis=1).round()
        else:
            raise NotImplementedError('Only handles 4 channels')

    def _calculate_accuracy(self, logits, target_meta, metric_key, metric):
        '''
        Since the train/validation/test steps all collect
        accuracy metrics, we create a single method to
        handle this.

        `logits` is the direct output of self.model (i.e. no sigmoid, etc.)
        `target_meta` is the full tuple of "output metadata" from the 
            underlying DataSet. For instance, it can contain isup_grades,
            gleason_scores, etc. 
        `metric_key` is the name of the metric that we are
            collecting via LightningModule's self.log method
        `metric` is the actual metric we are computing.
        '''
        # We track the performance by measuring the grade
        # prediction accuracy.
        stages, _, _, _ = target_meta

        predictions = self._make_prediction(logits)
        metric(predictions, stages)
        self.log(metric_key, metric)

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
