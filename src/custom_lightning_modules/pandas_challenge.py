import numpy as np

from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from optimizers import load_optimizer_and_lr_scheduler
from checkpoints.pandas_challenge import PandasChallengeCheckpoint


class PandasModule(LightningModule):

    NAME = 'pandas_challenge'

    def __init__(self, model, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = cfg
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(task="multiclass",
                                  num_classes=self.config.dataset.num_grades)
        self.valid_acc = Accuracy(task="multiclass",
                                  num_classes=self.config.dataset.num_grades)
        self.test_acc = Accuracy(task="multiclass",
                                  num_classes=self.config.dataset.num_grades)
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

        In the Kaggle competition code, they used a combination 
        of the isup_grade and gleason score to represent a target
        vector.

        The argument `y` corresponds to a batch of outputs, so there are 
        multiple elements.
        '''
        def encode_grades(n, grades):
            '''
            Internal function to encode the grades in the desired
            binary format.

            In the Kaggle work, they take a grade of 3 and encode
            as an array of the form [1,1,1,0,0] (i.e. since the 
            grade is 3, the first 3 entries are 1's). This
            function does that conversion using numpy broadcasting
            '''
            v = np.arange(n)[np.newaxis, :]
            grades = grades[:, np.newaxis]
            return (v < grades).astype(np.int32)

        # each of those is some iterable with batch-size length:
        isup_grades, gleason_scores, data_providers = y

        output_channels = self.config.model.params.output_channels

        if output_channels == 10:
            gleason_first = np.array([int(x.split("+")[0]) for x in gleason_scores])
            m1 = encode_grades(5, isup_grades)
            m2 = encode_grades(5, gleason_first)
            return np.concatenate([m1,m2], axis=1)
        elif output_channels == 5:
            return encode_grades(5, isup_grades)
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
        (isup_grade, gleason_score, data_provider) 
        '''
        logits = self.model(imgs)
        targets = self._create_target(target_meta)
        return logits, self.loss_fn(logits, targets)

    def configure_optimizers(self):
        optimizer, lr_scheduler = load_optimizer_and_lr_scheduler(
            self.parameters(), self.config, self.trainer)

        # Note that in the Kaggle impl, they used a GradualWarmupScheduler
        # which wraps the 'default' scheduler, e.g.
        # scheduler = GradualWarmupScheduler(
        #         optimizer,
        #         multiplier=10,
        #         total_epoch=1,
        #         after_scheduler=scheduler_default)
        # Unless we run into problems, leave this out as it introduces
        # a dependency from an old/unmaintained project

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
        return logits.sigmoid().sum(axis=1).round()

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
        isup_grades, _, _ = target_meta

        predictions = self._make_prediction(logits)
        metric(predictions, isup_grades)
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
        chkpt = PandasChallengeCheckpoint()
        cb_list.append(chkpt)
        return cb_list