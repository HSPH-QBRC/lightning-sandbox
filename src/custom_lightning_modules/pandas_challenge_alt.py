import torch

from .pandas_challenge import PandasModule


class PandasAltModule(PandasModule):
    '''
    Matches the functionality of the PandasModule, but changes
    the target and loss functions
    '''

    NAME = 'pandas_challenge_alternate'

    def __init__(self, model, cfg, *args, **kwargs):
        super().__init__(model, cfg, *args, **kwargs)

        # overwrite the loss_fn from the parent class
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def _create_target(self, y):
        # each of those is some iterable with batch-size length:
        isup_grades, gleason_scores, data_providers, img_ids = y

        if self.output_channels == 6:
            # isup_grades is {0,...,5}
            return isup_grades.to(dtype=torch.int64)
        else:
            raise NotImplementedError('Labels are not available for'
                                        ' this number of output channels')
        
    def _make_prediction(self, logits):
        '''
        Takes the logits and makes a prediction
        '''
        return logits.argmax(dim=1)