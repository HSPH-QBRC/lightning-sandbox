from lightning.pytorch.callbacks import ModelCheckpoint


class PandasChallengeCheckpoint(ModelCheckpoint):
    '''
    A simple checkpoint implementation to be paired with the
    custom_lightning_modules.pandas_challenge.PandasModule
    lightning module.

    This monitors the validation accuracy.
    '''
    def __init__(self, *args):
        super().__init__(
            filename='{epoch}-{val_acc:.2f}',
            monitor='val_acc',
            # since we are monitoring the accuracy, we want the max
            mode='max',
            save_last=True,
            save_top_k=3,
            *args)
