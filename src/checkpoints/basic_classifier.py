from lightning.pytorch.callbacks import ModelCheckpoint


class BasicClassifierCheckpoint(ModelCheckpoint):
    '''
    A simple checkpoint implementation to be paired with the
    custom_lightning_modules.basic_classifier.BasicClassifierModule
    lightning module.

    This monitors the validation accuracy.
    '''
    def __init__(self, *args):
        monitor = 'val_acc'
        mode = 'max'  # since we are monitoring the accuracy, we want the max
        filename = '{epoch}-{val_acc:.2f}'
        super().__init__(filename=filename, monitor=monitor, mode=mode, *args)
