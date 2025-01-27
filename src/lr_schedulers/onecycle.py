from torch.optim.lr_scheduler import OneCycleLR


class OneCycle(OneCycleLR):
    '''
    This is a thin wrapper around a torch.optim.lr_scheduler.OneCycleLR
    scheduler which allows us to pass information about the
    training set details. The wrapping permits us to more generically
    instantiate class instances in this package's __init__.py
    '''

    def __init__(self, scheduler_cfg, optimizer, trainer):
        train_dl = trainer.datamodule.train_dataloader()
        n_train = len(train_dl.dataset)
        batch_size = train_dl.batch_size
        steps_per_epoch = n_train // batch_size
        total_epochs = trainer.max_epochs

        # "dynamic" params (i.e. determined by the training set):
        params = {
            'epochs': total_epochs,
            'steps_per_epoch': steps_per_epoch,
        }

        # now update the params with those passed from the config:
        params.update(**scheduler_cfg.params)

        super().__init__(optimizer, **params)
