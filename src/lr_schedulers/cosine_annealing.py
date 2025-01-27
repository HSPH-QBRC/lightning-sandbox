from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealing(CosineAnnealingLR):
    '''
    This is a thin wrapper around a torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler which allows us to pass information about the
    training set details. The wrapping permits us to more generically
    instantiate class instances in this package's __init__.py
    '''

    def __init__(self, scheduler_cfg, optimizer, trainer):

        # only need to pass the params that were provided in the yaml
        params = dict(scheduler_cfg.params)
        super().__init__(optimizer, **params)