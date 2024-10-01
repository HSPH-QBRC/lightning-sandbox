import sys

from lr_schedulers.cosine_annealing import CosineAnnealing
from lr_schedulers.onecycle import OneCycle

AVAILABLE_SCHEDULERS = {
    'onecycle': OneCycle,
    'cosine_annealing': CosineAnnealing
}


def load_scheduler(scheduler_cfg, optimizer, trainer):
    '''
    Creates/returns an instance of a learning rate scheduler.

    Since some schedulers depend on things like the training set
    size or epochs, we pass a pytorch_lightning.Trainer instance
    '''
    scheduler_name = scheduler_cfg.name
    try:
        lr_scheduler_cls = AVAILABLE_SCHEDULERS[scheduler_name]
    except KeyError as ex:
        sys.stderr.write('The learning rate scheduler identified by'
                         f' the name {scheduler_name} was not found')
        raise ex

    return lr_scheduler_cls(scheduler_cfg, optimizer, trainer)
