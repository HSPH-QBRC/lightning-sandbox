import sys

import torch.optim

from lr_schedulers import load_scheduler


def load_optimizer_and_lr_scheduler(model_params, cfg, trainer):
    '''
    Given the configuration object (omegaconf.DictConfig) and
    pytorch_lightning.Trainer instance,
    return an optimizer and learning rate scheduler as a tuple
    '''

    # we require an optimizer
    if 'optimizer' not in cfg:
        sys.stderr.write('You need to specify an optimizer.')
        sys.exit(1)

    optimizer_name = cfg.optimizer.name
    try:
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(model_params, **cfg.optimizer.params)
    except AttributeError as ex:
        sys.stderr.write('Could not locate an optimizer with'
                         f' name {optimizer_name}')
        sys.exit(1)

    # learning-rate schedulers are NOT required, so they might not be included
    if 'lr_scheduler' in cfg:
        lr_scheduler = load_scheduler(cfg.lr_scheduler, optimizer, trainer)
    else:
        lr_scheduler = None

    return (optimizer, lr_scheduler)
