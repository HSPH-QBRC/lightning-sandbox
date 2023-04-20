import sys

from .basic_classifier import BasicClassifierModule

PL_MODULE_LIST = [
    BasicClassifierModule
]

AVAILABLE_MODULES = {x.NAME: x for x in PL_MODULE_LIST}


def load_pl_module(cfg, selected_model):
    '''
    Loads/returns the LightningModule subclass dictated by the
    `cfg` configuration object.

    `selected_model` is an instance of torch.nn.Module and
    is passed to the constructor of the LightningModule
    '''
    try:
        pl_module_class = AVAILABLE_MODULES[cfg.pl_module.name]
    except KeyError:
        sys.stderr.write('Could not locate a LightningModule subclass'
                         f'  identified by{cfg.pl_module.name}. Available'
                         f' names are {",".join(AVAILABLE_MODULES.keys())}')
    return pl_module_class(selected_model, cfg)
