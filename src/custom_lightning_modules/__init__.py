import sys

from .basic_classifier import BasicClassifierModule
from .pandas_challenge import PandasModule
from .pandas_challenge_alt import PandasAltModule
from .dlbcl_slides import TCIADLBCLModule
from .wm_slides import WMBinaryModule


PL_MODULE_LIST = [
    BasicClassifierModule,
    PandasModule,
    TCIADLBCLModule,
    PandasAltModule,
    WMBinaryModule
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
        sys.exit(1)
    return pl_module_class(selected_model, cfg)
