import sys

from .basic_ann import BasicAnnModel
from .resnet import ResnetModel
from .pandas_effnet import PandasEfficientNetB1Model
from .dlbcl_effnet import DLBCLEfficientNetB1Model
from .wm_binary_effnet import WMBinaryEfficientNetB1Model

MODEL_LIST = [
    BasicAnnModel,
    ResnetModel,
    PandasEfficientNetB1Model,
    DLBCLEfficientNetB1Model,
    WMBinaryEfficientNetB1Model
]

AVAILABLE_MODELS = {x.NAME: x for x in MODEL_LIST}


def load_model(model_cfg):
    '''
    Loads/returns the model dictated by the model_cfg configuration object
    '''
    try:
        model_class = AVAILABLE_MODELS[model_cfg.model_name]
    except KeyError:
        sys.stderr.write('Could not locate model identified by'
                         f' {model_cfg.model_name}. Available names are'
                         f' {",".join(AVAILABLE_MODELS.keys())}')
        sys.exit(1)
    return model_class(model_cfg)
