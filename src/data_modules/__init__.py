import sys

from .mnist import MNISTDataModule
from .cifar10 import CIFAR10DataModule
from .pandas_challenge import PandasDataModule
from .wm_slides import WmSlideInferenceDataModule
from .dlbcl_slides import DLBCLDataModule
from .dlbcl_premade_patches import PremadeDLBCLPatchesDataModule

DATASET_LIST = [
    MNISTDataModule,
    CIFAR10DataModule,
    PandasDataModule,
    WmSlideInferenceDataModule,
    DLBCLDataModule,
    PremadeDLBCLPatchesDataModule
]

AVAILABLE_DATASETS = {x.NAME: x for x in DATASET_LIST}


def load_dataset(dataset_cfg):
    '''
    Loads/returns the dataset dictated by the dataset_cfg configuration object
    '''
    try:
        dataset_class = AVAILABLE_DATASETS[dataset_cfg.dataset_name]
    except KeyError:
        sys.stderr.write('Could not locate dataset identified by'
                         f' {dataset_cfg.dataset_name}. Available names are'
                         f' {",".join(AVAILABLE_DATASETS.keys())}')
        sys.exit(1)
    return dataset_class(dataset_cfg)
