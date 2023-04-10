import sys

from .mnist import MNISTDataModule

DATASET_LIST = [
    MNISTDataModule
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
    return dataset_class(dataset_cfg)
