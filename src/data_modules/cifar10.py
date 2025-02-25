from pathlib import Path

from hydra.utils import get_original_cwd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch import Generator
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):

    NAME = 'cifar10'

    def __init__(self, dataset_cfg):
        super().__init__()

        # note that since we have hydra managing config,
        # the working directory *might* be different for
        # each run. Make the data directory outside the
        # run-specific directory so we that we are not
        # constantly downloading data
        try:
            self.data_dir = Path(get_original_cwd()) / \
                Path(dataset_cfg.base_dir) / Path(CIFAR10DataModule.NAME)
        except ValueError:
            self.data_dir = Path.cwd().parent / \
                Path(dataset_cfg.base_dir) / Path(CIFAR10DataModule.NAME)
        self.batch_size = dataset_cfg.batch_size
        self.num_workers = dataset_cfg.num_workers
        self.seed = dataset_cfg.seed
        self.train_fraction = dataset_cfg.train_fraction
        self.validation_fraction = 1 - self.train_fraction
        self._set_transforms()

    def _set_transforms(self):
        '''
        Apply transforms for training, validation, and testing
        '''
        # taken from https://github.com/Lightning-Universe/lightning-bolts/\
        # blob/master/pl_bolts/transforms/dataset_normalizations.py
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        self._train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        self._validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self._test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def prepare_data(self):
        '''
        This method is used to download data and should NOT assign
        instance state per the docs. In the case of single-node training, etc.
        this doesn't matter, but is important for multi-node/distributed
        implementations
        '''
        CIFAR10(self.data_dir, download=True, train=True)
        CIFAR10(self.data_dir, download=True, train=False)

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            # we first get the same datasets for train and validation, with
            # the only difference being the applied transforms.
            full_train_dataset = CIFAR10(self.data_dir, train=True,
                                         transform=self._train_transforms)
            full_validation_dataset = CIFAR10(self.data_dir, train=True,
                                              transform=self._validation_transforms)

            # we next split both "full" datasets to get our final train
            # and validation sets. Note that we use the same generator for
            # both so that we are not leaking training data into the
            # validation set
            split_fractions = [self.train_fraction, self.validation_fraction]
            self.train_dataset, _ = random_split(
                full_train_dataset, split_fractions,
                generator=Generator().manual_seed(self.seed))
            _, self.val_dataset = random_split(
                full_validation_dataset, split_fractions,
                generator=Generator().manual_seed(self.seed))

        if stage == 'test':
            self.test_dataset = CIFAR10(self.data_dir, train=False,
                                        transform=self._test_transforms)

        if stage == 'predict':
            self.test_dataset = CIFAR10(self.data_dir, train=False,
                                        transform=self._test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
