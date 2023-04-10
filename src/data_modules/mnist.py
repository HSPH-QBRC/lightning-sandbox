from pathlib import Path

from hydra.utils import get_original_cwd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):

    NAME = 'mnist'

    def __init__(self, dataset_cfg):
        super().__init__()

        # note that since we have hydra managing config,
        # the working directory *might* be different for
        # each run. Make the data directory outside the
        # run-specific directory so we that we are not
        # constantly downloading data
        self.data_dir = Path(get_original_cwd()) / \
            Path(dataset_cfg.base_dir) / Path(MNISTDataModule.NAME)
        self.batch_size = dataset_cfg.batch_size
        self.num_workers = dataset_cfg.num_workers
        self.transformations = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self) -> None:
        '''
        This method is used to download data and should NOT assign
        instance state per the docs. In the case of single-node training, etc.
        this doesn't matter, but is important for multi-node/distributed
        implementations
        '''
        MNIST(self.data_dir, download=True, train=True)
        MNIST(self.data_dir, download=True, train=False)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            full_dataset = MNIST(self.data_dir, train=True,
                                 transform=self.transformations)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [0.8, 0.2])

        if stage == 'test':
            self.test_dataset = MNIST(self.data_dir, train=False,
                                      transform=self.transformations)

        if stage == 'predict':
            self.test_dataset = MNIST(self.data_dir, train=False,
                                      transform=self.transformations)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
