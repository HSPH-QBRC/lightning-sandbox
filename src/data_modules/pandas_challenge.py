from pathlib import Path

import numpy as np
import pandas as pd

import cv2

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch import Generator
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2 as v2_transforms
from torch.utils.data import Dataset

from image_utils import extract_tiles


class PandasDataset(Dataset):

    def __init__(self, dataset_cfg, phase, transform):
        super().__init__()

        self.phase = phase
        self.transform = transform

        # how many tiles are in the 'composite' image
        self.num_tiles = dataset_cfg.num_tiles

        # number of pixels in the individual tiles
        self.tile_size = dataset_cfg.tile_size

        # size of the composite image
        self.img_size = dataset_cfg.img_size

        # which fold is being held-out
        self.kfold = dataset_cfg.kfold

        # CSV-format file giving the image ID, source, Gleason scores, etc.
        self.image_meta_path = dataset_cfg.image_meta_path
        self.image_meta_df = pd.read_csv(self.image_meta_path)

        self.randomize_tiles = dataset_cfg.randomize_tiles
        self.random_rotate_tile = dataset_cfg.random_rotate_tile
        self.random_tile_white = dataset_cfg.random_tile_white

        # assert that the number of tiles, tile size, and image size are compatible
        assert(np.allclose(
            self.img_size,
            np.sqrt(self.num_tiles) * self.tile_size
        ))

        # we have two directories where we have extracted tiles. They differ
        # based on the initial tile positioning. This is the "offset mode"
        self.train_input_tile_dirs = [
            # TODO change
            f'../input/numtile-{self.num_tiles}-tilesize-{self.tile_size}-res-1-offset-mode-{m}'
            for m in [0, 2]
        ]

    def __len__(self):
        return self.image_meta_df.shape[0]

    def __getitem__(self, idx):

        row = self.image_meta_df.iloc[idx]

        tiles = self._get_tiles(row['image_id'])
        image = self._concat_tiles_with_augmentation(
            tiles,
            random_tile=self.randomize_tiles,
            random_rotate=self.random_rotate_tile,
            random_tile_white=self.random_tile_white)

        # performs normalization, etc. - no actual transformations
        image = self._transform_composite(image)

        # self.transform is an instance of alb.Compose
        # and requires keyword args
        augmented = self.transform(image=image)
        image = augmented["image"]

        data_provider = row['data_provider']
        isup_grade = row['isup_grade']
        gleason_score = row['gleason_score']

        # return the image PLUS some metadata- the model
        # will handle how this additional metadata is encoded
        # as the target, etc.
        return image, (isup_grade, gleason_score, data_provider)

    def _transform_composite(self, img):
        '''
        Performs some pre-processing/normalization on
        the composite image (e.g. the concatenated tiles)
        '''
        img = img.astype(np.float32)
        img = 255 - img
        img /= 255 
        return img

    def _concat_tiles(self, tiles):
        '''
        Arranges the tiles into a grid
        '''
        n = int(np.sqrt(self.num_tiles))
        grid_idx = np.arange(self.num_tiles).reshape(n, n)
        return cv2.hconcat([cv2.vconcat(tiles[ts]) for ts in grid_idx])

    def _tile_augmentation(self,
        tiles,
        random_tile = False,
        random_rotate = False,
        random_tile_white = False):
        '''
        Applies augmentations to the tiles themselves. Note that
        the composite image (once the tiles are stitched together)
        will have its own set of potential augmentations
        '''

        if random_rotate and np.random.rand() < 0.5:
            # note:
            # >>> cv2.ROTATE_90_CLOCKWISE
            # 0
            # >>> cv2.ROTATE_180
            # 1
            # >>> cv2.ROTATE_90_COUNTERCLOCKWISE
            # 2
            # So this dictates those three rotations and zero rotation
            indxs = np.random.randint(0, 4, len(tiles))
            for i, indx in enumerate(indxs):
                if indx != 3:
                    tiles[i] = cv2.rotate(tiles[i], indx)

        # randomly sorts the images
        if random_tile and np.random.rand() < 0.5:
            indxs = np.random.permutation(len(tiles))
            tiles = tiles[indxs]

        # picks one random tile and sets it to white
        if random_tile_white and np.random.rand() < 0.5:
            indx = np.random.randint(len(tiles))
            tiles[indx][:] = 255  # White

        return tiles

    def _concat_tiles_with_augmentation(self,
        tiles,
        random_tile = False,
        random_rotate = False,
        random_tile_white = False):

        # randomly rotate or permutate the individual tiles. If all kwargs
        # are false, then nothing is performed on the tiles
        tiles = self._tile_augmentation(
            tiles,
            random_tile=random_tile,
            random_rotate=random_rotate,
            random_tile_white=random_tile_white,
        )
        return self._concat_tiles(tiles)

    def _get_tiles_from_paths(self, paths):
        '''
        Given the list of paths (length n), return a n-length numpy array.
        Each item in that n-length array is itself a 2-d numpy array 
        representing a tile
        '''
        # cv2.imread is in BGR so we convert:
        tiles = np.array([
            cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths])
        return tiles

    def _get_input_tile_dir(self):
        '''
        There are multiple folders which can contain different versions
        of the tiles extracted from the original images. This method
        contains the logic for that.
        '''
        if self.phase == 'fit':
            if np.random.rand() < 0.5:
                return self.train_input_tile_dirs[0]
            else:
                return self.train_input_tile_dirs[1]
        elif self.phase == 'validate':
            return self.train_input_tile_dirs[0]
        else: # test/predict case
            return self.test_image_dirs

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        img_dir = self._get_input_tile_dir()
        if self.phase in ['fit', 'validate']:
            paths = [
                Path(f'{img_dir}/{image_id}_{i}.png')
                for i in range(self.num_tile)
            ]
            return self._get_tiles_from_paths(paths)
        else: # test/predict
            img_path = Path(f'{img_dir}/{image_id}.tiff')         
            tiles, _ = extract_tiles(img_path, 
                                     num_tiles=self.num_tiles,
                                     tile_size=self.tile_size)
            return tiles


class PandasDataModule(LightningDataModule):

    NAME = 'pandas'

    def __init__(self, dataset_cfg):
        super().__init__()

        self.dataset_cfg = dataset_cfg

        # extract parameters related to data loading:
        self.batch_size = dataset_cfg.batch_size
        self.num_workers = dataset_cfg.num_workers
        self.seed = dataset_cfg.seed
        self.train_fraction = dataset_cfg.train_fraction
        self.validation_fraction = 1 - self.train_fraction

        # create transform/augmentations for each phase
        self._set_transforms(dataset_cfg)

    def _create_transforms(self, aug_cnf):
        '''
        Helper method which returns a list of augmentations based
        on the `aug_cnf` config
        '''
        # internal function to assist with creating the transforms
        def get_object(transform_spec):
            if transform_spec.name in {"Compose", "OneOf"}:
                augs_tmp = [get_object(aug) for aug in transform_spec.options]
                return getattr(alb, transform_spec.name)(augs_tmp, **transform_spec.params)

            try:
                return getattr(alb, transform_spec.name)(**transform_spec.params)
            except Exception as ex:
                print('Could not find the prescribed transform')
                raise ex

        if aug_cnf is None:
            augs = []
        else:
            augs = [get_object(aug) for aug in aug_cnf]
        return augs

    def _get_transforms_for_phase(self, dataset_cfg, phase):
        key = f'{phase}_augmentations'
        if key in dataset_cfg:
            augmentations_cfg = dataset_cfg[key]
        else:
            augmentations_cfg = None
        transform_list = self._create_transforms(augmentations_cfg)
        transform_list.append(ToTensorV2())
        return alb.Compose(transform_list)

    def _set_transforms(self, dataset_cfg):
        '''
        Called by __init__ to set attributes for train/validation/test
        transformations/augmentations
        '''          
        self._train_transforms = self._get_transforms_for_phase(dataset_cfg, 'fit')
        self._validation_transforms = self._get_transforms_for_phase(dataset_cfg, 'validate')
        self._test_transforms = self._get_transforms_for_phase(dataset_cfg, 'test')

    def setup(self, stage):
        
        if stage == 'fit' or stage is None:
            full_train_dataset = PandasDataset(self.dataset_cfg,
                                               stage,
                                               self._train_transforms)
            full_validation_dataset = PandasDataset(self.dataset_cfg,
                                                    stage,
                                                    self._validation_transforms)

            # now do a train/validation split:
            split_fractions = [self.train_fraction, self.validation_fraction]
            self.train_dataset, _ = random_split(
                full_train_dataset, split_fractions,
                generator=Generator().manual_seed(self.seed))
            _, self.val_dataset = random_split(
                full_validation_dataset, split_fractions,
                generator=Generator().manual_seed(self.seed))

        if stage == 'test' or stage == 'predict':
            self.test_dataset = PandasDataset(self.dataset_cfg,
                                              stage,
                                              self._test_transforms)

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
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size)