from pathlib import Path

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
import skimage.io
from torch.utils.data import DataLoader, \
    Dataset

from utils.image_utils import extract_tiles


class WmSlideInferenceDataset(Dataset):
    '''
    This model is based off the one used to train the base model
    for the PANDAS challenge.

    NOTE that it does NOT implement any training-based
    behavior since this is currently only used for 
    inference.
    '''

    def __init__(self, dataset_cfg, phase, image_meta_df, transform):
        super().__init__()

        self.phase = phase
        self.transform = transform

        self.image_meta_df = image_meta_df

        # how many tiles are in the 'composite' image
        self.num_tiles = dataset_cfg.num_tiles

        # number of pixels in the individual tiles
        self.tile_size = dataset_cfg.tile_size

        # size of the composite image
        self.img_size = dataset_cfg.img_size

        # resolution level of the tiled images (which level of
        # the TIFF they were extracted from)
        self.img_resolution = dataset_cfg.resolution

        # assert that the number of tiles, tile size, and image size are compatible
        assert(np.allclose(
            self.img_size,
            np.sqrt(self.num_tiles) * self.tile_size
        ))

        # For inference, we will only use the '0' mode of the extracted tiles.
        tile_mode = 0
        tiles_root_dir = dataset_cfg.base_dir
        self.input_tiles_dir = f'{tiles_root_dir}/tiles_res{self.img_resolution}_mode{tile_mode}_num{self.num_tiles}_size{self.tile_size}'
            

    def __len__(self):
        return self.image_meta_df.shape[0]

    def __getitem__(self, idx):

        row = self.image_meta_df.iloc[idx]

        tiles = self._get_tiles(row['image_id'])
        image = self._concat_tiles(tiles)

        # performs normalization, etc. - no actual transformations
        image = self._transform_composite(image)
        augmented = self.transform(image=image)
        image = augmented["image"]
        # return the image PLUS some metadata- the model
        # will handle how this additional metadata is encoded
        # or used
        return image, row['image_id']

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
        return np.hstack([np.vstack(tiles[ts]) for ts in grid_idx])

    def _get_tiles_from_paths(self, paths):
        '''
        Given the list of paths (length n), return a n-length numpy array.
        Each item in that n-length array is itself a 2-d numpy array 
        representing a tile
        '''
        return np.array([skimage.io.imread(p) for p in paths])

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        paths = [
            Path(f'{self.input_tiles_dir}/{image_id}.tile_{i}.png')
            for i in range(self.num_tiles)
        ]
        return self._get_tiles_from_paths(paths)


class WmSlideInferenceDataModule(LightningDataModule):

    NAME = 'wm_slides'

    def __init__(self, dataset_cfg):
        super().__init__()

        self.dataset_cfg = dataset_cfg

        # extract parameters related to data loading:
        self.batch_size = dataset_cfg.batch_size
        self.num_workers = dataset_cfg.num_workers
        self.seed = dataset_cfg.seed

        # create transform/augmentations for each phase
        self._set_transforms()

        # load the image metadata (which has the training folds).
        # Used when creating train/validation splits in the `setup`
        # method
        self.image_meta_path = self.dataset_cfg.image_meta_path
        self.image_meta_df = pd.read_table(self.image_meta_path)

    def prepare_data(self):
        pass

    def _create_transforms(self, aug_cnf):
        '''
        Helper method which returns a list of augmentations based
        on the `aug_cnf` config
        '''
        # for now, return an empty list does nothing
        return []

    def _get_transforms_for_phase(self, phase):
        augmentations_cfg = None
        transform_list = self._create_transforms(augmentations_cfg)
        transform_list.append(ToTensorV2())
        return alb.Compose(transform_list)

    def _set_transforms(self):
        '''
        Called by __init__ to set attributes for train/validation/test
        transformations/augmentations
        '''          
        self._test_transforms = self._get_transforms_for_phase('test')

    def setup(self, stage):

        if stage == 'predict':
            self.test_dataset = WmSlideInferenceDataset(self.dataset_cfg,
                                              stage,
                                              self.image_meta_df,
                                              self._test_transforms)
        else:
            raise Exception('This data module only performs inference, yet'
                            f' it was passed stage={stage}')

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size)
