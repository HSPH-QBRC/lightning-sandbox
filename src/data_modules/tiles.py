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


class TileBasedDataset(Dataset):
    '''
    This is a base class for Datasets that are 
    constructed by loading and arranging tiles
    in a grid that were previously extracted from a large
    image such as SVS or TIFF. After the tiles are concatenated
    into a single image, we essentially have a large super-image
    '''

    def __init__(self, dataset_cfg, phase, image_meta_df, transform):
        super().__init__()

        self.phase = phase
        self.transform = transform

        # the directory where the tiles are stored
        self.base_input_tile_dir = dataset_cfg.base_dir

        # how many tiles are in the 'composite' image
        self.num_tiles = dataset_cfg.num_tiles

        # number of pixels in the individual tiles
        self.tile_size = dataset_cfg.tile_size

        # size of the composite image
        self.img_size = dataset_cfg.img_size

        # resolution level of the tiled images (which level of
        # the TIFF they were extracted from)
        self.img_resolution = dataset_cfg.resolution

        # Dataframe giving the metadata such as image ID, staging, etc.
        # Specific requirements of this dataframe can be specified in
        # child classes, 
        self.image_meta_df = image_meta_df

        if self.phase == 'fit':
            self.randomize_tiles = dataset_cfg.randomize_tiles
            self.random_rotate_tile = dataset_cfg.random_rotate_tile
            self.random_tile_white = dataset_cfg.random_tile_white
        else:
            self.randomize_tiles = False
            self.random_rotate_tile = False
            self.random_tile_white = False

        # assert that the number of tiles, tile size, and image size are compatible
        assert(np.allclose(
            self.img_size,
            np.sqrt(self.num_tiles) * self.tile_size
        ))

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

        # return only the image here- child classes can add metadata
        return image

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
        Each item in that n-length array is itself a 3-d numpy array 
        representing a tile.

        Note that if a path does not exist, we return an all-white pixel image
        '''
        tile_arr = []
        missing_file_errors = 0
        for p in paths:
            try:
                img = skimage.io.imread(p)
            except FileNotFoundError:
                img = 255*np.ones((self.tile_size, self.tile_size, 3)).astype(np.uint8)
                missing_file_errors += 1
            tile_arr.append(img)
        if missing_file_errors == self.num_tiles:
            raise Exception('Was missing all tiles for a given image. Check your tile paths.')
        return np.array(tile_arr)

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        raise NotImplementedError('This must be implemented in a child class.')


class TileBasedDataModule(LightningDataModule):

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
        self.image_meta_df = pd.read_csv(self.image_meta_path)

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

    def _get_transforms_for_phase(self, phase):
        key = f'{phase}_augmentations'
        if key in self.dataset_cfg:
            augmentations_cfg = self.dataset_cfg[key]
        else:
            augmentations_cfg = None
        transform_list = self._create_transforms(augmentations_cfg)
        transform_list.append(ToTensorV2())
        return alb.Compose(transform_list)

    def _set_transforms(self):
        '''
        Called by __init__ to set attributes for train/validation/test
        transformations/augmentations
        '''          
        self._train_transforms = self._get_transforms_for_phase('fit')
        self._validation_transforms = self._get_transforms_for_phase('validate')
        self._test_transforms = self._get_transforms_for_phase('test')

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
