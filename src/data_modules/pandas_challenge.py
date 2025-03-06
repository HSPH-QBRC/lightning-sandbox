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

from utils.image_utils import DensityBasedTileExtractor, \
    TileInfo


class PandasDataset(Dataset):

    def __init__(self, dataset_cfg, phase, image_meta_df, transform):
        super().__init__()

        self.phase = phase
        self.transform = transform

        # how many tiles are in the 'composite' image
        self.num_tiles = dataset_cfg.num_tiles

        # number of pixels in the individual tiles
        self.tile_size = dataset_cfg.tile_size

        # size of the composite image
        self.img_size = dataset_cfg.img_size

        # resolution level of the tiled images (which level of
        # the TIFF they were extracted from)
        self.img_resolution = dataset_cfg.resolution

        # Dataframe giving the image ID, source, Gleason scores, etc.
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

        # we have two directories where we have extracted tiles. They differ
        # based on the initial tile positioning. This is the "offset mode"
        # Within those directories are subdirectories (which keeps the number
        # of files per directory manageable). These paths, do NOT have those-
        # they are just the 'root'
        self.base_input_tile_dir = dataset_cfg.base_dir
        self.train_input_tile_dirs = [
            f'{self.base_input_tile_dir}/numtile-{self.num_tiles}-tilesize-{self.tile_size}-res-{self.img_resolution}-mode-{m}'
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
        return image, (isup_grade, gleason_score, data_provider, row['image_id'])

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
        Each item in that n-length array is itself a 2-d numpy array 
        representing a tile
        '''
        return np.array([skimage.io.imread(p) for p in paths])

    def _get_input_tile_dir(self, image_id):
        '''
        There are multiple folders which can contain different versions
        of the tiles extracted from the original images. This method
        contains the logic for that.
        '''
        # to avoid issues with filesystems and excessive numbers of files
        # we locate the previously prepared tile images inside of
        # subdirectories. By using the image ID, we can determine
        # which subdirectory we need
        row = self.image_meta_df.loc[
            self.image_meta_df['image_id'] == image_id].iloc[0]
        subdir = row.image_subdir
        if self.phase == 'fit':
            if np.random.rand() < 0.5:
                root_dir = self.train_input_tile_dirs[0]
            else:
                root_dir = self.train_input_tile_dirs[1]
            return f'{root_dir}/{subdir}'
        elif self.phase == 'validate':
            return f'{self.train_input_tile_dirs[0]}/{subdir}'
        else: # test/predict case
            raise NotImplementedError('!!!')

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        img_dir = self._get_input_tile_dir(image_id)
        if self.phase in ['fit', 'validate']:
            paths = [
                Path(f'{img_dir}/{image_id}_{i}.png')
                for i in range(self.num_tiles)
            ]
            return self._get_tiles_from_paths(paths)
        else: # test/predict
            img_path = Path(f'{img_dir}/{image_id}.tiff')
            tile_info = TileInfo(self.num_tiles, self.tile_size, self.img_resolution, 0)
            extractor = DensityBasedTileExtractor(tile_info)
            tiles = extractor.extract(img_path)       
            return tiles


class PandasDataModule(LightningDataModule):

    NAME = 'pandas_challenge'

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

    def setup(self, stage):
        
        if stage == 'fit':

            # Note that based on the parameters passed into
            # the PandasDataset constructor (via self.dataset_cfg)
            # the train and validation parts are "pre-determined"
            # by the info in the training metadata file. Therefore,
            # we don't have to use `torch.utils.data.random_split`
            # or anything similar

            if self.dataset_cfg.kfold is not None:
                holdout = self.image_meta_df.kfold == self.dataset_cfg.kfold
                train_image_meta_df = self.image_meta_df.loc[~holdout]
                val_image_meta_df = self.image_meta_df.loc[holdout]
                self.train_dataset = PandasDataset(self.dataset_cfg,
                                                   stage,
                                                   train_image_meta_df,
                                                   self._train_transforms)
                self.val_dataset = PandasDataset(self.dataset_cfg,
                                                   stage,
                                                   val_image_meta_df,
                                                   self._train_transforms)
            else:
                raise Exception('Need to specify a fold at this time.')


        elif stage == 'validate':
            holdout = self.image_meta_df.kfold == self.dataset_cfg.kfold
            val_image_meta_df = self.image_meta_df.loc[holdout]
            self.val_dataset = PandasDataset(self.dataset_cfg,
                                                    stage,
                                                    val_image_meta_df,
                                                    self._validation_transforms)

        elif stage == 'test' or stage == 'predict':
            self.test_dataset = PandasDataset(self.dataset_cfg,
                                              stage,
                                              self.image_meta_df,
                                              self._test_transforms)
        else:
            # stage has values of only {fit, validate, test, predict}
            # but raise an exception here just so all the conditionals
            # are resolved
            raise Exception('Received invalid stage in DataModule.setup')

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
