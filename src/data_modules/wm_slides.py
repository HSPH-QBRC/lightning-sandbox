from pathlib import Path

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
import skimage.io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, \
    Dataset

from data_modules.tiles import TileBasedDataset, \
    TileBasedDataModule


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
        self.input_tiles_dir = f'{tiles_root_dir}/numtile-{self.num_tiles}-tilesize-{self.tile_size}-res-{self.img_resolution}-mode-{tile_mode}'            

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
            Path(f'{self.input_tiles_dir}/{image_id}_{i}.png')
            for i in range(self.num_tiles)
        ]
        return self._get_tiles_from_paths(paths)


class WmSlideInferenceDataModule(LightningDataModule):

    NAME = 'wm_slides_inference'

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


class WMSlideDataset(TileBasedDataset):

    def __init__(self, dataset_cfg, phase, image_meta_df, transform):
        super().__init__(dataset_cfg, phase, image_meta_df, transform)

        # since the WM slides images are especially large, we extract more tiles 
        # than just that needed to create our composite image here. For instance,
        # our composite image might consist of 64 tiles arranged in an (8,8) grid.
        # However, we can certainly make more than just 64. This specifies that number
        self.extracted_tile_count = dataset_cfg.extracted_tile_count
        self._set_tile_source_dirs()

    def _set_tile_source_dirs(self):

        # we have two directories where we have extracted tiles. They differ
        # based on the initial tile positioning. This is the "offset mode"
        # Within those directories are subdirectories (which keeps the number
        # of files per directory manageable). These paths, do NOT have those-
        # they are just the 'root'
        self.train_input_tile_dirs = [
            f'{self.base_input_tile_dir}/numtile-{self.extracted_tile_count}-tilesize-{self.tile_size}-res-{self.img_resolution}-mode-{m}'
            for m in [0, 2]
        ]

    def __getitem__(self, idx):

        image = super().__getitem__(idx)
        row = self.image_meta_df.iloc[idx]

        # return the image PLUS some metadata- the model
        # will handle how this additional metadata is used
        return image, (
            row['Subtype'], 
            row['image_id']
        )

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
            # can change if proper test set is created. This just allows
            # us to find images in the given folder- does NOT mean we are
            # testing using the training set
            return f'{self.train_input_tile_dirs[0]}/{subdir}'

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        img_dir = self._get_input_tile_dir(image_id)

        paths = [
            Path(f'{img_dir}/{image_id}.tile_{i}.png')
            for i in np.random.choice(
                np.arange(self.extracted_tile_count), 
                self.num_tiles, 
                replace=False)
        ]
        return self._get_tiles_from_paths(paths)


class WMSlideDatasetBinaryBCL(WMSlideDataset):
    '''
    Adaptation for binary predictions on BC-like or not
    '''

    def __init__(self, dataset_cfg, phase, image_meta_df, transform):
        super().__init__(dataset_cfg, phase, image_meta_df, transform)

    def __getitem__(self, idx):

        # since we are calling the method in the parent to get the
        # image itself, it ALSO returns metadata, which we don't care about
        # Here, we will be altering the metadata returned, so we discard
        # the metadata from the parent.
        image, _ = super().__getitem__(idx)
        row = self.image_meta_df.iloc[idx]

        # return the image PLUS some metadata- the model
        # will handle how this additional metadata is used
        return image, (
            row['Subtype'], 
            row['image_id'],
            row['binary_bcl']
        )


class WMDataModule(TileBasedDataModule):

    NAME = 'wm_slides'

    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)

    def setup(self, stage):

        # only do the train/val split if training or validating. If we're testing, we don't
        # bother with a split and just use the whole `self.image_meta_df` dataframe
        if stage in ['fit', 'validate']:
            train_image_meta_df, test_image_meta_df = train_test_split(self.image_meta_df, 
                                                                    test_size=self.dataset_cfg.validation_fraction,
                                                                    stratify=self.image_meta_df['Subtype'])
            train_image_meta_df.to_csv('train_set.csv', index=False)
            test_image_meta_df.to_csv('test_set.csv', index=False)
            
        if stage == 'fit':

            self.train_dataset = WMSlideDataset(self.dataset_cfg,
                                                stage,
                                                train_image_meta_df,
                                                self._train_transforms)
            self.val_dataset = WMSlideDataset(self.dataset_cfg,
                                                stage,
                                                test_image_meta_df,
                                                self._train_transforms)
        elif stage == 'validate':
            self.val_dataset = WMSlideDataset(self.dataset_cfg,
                                                    stage,
                                                    test_image_meta_df,
                                                    self._validation_transforms)

        elif stage == 'test' or stage == 'predict':
            self.test_dataset = WMSlideDataset(self.dataset_cfg,
                                        stage,
                                        self.image_meta_df,
                                        self._test_transforms)
        else:
            # stage has values of only {fit, validate, test, predict}
            # but raise an exception here just so all the conditionals
            # are resolved
            raise Exception('Received invalid stage in DataModule.setup')


class WMBinaryBCLDataModule(TileBasedDataModule):

    NAME = 'wm_slides_binary_bcl'

    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)

    def setup(self, stage):

        # only do the train/val split if training or validating. If we're testing, we don't
        # bother with a split and just use the whole `self.image_meta_df` dataframe
        if stage in ['fit', 'validate']:
            train_image_meta_df, test_image_meta_df = train_test_split(self.image_meta_df, 
                                                                    test_size=self.dataset_cfg.validation_fraction,
                                                                    stratify=self.image_meta_df['binary_bcl'])
            train_image_meta_df.to_csv('train_set.csv', index=False)
            test_image_meta_df.to_csv('test_set.csv', index=False)
            
        if stage == 'fit':

            self.train_dataset = WMSlideDatasetBinaryBCL(self.dataset_cfg,
                                                stage,
                                                train_image_meta_df,
                                                self._train_transforms)
            self.val_dataset = WMSlideDatasetBinaryBCL(self.dataset_cfg,
                                                stage,
                                                test_image_meta_df,
                                                self._train_transforms)
        elif stage == 'validate':
            self.val_dataset = WMSlideDatasetBinaryBCL(self.dataset_cfg,
                                                    stage,
                                                    test_image_meta_df,
                                                    self._validation_transforms)

        elif stage == 'test' or stage == 'predict':
            self.test_dataset = WMSlideDatasetBinaryBCL(self.dataset_cfg,
                                        stage,
                                        self.image_meta_df,
                                        self._test_transforms)
        else:
            # stage has values of only {fit, validate, test, predict}
            # but raise an exception here just so all the conditionals
            # are resolved
            raise Exception('Received invalid stage in DataModule.setup')