from pathlib import Path

import numpy as np

from data_modules.tiles import TileBasedDataset, \
    TileBasedDataModule
from utils.image_utils import DensityBasedTileExtractor, \
    TileInfo

class PandasDataset(TileBasedDataset):

    def __init__(self, dataset_cfg, phase, image_meta_df, transform):
        super().__init__(dataset_cfg, phase, image_meta_df, transform)
        self._set_tile_source_dirs()

    def _set_tile_source_dirs(self):

        # we have two directories where we have extracted tiles. They differ
        # based on the initial tile positioning. This is the "offset mode"
        # Within those directories are subdirectories (which keeps the number
        # of files per directory manageable). These paths, do NOT have those-
        # they are just the 'root'
        self.train_input_tile_dirs = [
            f'{self.base_input_tile_dir}/numtile-{self.num_tiles}-tilesize-{self.tile_size}-res-{self.img_resolution}-mode-{m}'
            for m in [0, 2]
        ]

    def __getitem__(self, idx):

        image = super().__getitem__(idx)
        row = self.image_meta_df.iloc[idx]

        # return the image PLUS some metadata- the model
        # will handle how this additional metadata is encoded
        # as the target, etc.
        return image, (
            row['isup_grade'], 
            row['gleason_score'], 
            row['data_provider'], 
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
            # note: can change for a true test case. However, we can
            # use this to perform prediction on the hold-out set 
            # if we are running inference instead of validation during training
            return f'{self.train_input_tile_dirs[0]}/{subdir}'

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        img_dir = self._get_input_tile_dir(image_id)
        paths = [
            Path(f'{img_dir}/{image_id}.tile_{i}.png')
            for i in range(self.num_tiles)
        ]
        return self._get_tiles_from_paths(paths)


class PandasDataModule(TileBasedDataModule):

    NAME = 'pandas_challenge'

    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)

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