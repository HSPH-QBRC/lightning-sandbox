from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data_modules.tiles import TileBasedDataset, \
    TileBasedDataModule


class DLBCLDataset(TileBasedDataset):

    def __init__(self, dataset_cfg, phase, image_meta_df, transform):
        super().__init__(dataset_cfg, phase, image_meta_df, transform)

        # since the DLBCL images are especially large, we extract more tiles 
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
            row['Stage'], 
            row['IPI_Score'], 
            row['IPI_Risk_Group_4_Class'], 
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
            raise NotImplementedError('!!!')

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        img_dir = self._get_input_tile_dir(image_id)
        if self.phase in ['fit', 'validate']:
            # we have self.extracted_tile_count number of potential tiles
            # to use when constructing the composite image. 
            paths = [
                Path(f'{img_dir}/{image_id}.tile_{i}.png')
                for i in np.random.choice(
                    np.arange(self.extracted_tile_count), 
                    self.num_tiles, 
                    replace=False)
            ]
            return self._get_tiles_from_paths(paths)
        else: # test/predict
            raise NotImplementedError('For test/predict, have not implemeted dataset.')


class DLBCLDataModule(TileBasedDataModule):

    NAME = 'tcia_dlbcl'

    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)

    def setup(self, stage):

        # only do the train/val split if training or validating. If we're testing, we don't
        # bother with a split and just use the whole `self.image_meta_df` dataframe
        if stage in ['fit', 'validate']:
            train_image_meta_df, test_image_meta_df = train_test_split(self.image_meta_df, 
                                                                    test_size=self.dataset_cfg.validation_fraction,
                                                                    stratify=self.image_meta_df['Stage'])
            train_image_meta_df.to_csv('train_set.csv', index=False)
            test_image_meta_df.to_csv('test_set.csv', index=False)
            
        if stage == 'fit':

            self.train_dataset = DLBCLDataset(self.dataset_cfg,
                                                stage,
                                                train_image_meta_df,
                                                self._train_transforms)
            self.val_dataset = DLBCLDataset(self.dataset_cfg,
                                                stage,
                                                test_image_meta_df,
                                                self._train_transforms)
        elif stage == 'validate':
            self.val_dataset = DLBCLDataset(self.dataset_cfg,
                                                    stage,
                                                    test_image_meta_df,
                                                    self._validation_transforms)

        elif stage == 'test' or stage == 'predict':
            self.test_dataset = DLBCLDataset(self.dataset_cfg,
                                        stage,
                                        self.image_meta_df,
                                        self._test_transforms)
        else:
            # stage has values of only {fit, validate, test, predict}
            # but raise an exception here just so all the conditionals
            # are resolved
            raise Exception('Received invalid stage in DataModule.setup')