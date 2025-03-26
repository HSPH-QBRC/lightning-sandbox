from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data_modules.tiles import TileBasedDataset, \
    TileBasedDataModule


class PremadeDLBCLPatchesDataset(TileBasedDataset):
    '''
    The TCIA DLBCL dataset includes a variable number of 224x224 PNGs
    for a subset of all subjects. The site claims 240x240
    https://www.cancerimagingarchive.net/collection/dlbcl-morphology/
    and says there are over 150k, but those likely include patches for 
    other stains instead of just H&E
    '''

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
            self.base_input_tile_dir
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
        In general there could be multiple folders which can contain different versions
        of the tiles extracted from the original images. This allows for that possibility
        without changing too much of the other methods
        '''
        # note that for this dataset, we have patient_id==image_id
        # since the pre-made patches are given at the patient level
        # (and for the SVS there were sometimes >1 SVS per patient)
        if self.phase in ['fit','validate']:
            root_dir = self.train_input_tile_dirs[0]
            return f'{root_dir}/{image_id}'
        else: # test/predict case
            raise NotImplementedError('!!!')

    def _get_tiles(self, image_id):
        '''
        Handles retrieving an array of tiles which were
        previously extracted from the original image.
        '''
        img_dir = self._get_input_tile_dir(image_id)
        if self.phase in ['fit', 'validate']:
            # each subject has a variable number of patches that were created
            all_patch_files = Path(img_dir).glob('*.png')

            if len(all_patch_files) < self.num_tiles:
                tiles = self._get_tiles_from_paths(all_patch_files)
                tiles = np.pad(tiles,
                        [
                            [0, self.num_tiles - len(tiles)],
                            [0, 0],
                            [0, 0],
                            [0, 0]
                        ], constant_values=255)
            else:
                selected_patches = np.random.choice(
                    np.arange(len(all_patch_files)),
                    self.num_tiles,
                    replace=False)
                selected_paths = [all_patch_files[i] for i in selected_patches]
                tiles = self._get_tiles_from_paths(selected_paths)
            return tiles      
        else: # test/predict
            raise NotImplementedError('For test/predict, have not implemeted dataset.')


class PremadeDLBCLPatchesDataModule(TileBasedDataModule):

    NAME = 'tcia_dlbcl_premade'

    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)

    def setup(self, stage):

        train_image_meta_df, test_image_meta_df = train_test_split(self.image_meta_df, 
                                                                   test_size=self.dataset_cfg.validation_fraction,
                                                                   stratify=self.image_meta_df['Stage'])
        
        if stage == 'fit':

            self.train_dataset = PremadeDLBCLPatchesDataset(self.dataset_cfg,
                                                stage,
                                                train_image_meta_df,
                                                self._train_transforms)
            self.val_dataset = PremadeDLBCLPatchesDataset(self.dataset_cfg,
                                                stage,
                                                test_image_meta_df,
                                                self._train_transforms)
        elif stage == 'validate':
            self.val_dataset = PremadeDLBCLPatchesDataset(self.dataset_cfg,
                                                    stage,
                                                    test_image_meta_df,
                                                    self._validation_transforms)

        elif stage == 'test' or stage == 'predict':
            raise NotImplementedError('Did not implement test/predict stage')
        else:
            # stage has values of only {fit, validate, test, predict}
            # but raise an exception here just so all the conditionals
            # are resolved
            raise Exception('Received invalid stage in DataModule.setup')