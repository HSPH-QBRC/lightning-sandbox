# This script is used to pre-allocated the root 
# directory and subsequent sub-directories in preparation
# for a sharded creation of input tile images.

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# the maximum number of files permitted in a single directory
N_FILES_MAX = 9999


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--output_dir',
                        required=True,
                        type=Path,
                        help='The directory where the tiles will be are stored.')
    parser.add_argument('-i', '--train_metadata_file',
                        required=True,
                        type=Path,
                        help='The training metadata file.')        
    parser.add_argument('-n', '--num_tiles', type=int, default=64)
    parser.add_argument('-s', '--tile_size', type=int, default=192)
    parser.add_argument('-l', '--level', type=int, default=0)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('-m', '--mode', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    n_tiles = args.num_tiles
    tile_size = args.tile_size
    level = args.level
    mode = args.mode

    # has image_id, data_provider, isup_grade, gleason_score
    train_metadata = pd.read_csv(args.train_metadata_file)

    # output directory name reflects how the images were made:
    output_dir = output_dir / f'numtile-{n_tiles}-tilesize-{tile_size}-res-{level}-mode-{mode}'
    output_dir.mkdir()

    # the number of full-sized training images:
    n_fullsize_train_imgs = train_metadata.shape[0]

    # to prevent us from crippling the filesystem with tons of files, 
    # we need to create subdirectories inside `output_dir`. Given that
    # each full-size image will generate `n_tiles`, create numbered subdirs
    # accordingly.
    
    # we want all the tiles corresponding to a single full-size image
    # to be in the same subdir. This dictates how many full-size images
    # we can assign to a particular subdir based on the max number of files
    # which can be stored in any one subdir:
    fullsize_img_per_subdir = N_FILES_MAX // n_tiles

    num_subdirs = 1 + (n_fullsize_train_imgs // fullsize_img_per_subdir)

    # pre-create the subdirs
    for i in range(num_subdirs):
        (output_dir / str(i)).mkdir()

    # to help with the sharding operation, we will specify which images will go to 
    # which subdirectories.
    image_subdirs = np.repeat(np.arange(num_subdirs), fullsize_img_per_subdir)[:n_fullsize_train_imgs]
    train_metadata['image_subdir'] = image_subdirs

    # write this file so we can shard the tile creation process.
    train_metadata.to_csv(f'{args.train_metadata_file.parent}/training_metadata_with_shards.csv', index=False)


if __name__ == '__main__':
    main()
