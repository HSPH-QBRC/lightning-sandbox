# This script is used to pre-allocated the root 
# directory and subsequent sub-directories in preparation
# for a sharded creation of input tile images.

import argparse
from logging import root
from pathlib import Path

import pandas as pd
import numpy as np

# the maximum number of files permitted in a single directory
N_FILES_MAX = 9999

# we plan to create a tree of directories. Each leaf 
# directory will contain the num_tiles number of images.
# N_DIRS controls (roughly) how many directories there are 
# at each level above that. For example, if we have 10,000
# full-size images, we can have a top level of 100 folders
# each of while will contain 100 folders. 
TARGET_N_DIRS = 100


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
    parser.add_argument('-k', '--num_shards', type=int, default=50)
    return parser.parse_args()


def construct_hierarchy(root_dir, n_per_level, total_levels, current_level, total_dirs):
    dirs = []
    if current_level < total_levels:
        for i in range(n_per_level):
            dirs.extend(
                construct_hierarchy(root_dir / str(i),
                                    n_per_level,
                                    total_levels,
                                    current_level + 1,
                                    total_dirs)
            )
    else:
        for i in range(n_per_level):
            d = root_dir / str(i)
            d.mkdir(parents=True)
            dirs.append(d)
    return dirs


def main():
    args = parse_args()
    output_dir = args.output_dir
    n_tiles = args.num_tiles
    tile_size = args.tile_size
    level = args.level
    mode = args.mode
    num_shards = args.num_shards

    # has image_id, data_provider, isup_grade, gleason_score
    train_metadata = pd.read_csv(args.train_metadata_file)

    # output directory name reflects how the images were made:
    output_dir = output_dir / f'numtile-{n_tiles}-tilesize-{tile_size}-res-{level}-mode-{mode}'
    output_dir.mkdir()

    # the number of full-sized training images:
    n_fullsize_train_imgs = train_metadata.shape[0]

    # to prevent us from crippling the filesystem with tons of files, 
    # we need to create subdirectories inside `output_dir`.
    # Each full-size image will generate `n_tiles` which we place in the same
    # directory. Above that, we create a hierarchy such that we have a 'reasonable'
    # number of folders at each level. As an example, if we had 50k full-size
    # images, then dir_levels would be 2.0 and n_dirs = 224. Thus, at the top-level
    # of the hierachy, we would have 224 folders. At the second level, we would place
    # 224 directories in each of the parents until we reach the desired 50k directories.
    dir_levels = np.round(np.log(n_fullsize_train_imgs)/np.log(TARGET_N_DIRS))
    dirs_per_level = np.ceil(np.power(n_fullsize_train_imgs, 1/dir_levels))

    # pre-create the subdirs- not that this will create extra empty dirs if the
    # number of full-size images is not equal to dirs_per_level^dir_levels. Hence the truncation
    image_subdirs = construct_hierarchy(output_dir, dirs_per_level, dir_levels, 1, n_fullsize_train_imgs)[:n_fullsize_train_imgs]

    # Specify which images will go to which subdirectories.
    train_metadata['image_subdir'] = image_subdirs

    # to aid with sharding, we assign the data to one of the K shards:
    num_per_shard = (n_fullsize_train_imgs // num_shards) + 1
    shard_assignments = np.repeat(np.arange(num_shards), num_per_shard)[:n_fullsize_train_imgs]
    train_metadata['shard'] = shard_assignments

    # write this file so we can shard the tile creation process.
    train_metadata.to_csv(f'{args.train_metadata_file.parent}/training_metadata_with_shards_and_subdirs.csv', index=False)


if __name__ == '__main__':
    main()
