# This script creates tiles into a pre-allocated
# directory tree. The location of the tiles is dictated
# by the metadata file (the image_subdir column). It then
# filters the metadata file to files corresponding
# to that subdirectory and creates the desired number of
# tiles there. This is used to permit easy parallelization
# of image tile creation.

# Note: due to the import of a sibling package (utils.image_utils)
# need to call with 
# python3 -m data_preparation.create_tiling_sharded <args>

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import skimage.io

from utils.image_utils import extract_tiles


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input_dir',
                        required=True,
                        type=Path,
                        help='The directory where the full-size images are stored.')
    parser.add_argument('-o', '--output_dir',
                        required=True,
                        type=Path,
                        help='The root directory where the output tiles are stored.')
    parser.add_argument('-i', '--train_metadata_file',
                        required=True,
                        type=Path,
                        help='The training metadata file.')
    parser.add_argument('-p', '--shard', type=int, required=True)
    parser.add_argument('-n', '--num_tiles', type=int, default=64)
    parser.add_argument('-s', '--tile_size', type=int, default=192)
    parser.add_argument('-l', '--level', type=int, default=0)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('-m', '--mode', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_tiles = args.num_tiles
    tile_size = args.tile_size
    level = args.level
    mode = args.mode
    shard = args.shard

    # has image_id, data_provider, isup_grade, gleason_score
    # PLUS image_subdir which gives the shard.
    train_metadata = pd.read_csv(args.train_metadata_file)

    # output directory name reflects how the images were made. Must exist
    output_dir = output_dir / f'numtile-{n_tiles}-tilesize-{tile_size}-res-{level}-mode-{mode}'
    assert(output_dir.exists())

    # filter for the images of interest:
    train_metadata = train_metadata.loc[train_metadata.shard == shard]

    for i, row in train_metadata.iterrows():
        image_id = row['image_id']
        image_subdir = row['image_subdir']
        img_path = input_dir / str(image_subdir) / f'{image_id}.tiff'

        # tiles is an array of shape (n_tiles, tile_size, tile_size, 3)
        tiles, _ = extract_tiles(img_path,
                                 num_tiles=n_tiles,
                                 tile_size=tile_size,
                                 mode=mode)
        for ii in range(n_tiles):
            t = tiles[ii, :].astype(np.uint8)
            fout = output_dir / str(image_subdir) / f'{image_id}_{ii}.png'
            skimage.io.imsave(fout, t)



if __name__ == '__main__':
    main()