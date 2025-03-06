# This script creates tiles into a pre-allocated
# directory tree. The location of the tiles is dictated
# by the metadata file which contains the following named columns:
# - image_id
#     - contains the unique identifier, no suffix/format
# - (optional) image_subdir
#     - if the number of images is large enough that we choose
#       to save images to subdirectories (so as not to overwhelm
#       the filesystem), then this will tell the script where they 
#       are located relative to the `-d/--input_dir` argument. Also
#       dictates where they are placed in the output directory.
# - (optional) shard
#     - if we have many large slide files and wish to perform a dumb
#       parallelization to save time, you can add this column to the metadata
#       file which then enables calling this script with the `-s/--shard`
#       argument. It could be the case that the shard is the same as the 
#       image_subdir, but that's not required.  
#
# If the input image is located in a particular subdir (of the input folder)
# then the output image will also be located in a subdir of the same name
# in the output folder.
#
# Note: due to the import of a sibling package (utils.image_utils)
# need to call with 
# python3 -m data_preparation.create_tiling_sharded <args>

import argparse
from pathlib import Path

from utils.image_utils import TileInfo, \
    get_extractor_class


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
    parser.add_argument('-x', '--extraction_key',
                        required=True,
                        type=int,
                        help=('The tile extraction class to use. This is an integer'
                              ' which selects the proper implementation class.'))
    parser.add_argument('-s', '--shard', type=int, required=False, default=None)
    parser.add_argument('-n', '--num_tiles', type=int, default=64)
    parser.add_argument('-t', '--tile_size', type=int, default=192)
    parser.add_argument('-l', '--level', type=int, default=0)
    parser.add_argument('-m', '--offset_mode', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir 
    output_dir = args.output_dir
    n_tiles = args.num_tiles
    tile_size = args.tile_size
    level = args.level
    offset_mode = args.offset_mode
    shard = args.shard
    image_meta_path = args.train_metadata_file
    tile_extraction_class_key = args.extraction_key

    tile_info = TileInfo(n_tiles, 
                         tile_size, 
                         level, 
                         offset_mode)
    
    # will raise an exception if not a valid key
    extractor_class = get_extractor_class(tile_extraction_class_key)

    # run the tile extraction
    extractor = extractor_class(tile_info)
    extractor.extract_and_save(image_meta_path, input_dir, output_dir, shard=shard)


if __name__ == '__main__':
    main()