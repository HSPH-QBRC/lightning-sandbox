# Due to the sheer number of files, we can't just unpack
# the zip archive without sending files to subdirectories.
# This script handles that logic.

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--zipfile',
                        required=True,
                        type=Path,
                        help='The zipfile path.')
    parser.add_argument('-i', '--train_metadata_file',
                        required=True,
                        type=Path,
                        help='The training metadata file.') 
    parser.add_argument('-o', '--output_dir',
                        required=True,
                        type=Path,
                        help='The output directory root.') 
    return parser.parse_args()


def main():
    args = parse_args()

    # has image_id, data_provider, isup_grade, gleason_score
    # PLUS image_subdir which gives the shard.
    train_metadata = pd.read_csv(args.train_metadata_file)

    # subdirs are zero-indexed, so add 1
    num_subdirs = 1 + train_metadata.image_subdir.max()

    # pre-create the subdirs
    for i in range(num_subdirs):
        (args.output_dir / str(i)).mkdir()
        
    with zipfile.ZipFile(args.zipfile) as zf:
        for i, row in train_metadata.iterrows():
            image_id = row['image_id']
            subdir = row['image_subdir']
            extract_target = args.output_dir / str(subdir) / f'{image_id}.tiff'
            f = f'train_images/{image_id}.tiff'
            with open(extract_target, 'wb') as dest:
                with zf.open(f, 'r') as src:
                    shutil.copyfileobj(src, dest)


if __name__ == '__main__':
    main()