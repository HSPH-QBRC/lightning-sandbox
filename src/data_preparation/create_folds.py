# This script is used to pre-compute train/validation folds
# for the pandas challenge dataset.

import argparse
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--train_metadata_file',
                        required=True,
                        type=Path,
                        help='The training metadata file.')        
    parser.add_argument('-k', '--kfold',
                        type=int,
                        default=5,
                        help='The number of folds to create.')
    parser.add_argument('-d', '--duplicated_images_file',
                        required=True,
                        type=Path,
                        help='File with alleged duplicated image IDs.')
    parser.add_argument('-s', '--seed',
                        type=int,
                        default=1222)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.train_metadata_file)
    original_df = df.copy() # for later...

    # Using the file which purports to identify potential duplicates (or highly similar
    # images), go through and remove the duplicates. Note that this is temporary and
    # we will later "re-incoporate" those duplicates/similar images.
    duplicated_image_df = pd.read_csv(args.duplicated_images_file)

    # the file groups the images (`group_id`). Within each group, there is a 
    # 'index_in_group'. Below, we keep just the first image from each group of
    # similar/duplicated images.
    img_ids_excluded = set(duplicated_image_df[duplicated_image_df.index_in_group != 0].image_id)
    df = df[~df.image_id.isin(img_ids_excluded)].reset_index(drop=True)

    # create dummy vectors for the gleason score and provider. These will be
    # used to ultimately split the folds such that representation of each
    # score + provider is evenly distributed across the folds.
    label_unique = set(df.gleason_score.unique())
    label2id = dict(zip(label_unique, range(len(label_unique))))
    y_gleason = df.gleason_score.apply(lambda x: label2id[x]).values

    provider_unique = set(df.data_provider.unique())
    provider2id = dict(zip(provider_unique, range(len(provider_unique))))
    y_provider = df.data_provider.apply(lambda x: provider2id[x]).values

    y = np.stack([y_gleason, y_provider], axis=1)

    df["kfold"] = -1
    indxs = list(range(df.shape[0]))

    mskf = MultilabelStratifiedKFold(
        n_splits=args.kfold,
        random_state=args.seed,
        shuffle=True
    )

    for i, (train_index, test_index) in enumerate(mskf.split(indxs, y)):
        df.loc[test_index, "kfold"] = i + 1

    # Thus, to this point we have assigned all the "non-duplicated images" to
    # a fold. Rather than ignoring the similar/duplicated images, we can
    # go through and assign all those alleged duplicates to the same fold.
    # This prevents potential leakage into the validation set.
    # Here, following the merge, the similar images will have NaN values
    # which we will fill with -1 as markers.
    df2 = pd.merge(
        original_df,
        df[["image_id", "kfold"]],
        how="left",
        on="image_id")
    df2.kfold = df2.kfold.fillna(-1)

    new_kfolds = list()
    for row in df2.itertuples():
        kfold_old = int(row.kfold)
        img_id = row.image_id

        # this means it was a duplicate image
        if kfold_old == -1:
            group_id = (
                duplicated_image_df.loc[duplicated_image_df.image_id == img_id]
                .group_id.values[0]
            )

            # given the 'group' that the duplicated/similar image was assigned
            # to, find which fold the 'first' item of that group was assigned
            # to. Then we will assign this duplicate image to the same fold
            img_id_top = duplicated_image_df[
                (duplicated_image_df.group_id == group_id)
                &
                (duplicated_image_df.index_in_group == 0)
            ].image_id.values[0]
            kfold_new = df2[df2.image_id == img_id_top].kfold.values[0]
            new_kfolds.append(kfold_new)
        # since fold was != -1, then it does not need to be re-assigned
        else:
            new_kfolds.append(kfold_old)
    
    df2['kfold'] = new_kfolds
    df2['kfold'] = df2['kfold'].astype('int')
    new_path = args.train_metadata_file.parent / f'training_metadata_with_shards.{args.kfold}kfold.csv'
    df2.to_csv(new_path, index=False)


if __name__ == '__main__':
    main()