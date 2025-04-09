from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import openslide
import pandas as pd
import skimage
from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray
from skimage.transform import resize


def get_extractor_class(key):
    match key:
        case 0:
            return DensityBasedTileExtractor
        case 1:
            return NormalizingHandETileExtractor
        case 2:
            return PiecewiseTileExtractor
        case 3:
            return DensityBasedPiecewiseTileExtractor
        case _:
            raise Exception('Did not specify an appropriate key'
                            ' for choosing the tile extractor.')
    

@dataclass
class TileInfo():
    n_tiles: int
    tile_size: int
    level: int
    offset_mode: int


class BaseTileExtractor(object):
    '''
    Base class which handles extraction of "informative" tiles
    from large SVS/TIFF images
    '''

    def __init__(self, tile_info):
        self.tile_info = tile_info

    def _create_output_dir(self, output_dir):

        # the name of the directory will give info about how the tiles were created
        tile_dir = (f'numtile-{self.tile_info.n_tiles}-tilesize-{self.tile_info.tile_size}'
                    f'-res-{self.tile_info.level}-mode-{self.tile_info.offset_mode}')
        output_dir = output_dir / tile_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _check_format(self, img_path):
        multilayer_formats = ['svs', 'tiff']
        if not img_path.suffix.lower()[1:] in multilayer_formats:
            raise Exception(f'Unexpected format for path: {img_path}')

    def _load_image(self, img_path):
        return self._read_multilayer(img_path)

    def _open_multilayer(self, img_path):
        self._check_format(img_path)
        return openslide.OpenSlide(img_path)

    def _read_patch(self, o, level, start_tuple, stop_tuple):
        return np.array(o.read_region(
            start_tuple,
            level,
            stop_tuple
        ))[:,:,:3]  # in case there is an alpha channel, we take only the RGB channels      

    def _read_multilayer(self, img_path):
        '''
        Opens a multilayer format file (e.g. SVS, TIFF)
        and returns a numpy array of the pixels at the
        prescribed level. Reads entire slide
        '''
        level = self.tile_info.level

        o = self._open_multilayer(img_path)

        # a tuple of tuples, like ((19991, 42055), (4997, 10513), (1249, 2628))
        level_dimensions = o.level_dimensions

        if level >= o.level_count:
            raise Exception(f'Requested a resolution {level} that does not'
                            f' exist for the image at {img_path}, which has'
                            f' {o.level_count} levels')

        return self._read_patch(o, level, (0, 0), level_dimensions[level])
    
    def _calculate_padding(self, w, h):
        '''
        Given (w,h) calculate the padding to add such that
        the total image will permit an even number of tiles in 
        both horizontal and vertical directions.
        '''
        # for brevity:
        tile_size = self.tile_info.tile_size
        offset_mode = self.tile_info.offset_mode
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * offset_mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * offset_mode) // 2)
        return pad_w, pad_h

    def _apply_white_padding(self, img, pad_w, pad_h):
        return np.pad(img,
                [
                    [pad_h // 2, pad_h - pad_h // 2],
                    [pad_w // 2,pad_w - pad_w//2],
                    [0,0]
                ],
                constant_values=255)
    
    def _pad_image(self, img):
        '''
        Pad the numpy array such that the eventual
        tiling operations will create an integer number
        of tiles
        '''
        h, w, c = img.shape
        pad_w, pad_h = self._calculate_padding(w, h)
        return self._apply_white_padding(img, pad_w, pad_h)
    
    def _get_tiles_from_array(self, img, tile_index_filter=None):
        '''
        Given an array (i.e. an image), and 
        '''
        tile_size = self.tile_info.tile_size

        if tile_index_filter is not None:
            img_stack = []
            n_w = img.shape[1] // tile_size
            for idx in tile_index_filter:
                tile_row = idx // n_w
                tile_col = idx % n_w
                y0 = tile_row * tile_size
                x0 = tile_col * tile_size
                img_stack.append(img[y0:y0+tile_size,x0:x0+tile_size])
            img = np.stack(img_stack)
            return img
        else:
            # note that the reshape method first performs a 'ravel' 
            # (e.g. a flattening) before it chunks everything for the reshape.
            # Based on this, we get this somewhat awkward 5-tensor. This is
            # due to the way that the raveling and reshaping traverses the
            # original image
            img = img.reshape(img.shape[0] // tile_size,
                            tile_size,
                            img.shape[1] // tile_size,
                            tile_size,
                            3).astype(np.float32)

            # to ultimately create an array of (tile_size, tile_size, 3) tiles, we need to 
            # do this transpose and reshape. This gives array of shape 
            # (N, tile_size, tile_size, 3)
            # where N is the number of tiles (i.e. padded size // tile_size)
            return img.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size, 3)
        
    def _get_raw_tiles(self, img_path, tile_index_filter=None):
        '''
        This returns an array of shape (N, tile_size, tile_size, 3)
        where N is the number of tiles

        If `tile_index_filter` is not None, then specific tiles are
        selected based on their grid position. For example, if we are
        looking to tile an image and our tile size is such that we can
        fit (m,n) tiles in the vertical and horizontal directions, then
        an index filter of `k` will select/keep the tile at (k//n, k%n).
        This allows us to avoid tiling in regions of largely whitespace, etc.
        '''
        img = self._load_image(img_path)
        img = self._pad_image(img)
        return self._get_tiles_from_array(img, tile_index_filter=tile_index_filter)

    def _select_tiles(self, img_path, img_id):
        raise NotImplementedError('Need to implement a _select_tiles method'
                                  ' specific to your application.')
    
    def extract(self, img_path, img_id):
        return self._select_tiles(img_path, img_id)

    def _get_image_path(self, image_meta, input_root_dir):
        '''
        Given a pd.Series from the image metadata
        dataframe, find the image and return a path

        `input_root_dir` is a pathlib.Path which locates
        the top directory of the input directory tree
        '''
        image_id = image_meta['image_id']

        if 'image_subdir' in image_meta:
            input_image_dir = input_root_dir / str(image_meta['image_subdir'])
        else:
            input_image_dir = input_root_dir

        img_path = list(input_image_dir.glob(f'{image_id}.*'))
        if len(img_path) != 1:
            raise Exception('Could not locate a single file matching'
                            f' the pattern {image_id}.* in'
                            f' {input_image_dir}.')
        else:
            return img_path[0]
        
    def _get_final_output_dir(self, image_meta, output_root_dir):
        '''
        Given a pd.Series (`image_meta`) from the image metadata
        dataframe, return a path to the output folder

        `output_root_dir` is a pathlib.Path to the root of the output
        directory, which may contain subdirectories
        '''
        if 'image_subdir' in image_meta:
            output_dir = output_root_dir / str(image_meta['image_subdir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        else:
            return output_root_dir

    def extract_and_save(self, image_metadata_path, input_root_dir, output_root_dir, shard=None):
        '''
        Orchestrates the reading, tile extraction, and saving of tiles for
        a full dataset
        '''
        metadata_df = pd.read_csv(image_metadata_path)

        if shard is not None:
            self.sharded = True
            self.shard_num = shard
            metadata_df = metadata_df.loc[metadata_df.shard == self.shard_num]
        else:
            self.sharded = False

        output_root_dir = self._create_output_dir(output_root_dir)

        for i, row in metadata_df.iterrows():
            img_path = self._get_image_path(row, input_root_dir)
            output_dir = self._get_final_output_dir(row, output_root_dir)
            tiles = self._select_tiles(img_path, row['image_id'])

            for ii in range(tiles.shape[0]):
                t = tiles[ii, :].astype(np.uint8)
                fout = output_dir / f'{row['image_id']}.tile_{ii}.png'
                skimage.io.imsave(fout, t)


class DensityBasedTileMixin:
    '''
    Mixin class which helps with sorting and filtering tiles based on density.

    Will remove tiles that are low variance which likely represent black space
    near the edges of a slide image. Follows this by sorting tiles based on
    pixel density (e.g. those with the darkest overall colors)
    '''

    # used to remove dark tiles with very little variance
    # due to edges, etc. that are all black. Will also remove
    # largely white tiles which are uninformative, although
    # those can also be removed by sorting on pixel sums
    VARIANCE_THRESHOLD = 30

    def _post_process_tiles(self, tile_array):

        tile_vars = tile_array.var(axis=3) # (N, tile_size, tile_size)
        tile_mean_vars = tile_vars.mean(axis=(1,2))
        low_variance = tile_mean_vars < DensityBasedTileMixin.VARIANCE_THRESHOLD
        tile_array = tile_array[~low_variance]

        # based on the sum of the pixels in each tile, we sort and take the
        # top num_tiles. Recall that np.argsort gives ascending order. We
        # want this since more 'informative' images will have more colored pixels
        # which are lower pixel values.
        # Majority white tiles will have very large sums and are likely not
        # very informative.
        tile_sums = tile_array.reshape(tile_array.shape[0],-1).sum(-1)
        idxs = np.argsort(tile_sums)
        sorted_tile_sums = tile_sums[idxs]
        return tile_array[idxs], sorted_tile_sums
    

class DensityBasedTileExtractor(BaseTileExtractor, DensityBasedTileMixin):
    '''
    Returns tiles based on the pixel colors (e.g. darker is better)
    while attempting to remove low-variance dark regions likely to be
    slide artifacts
    '''

    def _select_tiles(self, img_path, img_id):
        '''
        This method for tile selection involves summing the RGB
        pixels for each tile and selecting those with the lowest values
        (since lower pixel values means darker images)
        '''
        num_tiles = self.tile_info.n_tiles

        tiles = self._get_raw_tiles(img_path)

        # remove low-variance tiles and sort such that
        # the tiles with the darkest pixels are at the 
        # top of the list:
        tiles, tile_pixel_sums = self._post_process_tiles(tiles)

        # if the total number of tiles in the full image was fewer than the number of requested
        # tiles, add the required number of fully white tiles
        if len(tiles) < num_tiles:
            tiles = np.pad(tiles,
                        [
                            [0, num_tiles - len(tiles)],
                            [0, 0],
                            [0, 0],
                            [0, 0]
                        ], constant_values=255)
        return tiles[:num_tiles]


class NormalizingHandETileExtractor(BaseTileExtractor):
    '''
    Implements a normalization scheme based on Macenko (2009):
    https://ieeexplore.ieee.org/document/5193250

    Original code in matlab:
    https://github.com/mitkovetta/staining-normalization/blob/master/normalizeStaining.m    
    '''
    IO = 240 # Transmitted light intensity, Normalizing factor for image intensities
    ALPHA = 1  #As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)
    BETA = 0.15 #As recommended in the paper. OD threshold for transparent pixels (default: 0.15)

    # Reference H&E OD matrix.
    # Can be updated if you know the best values for your image. 
    # Otherwise use the following default values. 
    H_REF = np.array([[0.5626, 0.2159],
                    [0.7201, 0.8012],
                    [0.4062, 0.5581]])
    
    # Reference maximum stain concentrations for H&E
    MAX_C_REF = np.array([1.9705, 1.0308])

    # number of tiles to use in the estimation
    SAMPLED_TILES = 64

    def _get_H_matrix(self, sampled_tiles):
        '''
        Helper method that determines a projection basis
        for the images provided in `sampled_tiles`
        which has size (n,h,w,3) 
        '''
        # arranges the n tiles (each of size h,w,3) so that columns [0,1,2] are R,G,B pixels values
        sampled_pixels = sampled_tiles.reshape(-1,3)

        opt_density = -np.log10((sampled_pixels.astype(np.float32)+1)/NormalizingHandETileExtractor.IO)

        # remove pixels that have very low optical density
        opt_density_hat = opt_density[~np.any(opt_density < NormalizingHandETileExtractor.BETA, axis=1)]

        # get the eigenvectors of the covariance matrix (3x3 matrix).
        # Note that eig_vecs is constructed such that the FINAL
        # columns correspond to the largest eigenvalues, so we reverse the column order
        # to put the first principal component in the first column for `eigenbasis`
        eig_vals, eig_vecs = np.linalg.eigh(np.cov(opt_density.T))
        eigenbasis = eig_vecs[:,2:0:-1]

        # project the RGB vectors into the plane spanned by the largest two eigenvectors
        # pp = "projected pixels". 
        pp = opt_density_hat.dot(eigenbasis)

        # https://numpy.org/doc/2.1/reference/generated/numpy.arctan2.html
        # For each projected pixel (in the 2-d eigenspace), get the angle of
        # the point with respect to the SECOND principal axis. Note that first 
        # arg is the "y" coord and second is "x", so this flips the role of 
        # the x and y coords. Note that if we used arctan2 "properly"
        # (e.g. np.arctan2(pp[:,1], pp[:,0])), then projected points near the
        # negative horizontal axis get min/min phi values around -180,180
        # and the v_min/max below are basically the same vector. Given that
        # the first principal component has the largest amount of variance
        # then this will typically work. Could be corner cases, however.
        phi = np.arctan2(pp[:,0], pp[:,1])

        # remove the extreme outliers to get "boundaries" on the majority
        # of the projected points
        min_phi = np.percentile(phi, NormalizingHandETileExtractor.ALPHA)
        max_phi = np.percentile(phi, 100-NormalizingHandETileExtractor.ALPHA)

        # construct two unit vectors in the eigenspace that align with the 
        # "boundaries" of the projected points
        v_min = eigenbasis.dot(np.array([(np.sin(min_phi), np.cos(min_phi))]).T)
        v_max = eigenbasis.dot(np.array([(np.sin(max_phi), np.cos(max_phi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the 
        # one corresponding to eosin second
        if v_min[0] > v_max[0]:    
            H = np.array((v_min[:,0], v_max[:,0])).T     
        else:
            H = np.array((v_max[:,0], v_min[:,0])).T
        return H

    def _project_strain_component(self, X, idx, shape):
        X2 = np.multiply(
            NormalizingHandETileExtractor.IO, 
            np.exp(
                np.dot(
                    -NormalizingHandETileExtractor.H_REF[:,idx][:, np.newaxis],
                    X[idx,:][np.newaxis, :]
                )
            )
        )
        X2[X2>255] = 254
        return np.reshape(X2.T, shape).astype(np.uint8)

    def _transform_tile(self, tile, tile_idx, H, 
                        pixel_dist_threshold=10, 
                        min_acceptable_black_pixel_frac=0.1):

        orig_tile = np.copy(tile)
        h, w, _ = tile.shape
        tile = tile.reshape((-1,3))
        opt_density = -np.log10((tile.astype(np.float32)+1)/NormalizingHandETileExtractor.IO)

        # now that we have the H matrix which defines 
        # get the coordinates of the optical density values (which was in RGB space)
        # in the subspace spanned by v_min and v_max (the column space of H)
        C = np.linalg.lstsq(H, opt_density.T, rcond=None)[0]

        # cut off the extreme values, then scale/stretch those values 
        # to NormalizingHandETileExtractor.MAX_C_REF
        max_C = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
        tmp = np.divide(max_C, NormalizingHandETileExtractor.MAX_C_REF)
        C2 = np.divide(C, tmp[:, np.newaxis])

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                normed_tile = np.multiply(
                    NormalizingHandETileExtractor.IO,
                    np.exp(
                        -(NormalizingHandETileExtractor.H_REF.dot(C2))
                    )
                )
            except RuntimeWarning as ex:
                print(f'Caught a runtime error: {ex}.')
                # err_file = f'tile_idx_{tile_idx}.C2.error.npy'
                # np.save(err_file, C2)
            
        normed_tile[normed_tile>255] = 254

        # if a tile has practically no H signal, then the projections
        # can get weird and you end up with MANY pixel values right 
        # near zero for all 3 channels.
        normed_pixel_dist = np.linalg.norm(normed_tile, axis=0)
        nn = (normed_pixel_dist < pixel_dist_threshold).sum()
        ff = nn/len(normed_pixel_dist)
        if ff > min_acceptable_black_pixel_frac:
            print(f'Warning!: possible tile exception for tile: {tile_idx}.')
            raise Exception('!!!')

        normed_tile = np.reshape(normed_tile.T, (h, w, 3)).astype(np.uint8)  

        # Separating H and E components
        H_component = self._project_strain_component(C2, 0, (h, w, 3))
        # if the eosin component is needed, use this:
        # E_component = self._project_strain_component(C2, 1, (h, w, 3))

        return (normed_tile, H_component)

    def _threshold_whole_image(self, img_path, scale_factor=4, fg_threshold=0.2):

        img = self._load_image(img_path)
        img = self._pad_image(img)

        tile_size = self.tile_info.tile_size
        h, w, c = img.shape
        n_tiles_w = w // tile_size
        n_tiles_h = h // tile_size

        scaled_tile_size = tile_size // scale_factor
        resize_h = n_tiles_h * scaled_tile_size
        resize_w = n_tiles_w * scaled_tile_size

        # downsample the image so we don't have to work as hard
        image_downscaled = resize(img, (resize_h, resize_w))
        
        greyscale = rgb2gray(image_downscaled) # otsu operates on a single-channel image
        thresholds = threshold_multiotsu(greyscale, classes=2)
        binary_img = np.digitize(greyscale, bins=thresholds)
        
        # note that the reshape method first performs a 'ravel' 
        # (e.g. a flattening) before it chunks everything for the reshape.
        # Based on this, we get this somewhat awkward 5-tensor. This is
        # due to the way that the raveling and reshaping traverses the
        # original image
        binary_img = binary_img.reshape(n_tiles_h,
                        scaled_tile_size,
                        n_tiles_w,
                        scaled_tile_size)

        # to ultimately create an array of (scaled_tile_size, scaled_tile_size, 3) tiles, we need to 
        # do this transpose and reshape. This gives array of shape 
        # (N, scaled_tile_size, scaled_tile_size)
        # where N is the number of tiles (i.e. padded size // tile_size)
        tile_array = binary_img.transpose(0,2,1,3).reshape(-1, scaled_tile_size, scaled_tile_size)
        # sum over each tile, then subtract from 1 since 1=background, 0=foreground
        return np.where((1 - (tile_array.sum(axis=(1,2))/scaled_tile_size**2)) > fg_threshold)[0]

    def _remove_outliers(self, v, threshold=3):
        '''
        Using a median absolute deviation strategy, return a boolean
        array where we mark elements of `v` which are outliers.
        '''
        diff = np.sqrt((v - np.median(v))**2)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > threshold

    def _select_tiles(self, img_path, img_id):
        '''
        Applies the method of Macenko (2009) to normalize 
        the image. 

        Tile selection is based on those tiles with highest H-density
        (i.e. those with the most nuclei represented)
        '''
        print(f'Working on {img_path}')
        # by thresholding the entire image, we can focus on regions that have
        # content relative to a largely white background. We do this globally
        # instead of a tile-by-tile basis. If a region is largely white, the 
        # typical H&E transforms can fail or it can end up selecting undesired
        # regions
        passing_tiles = self._threshold_whole_image(img_path)


        # passing_tiles has the indices of tiles that pass the thresholding,
        # and we pass that to _get_raw_tiles
        tiles = self._get_raw_tiles(img_path, tile_index_filter=passing_tiles)

        # now that we have tiles containing foreground content, extract the H component
        # and rank the tiles based on that. This will remove other tissue that does not
        # contain nuclei, ideally focusing on interesting tissue tiles
        h_vals = []
        for tile_idx,tile in enumerate(tiles):
            # use the reference H matrix to perform a first-pass on extracting the Hematoxylin component
            try:
                normed_tile, H_component = self._transform_tile(tile, tile_idx, NormalizingHandETileExtractor.H_REF)
                h_vals.append(H_component.sum())
            except:
                h_vals.append(np.nan)

        h_vals = np.array(h_vals)

        # count how many were marked as np.nan by our heuristic
        bad_tile_sum = np.isnan(h_vals).sum()
        print(f'Found {bad_tile_sum} tiles that were likely'
              ' deficient in H signal')

        # the sort puts the np.nan values at the end
        argsort_h_vals = np.argsort(h_vals)

        # argsort gives ascending order, so the darkest tiles will appear first- we want these.
        # Also trim off the ones that were likely problems.
        tiles = tiles[argsort_h_vals]
        h_vals = h_vals[argsort_h_vals]
        if bad_tile_sum > 0:
            tiles = tiles[:-bad_tile_sum]
            h_vals = h_vals[:-bad_tile_sum]

        # we have removed outright problem tiles at this point, but there can remain
        # those that passed threshold, yet are outliers
        outliers = self._remove_outliers(h_vals)
        h_vals = h_vals[~outliers]
        tiles = tiles[~outliers,:,:,:]

        # a (3,2) matrix used for projection. Use a sampling of the top tiles which
        # were based on the H_REF matrix. This gets us the customized projection matrix
        # specific to this slide
        H = self._get_H_matrix(tiles[:NormalizingHandETileExtractor.SAMPLED_TILES])

        normed_tiles = []
        h_components = []
        for tile_idx,tile in enumerate(tiles):
            try:
                normed_tile, H_component = self._transform_tile(tile, tile_idx, H)
                normed_tiles.append(normed_tile)
                h_components.append(H_component)
            except:
                print(f'Caught an exception on second-pass transform, index={tile_idx}')

        normed_tiles = np.stack(normed_tiles)

        # using the h_component arrays, get the darkest of those by sorting
        # and taking the first n_tiles. We first remove any outliers.
        h_vals = np.array([x.sum() for x in h_components])
        outliers = self._remove_outliers(h_vals)
        h_vals = h_vals[~outliers]
        normed_tiles = normed_tiles[~outliers]
        idx = np.argsort(h_vals)[:self.tile_info.n_tiles]
        return normed_tiles[idx]
    

class PiecewiseTileExtractor(BaseTileExtractor):
    '''
    For cases where the SVS/TIFF images are exceptionally large
    and the resolution is very high (e.g. getting very detailed info)
    then the loading and processing of the entire image can be prohibitive
    and take too long.

    This tile extractor effectively runs a thresholding on the entire image
    to ignore regions (supertiles) which likely have little information. We then
    process the supertiles.
    '''

    MAX_TILES_PER_REGION = 8000
    THUMBNAIL_SCALE_FACTOR = 20
    MINIMUM_FOREGROUND_FRACTION = 0.05

    # a temporary directory where we store tile images.
    # After determining the final set of tiles, we copy
    # from there to the final output directory
    TMP_DIR = '/scratch'

    def _select_tiles(self, img_path, img_id):

        # before we go any further, check that we have a 
        # directory at TMP_DIR
        tmp_dir = Path(PiecewiseTileExtractor.TMP_DIR)
        if not tmp_dir.exists():
            raise Exception('For this tile extractor, we need a temporary directory.'
                            f' Attempted to locate one at {tmp_dir}, but it did'
                            ' not exist.')

        osi = self._open_multilayer(img_path)
        self._calculate_supertiles(osi)
        
        # by using thresholding to get foreground/background,
        # we can limit which regions of the large image we visit
        # and extract tiles from
        supertile_tuples = self._get_informative_supertiles(osi)

        tile_stats = []
        for i,j in supertile_tuples:
            start_x, stop_x, start_y, stop_y = self._get_supertile_coords(osi, i, j)
            img_patch = self._read_patch(osi, 
                                         self.tile_info.level, 
                                         (start_x, start_y), 
                                         (stop_x, stop_y))
            # now that we have a manageable patch, tile it
            tiles_from_patch = self._get_tiles_from_array(img_patch)

            # perform things like filtering low-variance, etc. - implement
            # in a child class
            tiles_from_patch, tile_pixel_sums = self._post_process_tiles(tiles_from_patch)

            # track the tile statistics so we can then go back and find those
            # that were 'best' over the ENTIRE image
            tile_stats.extend(self._stash_tmp_tiles(
                tiles_from_patch, 
                tile_pixel_sums, 
                tmp_dir, 
                img_id,
                i, j
            ))

        # now that we've completed going region by region, review 
        # the 'best' tiles and save them to the final directory
        tile_stats = pd.DataFrame(tile_stats).sort_values('pixel_sum', ascending=True)
        return self._grab_final_tiles(tile_stats)

    def _grab_final_tiles(self, tile_stats):
        final_tile_array = []
        for i in range(self.tile_info.n_tiles):
            stats = tile_stats.iloc[i]
            src_tile = stats.tmp_path
            final_tile_array.append(skimage.io.imread(src_tile))
        return np.stack(final_tile_array)
    
    def _stash_tmp_tiles(self, tiles, tile_pixel_sums, tmp_dir, img_id, supertile_i, supertile_j):
        tile_stats = []
        for ii in range(tiles.shape[0]):
            t = tiles[ii, :].astype(np.uint8)
            fout = tmp_dir / f'{img_id}.supertile_{supertile_i}_{supertile_j}.tile_{ii}.png'
            s = {
                'pixel_sum': tile_pixel_sums[ii],
                'tmp_path': fout
            }
            tile_stats.append(s)
            skimage.io.imsave(fout, t)
        return tile_stats

    def _post_process_tiles(self, tile_array):
        return tile_array, None

    def _get_informative_supertiles(self, osi):

        # get a numpy array giving the reduced size image (thumbnail) 
        # with padding applied to rougly match the fullsize image. 
        # This has shape (h,w,3)
        thumbnail = self._load_thumbnail(osi)

        # a binary array giving 1=background, 0=foreground
        binary_img = self._threshold_whole_image(thumbnail)

        # scale the supertile sizes by the same factor
        scaled_supertile_w = (self.supertile_w_tilecount * self.tile_info.tile_size) // PiecewiseTileExtractor.THUMBNAIL_SCALE_FACTOR
        scaled_supertile_h = (self.supertile_h_tilecount * self.tile_info.tile_size) // PiecewiseTileExtractor.THUMBNAIL_SCALE_FACTOR

        informative_supertiles = []
        for i in range(self.num_supertile_h):
            for j in range(self.num_supertile_w):
                start_x = j * scaled_supertile_w
                start_y = i * scaled_supertile_h
                # since indexing into a np array, don't need to worry about exceeding the coordinates
                stop_x = (j+1)*scaled_supertile_w 
                stop_y = (i+1)*scaled_supertile_h

                binary_supertile = binary_img[start_y:stop_y, start_x:stop_x]
                # given that background=1, we subtract the background fraction from 1
                fraction_foreground = 1 - binary_supertile.sum() / np.prod(binary_supertile.shape)
                if fraction_foreground > PiecewiseTileExtractor.MINIMUM_FOREGROUND_FRACTION:
                    informative_supertiles.append((i,j))
        return informative_supertiles

    def _load_thumbnail(self, osi):
        '''
        Returns a numpy array which is a scaled version of the entire
        slide image. We also add scaled padding so that the image
        and subsequent padding are similar to the padded entire slide image.

        The returned array has dimension of (h,w,3) where h is the height of
        the thumbnail and w is the width
        '''
        raw_w, raw_h = osi.level_dimensions[self.tile_info.level]
        pad_w, pad_h = self._calculate_padding(raw_w, raw_h)

        # given the way the `get_thumbnail` method works, we can pass
        # the same number both times and it will appropriately scale
        # the other dimension.
        # Note that the `get_thumbnail` method returns a PIL.Image.Image
        # with height H and width W. After casting to a numpy array,
        # the np arry has shape (H,W,3)
        scaled_w = raw_w // PiecewiseTileExtractor.THUMBNAIL_SCALE_FACTOR
        thumbnail = osi.get_thumbnail((scaled_w, scaled_w))
        thumbnail = np.array(thumbnail)

        # add the padding (scaled) so we roughly match the full-size image
        scaled_pad_w = pad_w // PiecewiseTileExtractor.THUMBNAIL_SCALE_FACTOR
        scaled_pad_h = pad_h // PiecewiseTileExtractor.THUMBNAIL_SCALE_FACTOR

        return self._apply_white_padding(thumbnail, scaled_pad_w, scaled_pad_h)

    def _calculate_supertiles(self, osi):
        '''
        Helps with splitting up the large image into 'supertiles' that
        are close to a desired size determined by MAX_TILES_PER_REGION.

        In the end, this method will tell you how many supertiles there are
        and how many tiles fit into each supertile
        '''

        # dimensions (pre-padding) of the image at the desired level
        raw_w, raw_h = osi.level_dimensions[self.tile_info.level]
        
        # how many pixels to add such that we tile evenly
        pad_w, pad_h = self._calculate_padding(raw_w, raw_h)
        
        padded_w = raw_w + pad_w
        padded_h = raw_h + pad_h

        aspect_ratio = padded_w / padded_h

        # given the aspect ratio and `max_tiles_per_region`, we can create a supertile
        # This gives the number of tiles in the supertile
        supertile_h_tilecount = int(np.sqrt(PiecewiseTileExtractor.MAX_TILES_PER_REGION / aspect_ratio) )
        supertile_w_tilecount = int(aspect_ratio * supertile_h_tilecount)
        
        # number of supertiles. Since we are (roughly) matching the aspect
        # ratio of the image, the number of supertiles in both the vertical
        # and horizontal directions will be the similar. However, we choose
        # the maximum here. Otherwise, the supertiles could be too large
        # and we might end up with more tiles in each supertile
        total_tiles_h = padded_h // self.tile_info.tile_size
        total_tiles_w = padded_w // self.tile_info.tile_size
        num_supertile_h = 1 + total_tiles_h // supertile_h_tilecount
        num_supertile_w = 1 + total_tiles_w // supertile_w_tilecount

        self.num_supertile_w = num_supertile_w
        self.num_supertile_h = num_supertile_h
        self.supertile_w_tilecount = supertile_w_tilecount
        self.supertile_h_tilecount = supertile_h_tilecount

    def _get_supertile_coords(self, osi, i, j):
        '''
        Return the coordinates of the supertile while accounting for the padding, etc.
        i,j gives the row,col of the supertile
        supertile_*_tilecount gives the number of tiles in w or h directions for the supertile
        num_supertile gives the number of supertiles needed to span the whole image. Note that
        the final supertile may not be full-size in general.
        '''
        if (i >= self.num_supertile_h):
            raise Exception('Requesting a supertile that is out of bounds in the vertical direction')
        
        if (j >= self.num_supertile_w):
            raise Exception('Requesting a supertile that is out of bounds in the horizontal direction')
        
        # dimensions (pre-padding) of the image at the desired level
        raw_w, raw_h = osi.level_dimensions[self.tile_info.level]
        
        # how many pixels to add such that we tile evenly
        pad_w, pad_h = self._calculate_padding(raw_w, raw_h)

        # how the padding is applied to the whole image
        left_pad = pad_w // 2
        top_pad = pad_h // 2

        # dimensions of supertile in pixels
        s_w = self.supertile_w_tilecount * self.tile_info.tile_size
        s_h = self.supertile_h_tilecount * self.tile_info.tile_size
        
        # if we are on the borders, we need to account for the fact that we have padding applied
        # to the image such that we can evenly tile. the min/max accounts for this
        start_x = np.max([0, j*s_w - left_pad])
        start_y = np.max([0, i*s_h - top_pad])
        stop_x = np.min([(j+1)*s_w - left_pad, raw_w])
        stop_y = np.min([(i+1)*s_h - top_pad, raw_h])

        return start_x, stop_x, start_y, stop_y

    def _threshold_whole_image(self, img):        
        greyscale = rgb2gray(img) # otsu operates on a single-channel image
        thresholds = threshold_multiotsu(greyscale, classes=2)
        return np.digitize(greyscale, bins=thresholds)
        

class DensityBasedPiecewiseTileExtractor(DensityBasedTileMixin, PiecewiseTileExtractor):
    pass
