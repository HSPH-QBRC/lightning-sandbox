import numpy as np
import skimage.io


def extract_tiles(img_path, num_tiles=64, tile_size=192, mode=0):
    '''
    Opens the image located at the provided path and extracts 
    the requested number of tiles of shape (tile_shape, tile_shape)
    '''
    result = []

    # Note that the image is given in (H,W,C) and the color channels are RGB
    # (in contrast to cv2.imread which loads BGR)
    img = skimage.io.imread(img_path)
    h, w, c = img.shape

    # Padding which will permit an integer number of (tile_size, tile_size) tiles
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    # pad the image with white pixels (255)
    img = np.pad(img,
                  [
                      [pad_h // 2, pad_h - pad_h // 2],
                      [pad_w // 2,pad_w - pad_w//2],
                      [0,0]
                  ],
                  constant_values=255)

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
    # do this transpose and reshape. This gives array of shape (N, tile_size, tile_size,3)
    # where N is the number of tiles (i.e. padded size // tile_size)
    img = img.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size, 3)

    # a fully white tile has a pixel sum as follows:
    full_white_tile_sum = (tile_size ** 2) * 3 * 255

    # count the number of tiles that are not pure white
    num_tiles_with_info = (img.reshape(img.shape[0],-1).sum(1) < full_white_tile_sum).sum()

    # if the total number of tiles in the full image was fewer than the number of requested
    # tiles, add the required number of fully white tiles
    if len(img) < num_tiles:
        img = np.pad(img,
                    [
                        [0, num_tiles - len(img)],
                        [0, 0],
                        [0, 0],
                        [0, 0]
                    ], constant_values=255)

    # based on the sum of the pixels in each tile, we sort and take the
    # top num_tiles. Recall that np.argsort gives ascending order. We
    # want this since more 'informative' images will have more colored pixels.
    # Majority white tiles will have very large sums and are likely not
    # very informative.
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:num_tiles]
    img = img[idxs]
    return img, num_tiles_with_info >= num_tiles