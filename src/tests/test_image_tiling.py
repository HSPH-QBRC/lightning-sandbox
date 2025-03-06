import unittest
import unittest.mock as mock
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd

from utils.image_utils import TileInfo, \
    BaseTileExtractor, \
    DensityBasedTileExtractor, \
    NormalizingHandETileExtractor

class TestDensityBasedTileExtractor(unittest.TestCase):

    def setUp(self):
        self.num_tiles = 8
        tile_info = TileInfo(
            self.num_tiles,
            192,
            1,
            0
        )
        self.extractor = DensityBasedTileExtractor(tile_info)

    @mock.patch.object(DensityBasedTileExtractor, '_get_raw_tiles')
    def test_extract(self, mock_get_raw_tiles):

        tiles = []
        for i in range(3):
            tiles.append(np.zeros((5,4,3)) + (3-i))
        tiles = np.stack(tiles)
        orig_means = tiles.mean(axis=(1,2,3))
        self.assertTrue(np.allclose(orig_means, np.array([3,2,1])))

        mock_get_raw_tiles.return_value = tiles
        sorted_tiles = self.extractor.extract('')
        sorted_means = sorted_tiles.mean(axis=(1,2,3))
        
        # since we requested self.num_tiles, the function pads
        # out the result with matrices of all white (255)
        self.assertTrue(np.allclose(
            sorted_means, 
            np.concatenate([np.array([1,2,3]), np.repeat(255,self.num_tiles - 3)])
        ))


class TestBaseTileExtractor(unittest.TestCase):

    def setUp(self):
        self.tile_size = 192
        tile_info = TileInfo(
            16,
            self.tile_size,
            1,
            0
        )
        self.extractor = BaseTileExtractor(tile_info)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_image_pad(self):
        orig_h = 5 * self.tile_size - 3
        orig_w = 3 * self.tile_size - 1
        img = np.random.randint(0,255,size=(orig_h, orig_w, 3))
        padded_img = self.extractor._pad_image(img)
        new_h, new_w, _ = padded_img.shape
        self.assertEqual(new_h, 5*self.tile_size)
        self.assertEqual(new_w, 3*self.tile_size)

    @mock.patch.object(BaseTileExtractor, '_load_image')
    @mock.patch.object(BaseTileExtractor, '_pad_image')
    def test_get_raw_tiles(self, mock_pad, mock_load):
        '''
        Tests that the extraction of tiles from the image
        works as expected
        '''
        tile_size = 4
        tile_rows = 2
        tile_cols = 3
        tile_info = TileInfo(
            tile_rows*tile_cols,
            tile_size,
            1,
            0
        )
        extractor = BaseTileExtractor(tile_info)

        # create an image that has pixels such that we can check that
        # the tiling operation worked as expected due to potentiall complex
        # reshaping, etc.
        pixel_val = 0
        img = np.empty((tile_rows*tile_size, tile_cols*tile_size, 3))
        for tile_idx in range(tile_rows*tile_cols):
            i = tile_idx // tile_cols
            j = tile_idx % tile_cols
            # this creates a (tile_size, tile_size, 3) array such that the first layer
            # e.g. the 'red' channel has the same value of `pixel_val`. The second channel
            # e.g. the 'green' has the same value of `pixel_val + 1`, etc.
            x = np.tile(np.arange(pixel_val, pixel_val+3), tile_size**2).reshape(tile_size, tile_size, 3)
            i0 = i*tile_size
            j0 = j*tile_size
            img[i0:i0+tile_size, j0:j0+tile_size, :] = x
            pixel_val += 3
            
        mock_pad.return_value = img
        mock_path = 'some-path'
        tiles = extractor._get_raw_tiles(mock_path)
        self.assertEqual(tiles.shape, (tile_rows*tile_cols, tile_size, tile_size, 3))
        mock_load.assert_called_once_with(mock_path)
        # now check that the tiles were extracted in the expected manner
        for i,t in enumerate(tiles):
            x = np.tile(np.arange(i*3, i*3+3), tile_size**2).reshape(tile_size, tile_size, 3)
            self.assertTrue(np.allclose(x, t))

        # now try with a filter:
        filtered_tiles = extractor._get_raw_tiles(mock_path, tile_index_filter=[1,5])
        self.assertTrue(len(filtered_tiles) == 2)
        self.assertTrue(np.allclose(filtered_tiles[0], tiles[1]))
        self.assertTrue(np.allclose(filtered_tiles[1], tiles[5]))

    @mock.patch.object(BaseTileExtractor, '_create_output_dir')
    @mock.patch.object(BaseTileExtractor, '_get_image_path')
    @mock.patch.object(BaseTileExtractor, '_get_final_output_dir')
    @mock.patch.object(BaseTileExtractor, '_select_tiles')
    @mock.patch('utils.image_utils.pd')
    @mock.patch('utils.image_utils.skimage')
    def test_extract_and_save_sharded(self, 
                              mock_skimage,
                              mock_pandas, 
                              mock_select_tiles, 
                              mock_get_final_output_dir, 
                              mock_get_image_path, 
                              mock_create_output_dir):
        '''
        This checks that we make the proper number of calls to 
        save images when we are sharding the process. The 
        `test_extract_and_save` method tests that the structure
        of those calls are as expected.
        '''
        num_img = 20
        mock_pandas.read_csv.return_value = pd.DataFrame({
            'image_id': [f'img{x}' for x in range(num_img)],
            'shard': np.concatenate([np.repeat(0,10), np.repeat(1,10)])
        })
        num_tiles = 5
        mock_tiles = np.random.randint(0, 255, size=(num_tiles, 2,2,3)).astype(np.uint8)
        mock_select_tiles.return_value = mock_tiles
        self.extractor.extract_and_save('', '', '', shard=0)
        mock_calls = mock_skimage.io.imsave.mock_calls
        self.assertTrue(len(mock_calls) == 50)

    @mock.patch.object(BaseTileExtractor, '_create_output_dir')
    @mock.patch.object(BaseTileExtractor, '_get_image_path')
    @mock.patch.object(BaseTileExtractor, '_get_final_output_dir')
    @mock.patch.object(BaseTileExtractor, '_select_tiles')
    @mock.patch('utils.image_utils.pd')
    @mock.patch('utils.image_utils.skimage')
    def test_extract_and_save_case1(self, 
                              mock_skimage,
                              mock_pandas, 
                              mock_select_tiles, 
                              mock_get_final_output_dir, 
                              mock_get_image_path, 
                              mock_create_output_dir):
        num_img = 2
        num_tiles = 3
        mock_pandas.read_csv.return_value = pd.DataFrame({
            'image_id': [f'img{x}' for x in range(num_img)]
        })

        mock_get_image_path.side_effect = ['p1','p2']
        mock_final_output_dirs = [Path('outdir1'), Path('outdir2')]
        mock_get_final_output_dir.side_effect = mock_final_output_dirs
        mock_tiles = np.random.randint(0, 255, size=(num_tiles, 2,2,3)).astype(np.uint8)
        mock_select_tiles.return_value = mock_tiles
        mock_image_metadata_path = '/some/path'
        mock_input_dir = '/my/inputs'
        mock_output_dir = '/my/outputs'
        self.extractor.extract_and_save(mock_image_metadata_path, mock_input_dir, mock_output_dir)
        mock_calls = mock_skimage.io.imsave.mock_calls
        call_idx = 0
        for i in range(num_img):
            for j in range(num_tiles):
                f = mock_final_output_dirs[i] / f'img{i}.tile_{j}.png'
                actual_call = mock_calls[call_idx]
                first_arg = actual_call.args[0]
                second_arg = actual_call.args[1]
                self.assertTrue(first_arg == f)
                self.assertTrue(np.allclose(second_arg, mock_tiles[j]))
                call_idx += 1




    @mock.patch.object(BaseTileExtractor, '_create_output_dir')
    @mock.patch.object(BaseTileExtractor, '_get_image_path')
    @mock.patch.object(BaseTileExtractor, '_select_tiles')
    @mock.patch('utils.image_utils.pd')
    def test_extract_and_save_case2(self, 
                              mock_pandas, 
                              mock_select_tiles, 
                              mock_get_image_path, 
                              mock_create_output_dir):
        num_img = 2
        num_tiles = 3
        mock_pandas.read_csv.return_value = pd.DataFrame({
            'image_id': [f'img{x}' for x in range(num_img)]
        })

        mock_get_image_path.side_effect = ['p1','p2']

        mock_tiles = np.random.randint(0, 255, size=(num_tiles, 2,2,3)).astype(np.uint8)
        mock_select_tiles.return_value = mock_tiles
        self.extractor.extract_and_save('', '', Path(self.test_dir))

        for i in range(num_img):
            for j in range(num_tiles):
                fname = Path(self.test_dir) / f'img{i}.tile_{j}.png'
                self.assertTrue(fname.exists())

    @mock.patch.object(BaseTileExtractor, '_create_output_dir')
    @mock.patch.object(BaseTileExtractor, '_get_image_path')
    @mock.patch.object(BaseTileExtractor, '_select_tiles')
    @mock.patch('utils.image_utils.pd')
    def test_extract_and_save_case3(self, 
                              mock_pandas, 
                              mock_select_tiles, 
                              mock_get_image_path, 
                              mock_create_output_dir):
        num_img = 2
        num_tiles = 3
        mock_pandas.read_csv.return_value = pd.DataFrame({
            'image_id': [f'img{x}' for x in range(num_img)],
            'image_subdir': [f'subdir{x}' for x in range(num_img)]
        })

        mock_get_image_path.side_effect = ['p1','p2']

        mock_tiles = np.random.randint(0, 255, size=(num_tiles, 2,2,3)).astype(np.uint8)
        mock_select_tiles.return_value = mock_tiles
        self.extractor.extract_and_save('', '', Path(self.test_dir))

        for i in range(num_img):
            for j in range(num_tiles):
                fname = Path(self.test_dir)/ f'subdir{i}' / f'img{i}.tile_{j}.png'
                self.assertTrue(fname.exists())


if __name__ == '__main__':
    unittest.main()