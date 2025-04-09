import unittest
import unittest.mock as mock
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd
import skimage

from utils.image_utils import TileInfo, \
    BaseTileExtractor, \
    DensityBasedTileExtractor, \
    NormalizingHandETileExtractor, \
    PiecewiseTileExtractor, \
    DensityBasedTileMixin, \
    DensityBasedPiecewiseTileExtractor


class TestDensityBasedTileMixin(unittest.TestCase):

    def test_post_process(self):
        '''
        Tests the method which removes low-variance tiles
        and sorts the tiles
        '''
        # create 5 tiles. Each one is a (5,4) 'image'
        # with 3 channels. Tiles with index 1,3 will have uniform 
        # pixel values of 1,3 respectively. Those will
        # ultimately be removed since they have no variance
        # The even tiles will have sufficient variance
        tiles = []
        expected_sums = []
        for i in range(5):
            if i % 2 == 0:
                x = np.random.randint(0,255,size=(5,4,3))
                tiles.append(x)
                expected_sums.append(x.sum())
            else:
                tiles.append(np.zeros((5,4,3)) + i)
        expected_sums = sorted(expected_sums)
        tiles = np.stack(tiles)

        extractor = DensityBasedTileMixin()
        filtered_tiles, tile_sums = extractor._post_process_tiles(tiles)
        self.assertTrue(np.allclose(expected_sums, tile_sums))
        self.assertTrue(len(filtered_tiles) == 3)
        for i,t in enumerate(filtered_tiles):
            ts = t.sum()
            self.assertTrue(np.allclose(ts, tile_sums[i]))



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
    @mock.patch.object(DensityBasedTileExtractor, '_post_process_tiles')
    def test_extract(self, mock_post_process_tiles, mock_get_raw_tiles):
        '''
        Checks that we add white tiles to get the required number
        of total tiles.
        '''

        # create 3 tiles. Each one is a (5,4) 'image'
        # with 3 channels. The first tile has pixel values
        # of all 3s, second tile has all 2s, etc
        tiles = []
        for i in range(3):
            tiles.append(np.zeros((5,4,3)) + (3-i))
        tiles = np.stack(tiles)
        # assert that the original tile ordering is as expected
        # and not sorted
        orig_means = tiles.mean(axis=(1,2,3))
        self.assertTrue(np.allclose(orig_means, np.array([3,2,1])))

        # the raw tiles are unsorted
        mock_get_raw_tiles.return_value = tiles

        # the post-processing will sort the tiles, so we need to 
        # mock that
        sorted_tiles = tiles[::-1]
        mock_post_process_tiles.return_value = (sorted_tiles, None)

        result_tiles = self.extractor.extract('', '')
        sorted_means = result_tiles.mean(axis=(1,2,3))

        # since we requested self.num_tiles, the function pads
        # out the result with matrices of all white (255)
        self.assertTrue(np.allclose(
            sorted_means, 
            np.concatenate([np.array([1,2,3]), np.repeat(255,self.num_tiles - 3)])
        ))

    @mock.patch.object(DensityBasedTileExtractor, '_get_raw_tiles')
    def test_extract_low_var_removed(self, mock_get_raw_tiles):
        '''
        Checks that we add white tiles to get the required number
        of total tiles.
        Here we 
        '''

        # create 3 tiles. Each one is a (5,4) 'image'
        # with 3 channels. The first tile has pixel values
        # of all 3s, second tile has all 2s, etc
        tiles = []
        for i in range(3):
            tiles.append(np.zeros((5,4,3)) + (3-i))

        # now create a high-variance tile:
        good_tile = np.random.randint(0,255,size=(5,4,3))
        # to make sure the test should actually work
        self.assertTrue(good_tile.var() > DensityBasedTileExtractor.VARIANCE_THRESHOLD)
        tiles.append(good_tile)
        tiles = np.stack(tiles)

        mock_get_raw_tiles.return_value = tiles
        sorted_tiles = self.extractor.extract('', '')
        sorted_means = sorted_tiles.mean(axis=(1,2,3))
        self.assertTrue(sorted_tiles.shape[0] == self.num_tiles)
        expected_means = [np.mean(good_tile)] + [255.0]*(self.num_tiles-1)
        self.assertTrue(np.allclose(
            sorted_means,
            np.array(expected_means)
        ))

    def test_padding(self):
        '''
        Tests that the proper padding is applied such that
        images can fit an integer number of tiles
        '''
        # given a tile size of 192 and the sizes below,
        # we can fit 520.83 and 416.67 tiles in the w,h
        # directions, respectively. Thus, the image
        # will be padded to accommodate (521,417) tiles
        w = 100000
        h = 80000
        expected_pad_w = (192*521) - w
        expected_pad_h = (192*417) - h
        pad_w, pad_h = self.extractor._calculate_padding(w,h)
        self.assertEqual(pad_w, expected_pad_w)
        self.assertEqual(pad_h, expected_pad_h)


class TestBaseTileExtractor(unittest.TestCase):

    def setUp(self):
        self.tile_size = 192
        self.tile_num = 16
        self.resolution = 1
        self.mode = 0
        tile_info = TileInfo(
            self.tile_num,
            self.tile_size,
            self.resolution,
            self.mode
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

    def test_get_tiles_from_array(self):
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
        # the tiling operation worked as expected due to potentially complex
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
            
        tiles = extractor._get_tiles_from_array(img)
        self.assertEqual(tiles.shape, (tile_rows*tile_cols, tile_size, tile_size, 3))
        # now check that the tiles were extracted in the expected manner
        for i,t in enumerate(tiles):
            x = np.tile(np.arange(i*3, i*3+3), tile_size**2).reshape(tile_size, tile_size, 3)
            self.assertTrue(np.allclose(x, t))

        # now try with a filter:
        filtered_tiles = extractor._get_tiles_from_array(img, tile_index_filter=[1,5])
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


    @mock.patch.object(BaseTileExtractor, '_get_image_path')
    @mock.patch.object(BaseTileExtractor, '_select_tiles')
    @mock.patch('utils.image_utils.pd')
    def test_extract_and_save_case1(self, 
                              mock_pandas, 
                              mock_select_tiles, 
                              mock_get_image_path):
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
                fname = Path(self.test_dir) / f'numtile-{self.tile_num}-tilesize-{self.tile_size}-res-{self.resolution}-mode-{self.mode}' / f'img{i}.tile_{j}.png'
                self.assertTrue(fname.exists())

    @mock.patch.object(BaseTileExtractor, '_get_image_path')
    @mock.patch.object(BaseTileExtractor, '_select_tiles')
    @mock.patch('utils.image_utils.pd')
    def test_extract_and_save_case2(self, 
                              mock_pandas, 
                              mock_select_tiles, 
                              mock_get_image_path):
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
        ff = Path(self.test_dir) / f'numtile-{self.tile_num}-tilesize-{self.tile_size}-res-{self.resolution}-mode-{self.mode}' / 'subdir0'
        for i in range(num_img):
            for j in range(num_tiles):
                fname = Path(self.test_dir) / f'numtile-{self.tile_num}-tilesize-{self.tile_size}-res-{self.resolution}-mode-{self.mode}' / f'subdir{i}' / f'img{i}.tile_{j}.png'
                self.assertTrue(fname.exists())


class TestPiecewiseTileExtractor(unittest.TestCase):

    def setUp(self):
        self.num_tiles = 8
        tile_info = TileInfo(
            self.num_tiles,
            192,
            1,
            0
        )
        self.extractor = PiecewiseTileExtractor(tile_info)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.tmp_dir)

    @mock.patch.object(PiecewiseTileExtractor, 'MAX_TILES_PER_REGION', new=1000)
    def test_supertile_calculation(self):
        mock_osi = mock.MagicMock()
        # given a tile size of 192 and the sizes below,
        # we can fit 520.83 and 416.67 tiles in the w,h
        # directions, respectively. Thus, the image
        # will be padded to accommodate (521,417) tiles
        # (e.g. the total image size with padding will be
        # 100032, 80064)
        # doesn't matter that these are the same below.
        raw_w = 100000
        raw_h = 80000
        mock_osi.level_dimensions = {
            0:[raw_w, raw_h],
            1:[raw_w, raw_h],
            2:[raw_w, raw_h]
        }

        # given the aspect ratio of 1.25, the supertiles will also have
        # that dimension (roughly). Since each supertile can have
        # MAX_TILES_PER_REGION=1000, we can put (34,28) tiles in each supertile.
        # Then each supertile has pixel dimension (6528, 5376). Thus, you can fit 
        # 15.32 supertiles in the horizontal direction and 14.89 in the vertical.
        # To avoid making the supertile larger than we want, we round up
        self.extractor._calculate_supertiles(mock_osi)
        self.assertEqual(self.extractor.num_supertile_w, 16)
        self.assertEqual(self.extractor.num_supertile_h, 15)
        self.assertEqual(self.extractor.supertile_w_tilecount, 34)
        self.assertEqual(self.extractor.supertile_h_tilecount, 28)
        
    def test_supertile_coords(self):
        '''
        Given the supertile counts and dimensions, ensure
        that the supertile extraction grabs the right part
        of the image
        '''
        mock_osi = mock.MagicMock()
        # given a tile size of 192 and the sizes below,
        # we can fit 520.83 and 416.67 tiles in the w,h
        # directions, respectively. Thus, the image
        # will be padded to accommodate (521,417) tiles
        # (e.g. the total image size with padding will be
        # 100032, 80064)
        # doesn't matter that these are the same below.
        raw_w = 100000
        raw_h = 80000
        mock_osi.level_dimensions = {
            0:[raw_w, raw_h],
            1:[raw_w, raw_h],
            2:[raw_w, raw_h]
        }

        # given these tile counts in each supertile, the dimension of a 
        # supertile in pixels is 34*192=6528 and 28*192=5376
        supertile_w_tilecount = 34
        supertile_h_tilecount = 28
        num_supertile_w = 16
        num_supertile_h = 15

        self.extractor.num_supertile_w = num_supertile_w
        self.extractor.num_supertile_h = num_supertile_h
        self.extractor.supertile_w_tilecount = supertile_w_tilecount
        self.extractor.supertile_h_tilecount = supertile_h_tilecount

        # top left tile has to account for top+left padding
        start_x, stop_x, start_y, stop_y = self.extractor._get_supertile_coords(mock_osi, 0, 0)
        self.assertEqual(start_x, 0)
        self.assertEqual(start_y, 0)
        self.assertEqual(stop_x, 6512) # 6528 - 32/2 = 6512
        self.assertEqual(stop_y, 5344) # 5376 - 64/2 = 5344

        # an internal tile still needs to account for the fact that the total
        # image was padded out
        start_x, stop_x, start_y, stop_y = self.extractor._get_supertile_coords(mock_osi, 2, 2)
        self.assertEqual(start_x, 13040) # 2*6528 - 32/2 = 13040
        self.assertEqual(start_y, 10720)     # 2*5376 - 64/2 = 10720
        self.assertEqual(stop_x, 13040+6528) # start_x + supertile dim w
        self.assertEqual(stop_y, 10720+5376) # start_y + supertile dim h


        start_x, stop_x, start_y, stop_y = self.extractor._get_supertile_coords(mock_osi, 2, 15)
        self.assertEqual(start_x, 97904) # 15*6528 - 32/2 = 97904
        self.assertEqual(start_y, 10720)     # 2*5376 - 64/2 = 10720
        self.assertEqual(stop_x, 100000) # raw image has max dimension of 100,000
        self.assertEqual(stop_y, 10720+5376) # start_y + supertile dim h

        start_x, stop_x, start_y, stop_y = self.extractor._get_supertile_coords(mock_osi, 14, 15)
        self.assertEqual(start_x, 97904) # 15*6528 - 32/2 = 97904
        self.assertEqual(start_y, 75232)     # 14*5376 - 64/2 = 75232
        self.assertEqual(stop_x, 100000) # raw image has max width of 100,000
        self.assertEqual(stop_y, 80000) # raw image has max height of 80,000

        # requesting an out of bound tile in the horizontal direction
        with self.assertRaisesRegex(Exception, 'out of bounds in the horizontal'):
            self.extractor._get_supertile_coords(mock_osi, 0, 16)
            
        # requesting an out of bound tile in the vertical direction
        with self.assertRaisesRegex(Exception, 'out of bounds in the vertical'):
            self.extractor._get_supertile_coords(mock_osi, 16, 3)

    @mock.patch.object(PiecewiseTileExtractor, 'THUMBNAIL_SCALE_FACTOR', new=10)
    def test_load_thumbnail(self):

        mock_osi = mock.MagicMock()
        raw_w = 100000
        raw_h = 80000
        mock_osi.level_dimensions = {
            0:[raw_w, raw_h],
            1:[raw_w, raw_h],
            2:[raw_w, raw_h]
        }
        mock_osi.get_thumbnail.return_value = np.random.randint(0, 255, size=(8000,10000,3))

        # if scale factor is 10, then we would get back an unpadded
        # image of size 10000,8000. The horizontal padding on the whole
        # image is 32, so that gets scaled to 3.2 -> 3. Similarly 64 -> 6.4 ->6
        # Thus, we expect back an image that has final size of 10003,8006
        r = self.extractor._load_thumbnail(mock_osi)
        self.assertTrue(r.shape[0] == 8006)
        self.assertTrue(r.shape[1] == 10003)
        self.assertTrue(r.shape[2] == 3)
        
    @mock.patch.object(PiecewiseTileExtractor, 'THUMBNAIL_SCALE_FACTOR', new=100)
    @mock.patch.object(PiecewiseTileExtractor, '_load_thumbnail')
    def test_get_informative_supertiles(self, mock_load_thumbnail):
        '''
        Tests methods associated with extracting supertiles that 
        have some information
        '''

        mock_osi = mock.MagicMock()
        # given a tile size of 192 and the sizes below,
        # we can fit 520.83 and 416.67 tiles in the w,h
        # directions, respectively. Thus, the image
        # will be padded to accommodate (521,417) tiles
        # (e.g. the total image size with padding will be
        # 100032, 80064).
        # This padded image sets up a supertile calculation
        # that will place 16 supertiles in width direction (15.32) and
        # 15 (14.89) in the vertical. 
        # Each supertile will contain (34,28) tiles
        # of size 192 so each supertile has pixel dimension (6528, 5376)
        raw_w = 100000
        raw_h = 80000
        # Note: doesn't matter that these are the same below.
        mock_osi.level_dimensions = {
            0:[raw_w, raw_h],
            1:[raw_w, raw_h],
            2:[raw_w, raw_h]
        }

        # set these values which concern the whole image
        supertile_w_tilecount = 34
        supertile_h_tilecount = 28
        num_supertile_w = 16
        num_supertile_h = 15
        self.extractor.num_supertile_w = num_supertile_w
        self.extractor.num_supertile_h = num_supertile_h
        self.extractor.supertile_w_tilecount = supertile_w_tilecount
        self.extractor.supertile_h_tilecount = supertile_h_tilecount

        # if the scaling factor for the thumbnail is 100, then the thumbnail
        # returned is 1000 wide and 800 high. The padding of 32 and 64, when
        # scaled by 100 rounds to zero since it's inconsequential, so
        # the array returned by PiecewiseTileExtractor._load_thumbnail
        # will be (800,1000,3)
        # start by creating an all-white image which will our "thumbnail"
        mock_thumbnail = 255*np.ones((800,1000,3))

        # add a (w=200, h=100) rectangle to the image starting at pixel
        # (x=50, y=150)
        y_start = 150
        x_start = 50
        rect_h = 100
        rect_w = 200
        mock_thumbnail[y_start:y_start + rect_h, x_start:x_start+rect_w,:] = 0

        # now, given the dimensions, the supertiles in the original image were
        # (6528, 5376) pixels so the scaled dimensions are (65,53). Then,
        # 1000/65 = 15.38 and 800/53=15.09 in terms of the number of supertiles
        # Note that this does create a discrepancy between the original image
        # permitting 14.89 vertical supertiles. However, this sacrifices only a
        # few pixels on the edges so it's not a big deal. 
        # The supertiles and their corresponding coverage fraction are as follows:
        # (2,0): 0.03918722786647311
        # (2,1): 0.16981132075471694
        # (2,2): 0.16981132075471694
        # (2,3): 0.14368650217706824
        # (3,0): 0.23076923076923073
        # (3,1): 1.0
        # (3,2): 1.0
        # (3,3): 0.8461538461538461
        # (4,0): 0.1654571843251088
        # (4,1): 0.7169811320754718
        # (4,2): 0.7169811320754718
        # (4,3): 0.6066763425253991
        mock_load_thumbnail.return_value = mock_thumbnail

        nonzero_supertiles = set([
            (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), 
            (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3)])
        
        # by setting the minimum fraction to 0.1, we will remove tile (2,0)
        PiecewiseTileExtractor.MINIMUM_FOREGROUND_FRACTION = 0.1
        results = self.extractor._get_informative_supertiles(mock_osi)
        dropped = [(2,0)]
        expected_results = [x for x in nonzero_supertiles if not x in dropped]
        self.assertTrue(set(results) == set(expected_results))


        # by setting the minimum fraction to 0.5, we will remove tile (2,0)
        PiecewiseTileExtractor.MINIMUM_FOREGROUND_FRACTION = 0.5
        results = self.extractor._get_informative_supertiles(mock_osi)
        dropped = [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (4, 0)]
        expected_results = [x for x in nonzero_supertiles if not x in dropped]
        self.assertTrue(set(results) == set(expected_results))

        PiecewiseTileExtractor.MINIMUM_FOREGROUND_FRACTION = 0.99
        results = self.extractor._get_informative_supertiles(mock_osi)
        expected_results = [(3,2),(3,1)]
        self.assertTrue(set(results) == set(expected_results))

    @mock.patch('utils.image_utils.skimage')
    def test_grab_final_tiles(self, mock_skimage):
        # save some mock tiles
        num_mock_tiles = 10
        mock_tiles = np.random.randint(0,255, size=(num_mock_tiles,4,4,3)).astype(np.uint8)
        mock_stats = []
        for i in range(num_mock_tiles):
            t = mock_tiles[i]
            output_path = Path(self.tmp_dir) / f'abc.supertile_0_1.tile_{i}.png'
            skimage.io.imsave(output_path, t)
            mock_stats.append({'pixel_sum': t.sum(), 'tmp_path': output_path})
        mock_stats = pd.DataFrame(mock_stats).sort_values('pixel_sum', ascending=True)

        # even though in reality the image reading would return
        # an array representing an image, we just have it return a 
        # single number in this mock
        mock_skimage.io.imread.side_effect = np.arange(num_mock_tiles)
        # now check that we get self.num_tiles back
        result = self.extractor._grab_final_tiles(mock_stats)
        self.assertTrue(result.shape[0] == self.num_tiles)
        self.assertCountEqual(result, list(np.arange(self.num_tiles)))

    @mock.patch('utils.image_utils.skimage')
    def test_stash_tmp_tiles(self, mock_skimage):
        '''
        Checks that we save tiles to a tmp dir and return a 
        data structure that will allow us to evaluate the most
        promising tiles at a later step
        '''
        # create 5 4x4 RGB "image tiles"
        mock_tiles = np.random.randint(0,255, size=(5,4,4,3))
        tile_pixel_sums = np.array([1,2,3,4,5])
        result = self.extractor._stash_tmp_tiles(mock_tiles, 
                                        tile_pixel_sums, Path(self.tmp_dir), 'abc', 22, 34)
        for i, item in enumerate(result):
            s = item['pixel_sum']
            actual_name = item['tmp_path'].name
            expected_name = f'abc.supertile_22_34.tile_{i}.png'
            self.assertEqual(actual_name, expected_name)
            self.assertEqual(s, i+1)
        mock_skimage.io.imsave.assert_called()
        self.assertEqual(5, mock_skimage.io.imsave.call_count)

    def test_post_process(self):
        arr = np.random.randint(0, 255, size=(5,4,4,3))
        result, x = self.extractor._post_process_tiles(arr)
        self.assertTrue(np.allclose(arr, result))
        self.assertIsNone(x)


class TestDensityBasedPiecewiseTileExtractor(unittest.TestCase):
    
    def test_post_process(self):
        tile_info = TileInfo(
            8,
            192,
            1,
            0
        )
        extractor = DensityBasedPiecewiseTileExtractor(tile_info)

        # create 5 tiles. Each one is a (5,4) 'image'
        # with 3 channels. Tiles with index 1,3 will have uniform 
        # pixel values of 1,3 respectively. Those will
        # ultimately be removed since they have no variance
        # The even tiles will have sufficient variance
        tiles = []
        expected_sums = []
        for i in range(5):
            if i % 2 == 0:
                x = np.random.randint(0,255,size=(5,4,3))
                tiles.append(x)
                expected_sums.append(x.sum())
            else:
                tiles.append(np.zeros((5,4,3)) + i)
        expected_sums = sorted(expected_sums)
        tiles = np.stack(tiles)

        filtered_tiles, tile_sums = extractor._post_process_tiles(tiles)
        self.assertTrue(np.allclose(expected_sums, tile_sums))
        self.assertTrue(len(filtered_tiles) == 3)
        for i,t in enumerate(filtered_tiles):
            ts = t.sum()
            self.assertTrue(np.allclose(ts, tile_sums[i]))


if __name__ == '__main__':
    unittest.main()