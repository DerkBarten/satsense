from ..features import NirNDVI, WVSI
from satsense.image import SatelliteImage
from satsense.generators import CellGenerator
import numpy as np
from skimage import filters
from scipy.ndimage import zoom
from satsense.bands import MASK_BANDS
import matplotlib.pyplot as plt
import rasterio
import fiona
#from bands import MASK_BANDS

from ..util import save_mask2file, load_mask_from_file

class Mask():
    def __init__(self, mask):
        self._mask = mask

    @property
    def shape(self):
        return self._mask.shape

    @property
    def mask(self):
        return self._mask
    
    def save(self, path):
        save_mask2file(self._mask, path)

    def load_from_file(path):
        mask = load_mask_from_file(path)
        return Mask(mask)

    def overlay(self, rgb_image):
        zoom_w = rgb_image.shape[1] / self.mask.shape[1]
        zoom_h = rgb_image.shape[0] / self.mask.shape[0]
        zoomed_mask = zoom(self.mask, (zoom_h, zoom_w), order=0)

        plt.imshow(zoomed_mask, cmap='hot', alpha=0.3)

    def __and__(self, other):
        return Mask(np.uint8(np.logical_and(self.mask, other.mask)))

    def __or__(self, other):
        return Mask(np.uint8(np.logical_or(self.mask, other.mask)))

    def __invert__(self):
        return Mask(np.uint8(np.logical_not(self.mask)))

    def __sub__(self, other):
        """ V & ~S
            Removes everything from S that is present in V (in essence removes
            slums that are included in vegetation)
        """
        m = np.logical_and(self.mask, np.logical_not(other.mask))
        return Mask(np.uint8(m))

class VegetationMask(Mask):   
    @staticmethod 
    def create(generator):
        mask = np.zeros(generator.shape)
        ndvi = NirNDVI()
        for cell in generator:
            mask[cell.x, cell.y] = ndvi(cell)
        mask = np.uint8(mask < filters.threshold_otsu(mask))
        return VegetationMask(mask)

# Note: smaller than sign instead bigger than sign
class SoilMask(Mask):
    @staticmethod 
    def create(generator):
        mask = np.zeros(generator.shape)
        wvsi = WVSI()
        for cell in generator:
            mask[cell.x, cell.y] = wvsi(cell)
        mask = np.uint8(mask > filters.threshold_otsu(mask))
        return VegetationMask(mask)

class OnesMask(Mask):
    @staticmethod
    def create(generator):
        mask = np.ones(generator.shape, dtype=np.uint8)
        return OnesMask(mask)

class ShapefileMask(Mask):
    def create(shapefile, imagefile, size):
        with fiona.open(shapefile, "r") as sf:
            geoms = [feature["geometry"] for feature in sf]

        with rasterio.open(imagefile) as src:
            out_image, _ = rasterio.mask.mask(src, geoms, crop=True,
                                                        invert=False)
            out_image[out_image == np.max(out_image)] = 0
            out_image[out_image > 0] = 1
            out_image = out_image[0]
            out_image = np.reshape(out_image, (out_image.shape[0], out_image.shape[1], 1))
            sat_im = SatelliteImage(None, out_image, MASK_BANDS)
        
            generator = CellGenerator(sat_im, size)

            mask = np.zeros(generator.shape)
            for cell in generator:
                mean = np.mean(cell.raw)
                if mean > 0.1:
                    mask[cell.x, cell.y] = 1
        return ShapefileMask(np.uint8(mask))
