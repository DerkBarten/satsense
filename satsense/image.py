"""
Methods for loading images
"""

import math
import warnings

import numpy as np
from osgeo import gdal
from six import iteritems
from skimage import color, img_as_ubyte
from skimage.feature import canny
from skimage.filters.rank import equalize
from skimage.morphology import disk

from .bands import MONOCHROME, RGB

gdal.AllRegister()


class Image:
    def __init__(self, image, bands):
        self._bands = bands
        self.raw = image

        # Most common image formats for features
        self._normalized_image = None
        self._rgb_image = None
        self._grayscale_image = None
        self._gray_ubyte_image = None
        self._canny_edge_image = None

        self._normalization_parameters = {
            'technique': 'cumulative',
            'percentiles': [2.0, 98.0],
            'numstds': 2,
        }

    @property
    def bands(self):
        return self._bands

    @property
    def normalized(self):
        if self._normalized_image is None:
            self._normalized_image = normalize_image(
                self.raw, self.bands, **self._normalization_parameters)
        return self._normalized_image

    @property
    def rgb(self):
        if self._rgb_image is None:
            self._rgb_image = get_rgb_bands(self.normalized, self.bands)
        return self._rgb_image

    @property
    def grayscale(self):
        if self._grayscale_image is None:
            self._grayscale_image = get_grayscale_image(self.rgb, RGB)
        return self._grayscale_image

    @property
    def gray_ubyte(self):
        if self._gray_ubyte_image is None:
            self._gray_ubyte_image = get_gray_ubyte_image(self.rgb, RGB)
        return self._gray_ubyte_image

    @property
    def canny_edged(self):
        if self._canny_edge_image is None:
            if isinstance(self, Window):
                raise ValueError("Unable to compute canny_edged on Window, "
                                 "compute this on the full image.")
            self._canny_edge_image = get_canny_edge_image(
                self.grayscale, radius=30, sigma=0.5)

        return self._canny_edge_image

    @property
    def shape(self):
        return self.raw.shape

    def shallow_copy_range(self, x_range, y_range, pad=True):
        img = Image(self.raw[x_range, y_range], self._bands)

        # We need a normalized image, because normalization breaks
        # if you do it on a smaller range
        img._normalized_image = self.normalized[x_range, y_range]

        # These we can calculate later if they do not exist
        if self._rgb_image is not None:
            img._rgb_image = self._rgb_image[x_range, y_range]
        if self._grayscale_image is not None:
            img._grayscale_image = self._grayscale_image[x_range, y_range]
        if self._gray_ubyte_image is not None:
            img._gray_ubyte_image = self._gray_ubyte_image[x_range, y_range]

        # Get canny edged image and automatically calculate it if it was not defined yet.
        if self._canny_edge_image is not None:
            img._canny_edge_image = self.canny_edged[x_range, y_range]

        # Check whether we need padding. This should only be needed at the
        # right and bottom edges of the image
        x_pad_before = 0
        y_pad_before = 0

        x_pad_after = 0
        y_pad_after = 0
        pad_needed = False
        if x_range.stop > self.raw.shape[0]:
            pad_needed = True
            x_pad_after = math.ceil(x_range.stop - self.raw.shape[0])
        if y_range.stop > self.raw.shape[1]:
            pad_needed = True
            y_pad_after = math.ceil(y_range.stop - self.raw.shape[1])

        if pad and pad_needed:
            img.pad(x_pad_before, x_pad_after, y_pad_before, y_pad_after)

        return img

    def pad(self, x_pad_before: int, x_pad_after: int, y_pad_before: int,
            y_pad_after: int):
        image_formats = [
            'raw',
            '_normalized_image',
            '_rgb_image',
            '_grayscale_image',
            '_gray_ubyte_image',
            '_canny_edge_image',
        ]

        for image_format in image_formats:
            img = getattr(self, image_format, None)
            if img is None:
                continue

            pad_width = (
                (x_pad_before, x_pad_after),
                (y_pad_before, y_pad_after),
            )
            if len(img.shape) == 3:
                pad_width += ((0, 0), )

            img = np.pad(img, pad_width, 'constant', constant_values=0)
            setattr(self, image_format, img)


class Window(Image):
    """
    Part of an image at a certain x, y location
    with a x_range, y_range extent (slice)
    """

    def __init__(self,
                 image: Image,
                 x: int,
                 y: int,
                 x_range: slice,
                 y_range: slice,
                 orig: Image = None):
        super(Window, self).__init__(None, image.bands)

        self.raw = image.raw
        self._normalized_image = image._normalized_image
        self._rgb_image = image._rgb_image
        self._grayscale_image = image._grayscale_image
        self._gray_ubyte_image = image._gray_ubyte_image
        self._canny_edge_image = image._canny_edge_image

        self.x = x
        self.y = y
        self.x_range = x_range
        self.y_range = y_range

        if orig:
            self.image = orig
        else:
            self.image = image


class SatelliteImage(Image):
    def __init__(self, dataset, array, bands, image_name=''):
        super(SatelliteImage, self).__init__(array, bands)
        self.__dataset = dataset
        self.__name = image_name

    @property
    def name(self):
        return self.__name

    @staticmethod
    def load_from_file(path, bands):
        """
        Loads the specified path from file and loads the bands into a numpy array

        @returns dataset The raw gdal dataset
                image The image loaded as a numpy array
        """
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        array = dataset.ReadAsArray()

        if len(array.shape) == 3:
            # The bands column is in the first position, but we want it last
            array = np.rollaxis(array, 0, 3)
        elif len(array.shape) == 2:
            # This image seems to have one band, so we add an axis for ease
            # of use in the rest of the library
            array = array[:, :, np.newaxis]

        image = array.astype('float32')

        return SatelliteImage(dataset, image, bands)


def normalize_image(image,
                    bands,
                    technique='cumulative',
                    percentiles=(2.0, 98.0),
                    numstds=2):
    """
    Normalizes the image based on the band maximum
    """

    normalized_image = image.copy()
    for name, band in iteritems(bands):
        # print("Normalizing band number: {0} {1}".format(band, name))
        if technique == 'cumulative':
            percents = np.percentile(image[:, :, band], percentiles)
            new_min, new_max = percents
        elif technique == 'meanstd':
            mean = normalized_image[:, :, band].mean()
            std = normalized_image[:, :, band].std()

            new_min = mean - (numstds * std)
            new_max = mean + (numstds * std)
        else:
            new_min = normalized_image[:, :, band].min()
            new_max = normalized_image[:, :, band].max()

        normalized_image[:, :, band] = remap(normalized_image[:, :, band],
                                             new_min, new_max, 0, 1)

        np.clip(
            normalized_image[:, :, band],
            a_min=0,
            a_max=1,
            out=normalized_image[:, :, band])

    return normalized_image


def get_rgb_bands(image, bands):
    """
    Converts the image to rgb format.
    """
    if bands is not MONOCHROME:
        red = image[:, :, bands['red']]
        green = image[:, :, bands['green']]
        blue = image[:, :, bands['blue']]

        img = np.rollaxis(np.array([red, green, blue]), 0, 3)
    else:
        img = color.grey2rgb(image)

    return img


def remap(x, o_min, o_max, n_min, n_max):
    # range check
    if o_min == o_max:
        # print("Warning: Zero input range")
        return 0

    if n_min == n_max:
        # print("Warning: Zero output range")
        return 0

    # check reversed input range
    reverse_input = False
    old_min = min(o_min, o_max)
    old_max = max(o_min, o_max)
    if not old_min == o_min:
        reverse_input = True

    # check reversed output range
    reverse_output = False
    new_min = min(n_min, n_max)
    new_max = max(n_min, n_max)
    if not new_min == n_min:
        reverse_output = True

#     print("Remapping from range [{0}-{1}] to [{2}-{3}]"
#           .format(old_min, old_max, new_min, new_max))
    scale = (new_max - new_min) / (old_max - old_min)
    if reverse_input:
        portion = (old_max - x) * scale
    else:
        portion = (x - old_min) * scale

    if reverse_output:
        result = new_max - portion
    else:
        result = portion + new_min

    return result


def get_grayscale_image(image, bands=RGB):
    if bands is not RGB:
        rgb_image = get_rgb_bands(image, bands)
    else:
        rgb_image = image

    return color.rgb2gray(rgb_image)


def get_gray_ubyte_image(image, bands=RGB):
    """
    Converts image in 0 - 1 scale format to ubyte 0 - 255 format

    Uses img_as_ubyte from skimage
    """
    if bands is not MONOCHROME:
        gray = get_grayscale_image(image, bands)
    else:
        gray = image

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore loss of precision warning
        return img_as_ubyte(gray)


def get_canny_edge_image(image, radius, sigma):
    """
    Compute Canny edge image
    """
    # local histogram equalization
    grayscale = equalize(image, selem=disk(radius))
    try:
        return canny(grayscale, sigma=sigma)
    except TypeError:
        print("Canny type error")
        return np.zeros(image.shape)
