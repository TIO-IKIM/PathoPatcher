# encoding: utf-8

# Code adapted from histolab, but used because of dependency issues
# Original license: https://github.com/histolab/histolab?tab=Apache-2.0-1-ov-file#readme
# ------------------------------------------------------------------------
# Copyright 2022 All Histolab Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# This file consists of code derived from
# histolab/filters/image_filters.py
# histolab/filters/image_filters_functional.py
# histolab/util.py
# without any changes

from abc import abstractmethod
from functools import reduce
from typing import Union

import numpy as np
from PIL import Image
from skimage.util.dtype import img_as_ubyte

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


def np_to_pil(np_img: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a PIL Image.

    Parameters
    ----------
    np_img : np.ndarray
        The image represented as a NumPy array.

    Returns
    -------
    Image.Image
        The image represented as PIL Image
    """

    def _transform_bool(img_array: np.ndarray) -> np.ndarray:
        return img_array.astype(np.uint8) * 255

    def _transform_float(img_array: np.ndarray) -> np.ndarray:
        return (
            img_array.astype(np.uint8)
            if np.max(img_array) > 1
            else img_as_ubyte(img_array)
        )

    types_factory = {
        "bool": _transform_bool(np_img),
        "float64": _transform_float(np_img),
    }
    image_array = types_factory.get(str(np_img.dtype), np_img.astype(np.uint8))
    return Image.fromarray(image_array)


def apply_mask_image(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Mask image with the provided binary mask.

    Parameters
    ----------
    img : Image.Image
        Input image
    mask : np.ndarray
        Binary mask

    Returns
    -------
    Image.Image
        Image with the mask applied
    """
    img_arr = np.array(img)

    if mask.ndim == 2 and img_arr.ndim != 2:
        masked_image = np.zeros(img_arr.shape, "uint8")
        n_channels = img_arr.shape[2]
        for channel_i in range(n_channels):
            masked_image[:, :, channel_i] = img_arr[:, :, channel_i] * mask
    else:
        masked_image = img_arr * mask
    return np_to_pil(masked_image)


def red_filter(
    img: Image, red_thresh: int, green_thresh: int, blue_thresh: int
) -> np.ndarray:
    """Mask reddish colors in an RGB image.

    Create a mask to filter out reddish colors, where the mask is based on a pixel
    being above a red channel threshold value, below a green channel threshold value,
    and below a blue channel threshold value.

    Parameters
    ----------
    img : Image
        Input RGB image
    red_thresh : int
        Red channel lower threshold value.
    green_thresh : int
        Green channel upper threshold value.
    blue_thresh : int
        Blue channel upper threshold value.

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing the mask.
    """
    if np.array(img).ndim != 3:
        raise ValueError("Input must be 3D.")
    if not (
        0 <= red_thresh <= 255 and 0 <= green_thresh <= 255 and 0 <= blue_thresh <= 255
    ):
        raise ValueError("RGB Thresholds must be in range [0, 255]")

    img_arr = np.array(img)
    red = img_arr[:, :, 0] < red_thresh
    green = img_arr[:, :, 1] > green_thresh
    blue = img_arr[:, :, 2] > blue_thresh
    return red | green | blue


def green_filter(
    img: Image, red_thresh: int, green_thresh: int, blue_thresh: int
) -> np.ndarray:
    """Filter out greenish colors in an RGB image.
    The mask is based on a pixel being above a red channel threshold value, below a
    green channel threshold value, and below a blue channel threshold value.

    Note that for the green ink, the green and blue channels tend to track together, so
    for blue channel we use a lower threshold rather than an upper threshold value.

    Parameters
    ----------
    img : Image.Image
        RGB input image.
    red_thresh : int
        Red channel upper threshold value.
    green_thresh : int
        Green channel lower threshold value.
    blue_thresh : int
        Blue channel lower threshold value.

    Returns
    -------
    np.ndarray
        Boolean  NumPy array representing the mask.
    """
    if np.array(img).ndim != 3:
        raise ValueError("Input must be 3D.")
    if not (
        0 <= red_thresh <= 255 and 0 <= green_thresh <= 255 and 0 <= blue_thresh <= 255
    ):
        raise ValueError("RGB Thresholds must be in range [0, 255]")

    img_arr = np.array(img)
    red = img_arr[:, :, 0] > red_thresh
    green = img_arr[:, :, 1] < green_thresh
    blue = img_arr[:, :, 2] < blue_thresh
    return red | green | blue


def blue_filter(
    img: Image.Image, red_thresh: int, green_thresh: int, blue_thresh: int
) -> np.ndarray:
    """Filter out blueish colors in an RGB image.

    Create a mask to filter out blueish colors, where the mask is based on a pixel
    being above a red channel threshold value, above a green channel threshold value,
    and below a blue channel threshold value.

    Parameters
    ----------
    img : Image.Image
        Input RGB image
    red_thresh : int
        Red channel lower threshold value.
    green_thresh : int
        Green channel lower threshold value.
    blue_thresh : int
        Blue channel upper threshold value.

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing the mask.
    """
    if np.array(img).ndim != 3:
        raise ValueError("Input must be 3D.")
    if not (
        0 <= red_thresh <= 255 and 0 <= green_thresh <= 255 and 0 <= blue_thresh <= 255
    ):
        raise ValueError("RGB Thresholds must be in range [0, 255]")
    img_arr = np.array(img)
    red = img_arr[:, :, 0] > red_thresh
    green = img_arr[:, :, 1] > green_thresh
    blue = img_arr[:, :, 2] < blue_thresh
    return red | green | blue


def red_pen_filter(img: Image.Image) -> Image.Image:
    """Filter out red pen marks on diagnostic slides.

    The resulting mask is a composition of red filters with different thresholds
    for the RGB channels.

    Parameters
    ----------
    img : Image.Image
        Input RGB image.

    Returns
    -------
    Image.Image
        Input image with the pen marks filtered out.
    """
    parameters = [
        {"red_thresh": 150, "green_thresh": 80, "blue_thresh": 90},
        {"red_thresh": 110, "green_thresh": 20, "blue_thresh": 30},
        {"red_thresh": 185, "green_thresh": 65, "blue_thresh": 105},
        {"red_thresh": 195, "green_thresh": 85, "blue_thresh": 125},
        {"red_thresh": 220, "green_thresh": 115, "blue_thresh": 145},
        {"red_thresh": 125, "green_thresh": 40, "blue_thresh": 70},
        {"red_thresh": 100, "green_thresh": 50, "blue_thresh": 65},
        {"red_thresh": 85, "green_thresh": 25, "blue_thresh": 45},
    ]
    red_pen_filter_img = reduce(
        (lambda x, y: x & y), [red_filter(img, **param) for param in parameters]
    )
    return apply_mask_image(img, red_pen_filter_img)


def green_pen_filter(img: Image.Image) -> Image.Image:
    """Filter out green pen marks from a diagnostic slide.

    The resulting mask is a composition of green filters with different thresholds
    for the RGB channels.

    Parameters
    ---------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Input image with the green pen marks filtered out.
    """
    parameters = [
        {"red_thresh": 150, "green_thresh": 160, "blue_thresh": 140},
        {"red_thresh": 70, "green_thresh": 110, "blue_thresh": 110},
        {"red_thresh": 45, "green_thresh": 115, "blue_thresh": 100},
        {"red_thresh": 30, "green_thresh": 75, "blue_thresh": 60},
        {"red_thresh": 195, "green_thresh": 220, "blue_thresh": 210},
        {"red_thresh": 225, "green_thresh": 230, "blue_thresh": 225},
        {"red_thresh": 170, "green_thresh": 210, "blue_thresh": 200},
        {"red_thresh": 20, "green_thresh": 30, "blue_thresh": 20},
        {"red_thresh": 50, "green_thresh": 60, "blue_thresh": 40},
        {"red_thresh": 30, "green_thresh": 50, "blue_thresh": 35},
        {"red_thresh": 65, "green_thresh": 70, "blue_thresh": 60},
        {"red_thresh": 100, "green_thresh": 110, "blue_thresh": 105},
        {"red_thresh": 165, "green_thresh": 180, "blue_thresh": 180},
        {"red_thresh": 140, "green_thresh": 140, "blue_thresh": 150},
        {"red_thresh": 185, "green_thresh": 195, "blue_thresh": 195},
    ]

    green_pen_filter_img = reduce(
        (lambda x, y: x & y), [green_filter(img, **param) for param in parameters]
    )
    return apply_mask_image(img, green_pen_filter_img)


def blue_pen_filter(img: Image.Image) -> Image.Image:
    """Filter out blue pen marks from a diagnostic slide.

    The resulting mask is a composition of green filters with different thresholds
    for the RGB channels.

    Parameters
    ---------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Input image with the blue pen marks filtered out.
    """
    parameters = [
        {"red_thresh": 60, "green_thresh": 120, "blue_thresh": 190},
        {"red_thresh": 120, "green_thresh": 170, "blue_thresh": 200},
        {"red_thresh": 175, "green_thresh": 210, "blue_thresh": 230},
        {"red_thresh": 145, "green_thresh": 180, "blue_thresh": 210},
        {"red_thresh": 37, "green_thresh": 95, "blue_thresh": 160},
        {"red_thresh": 30, "green_thresh": 65, "blue_thresh": 130},
        {"red_thresh": 130, "green_thresh": 155, "blue_thresh": 180},
        {"red_thresh": 40, "green_thresh": 35, "blue_thresh": 85},
        {"red_thresh": 30, "green_thresh": 20, "blue_thresh": 65},
        {"red_thresh": 90, "green_thresh": 90, "blue_thresh": 140},
        {"red_thresh": 60, "green_thresh": 60, "blue_thresh": 120},
        {"red_thresh": 110, "green_thresh": 110, "blue_thresh": 175},
    ]

    blue_pen_filter_img = reduce(
        (lambda x, y: x & y), [blue_filter(img, **param) for param in parameters]
    )
    return apply_mask_image(img, blue_pen_filter_img)


@runtime_checkable
class Filter(Protocol):
    """Filter protocol"""

    @abstractmethod
    def __call__(
        self, img: Union[Image.Image, np.ndarray]
    ) -> Union[Image.Image, np.ndarray]:
        pass  # pragma: no cover

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


@runtime_checkable
class ImageFilter(Filter, Protocol):
    """Image filter protocol"""

    @abstractmethod
    def __call__(self, img: Image.Image) -> Union[Image.Image, np.ndarray]:
        pass  # pragma: no cover


class RedPenFilter(ImageFilter):
    """Filter out red pen marks on diagnostic slides.

    The resulting mask is a composition of red filters with different thresholds
    for the RGB channels.

    Parameters
    ----------
    img : Image.Image
        Input RGB image.

    Returns
    -------
    Image.Image
        Image the green red marks filtered out.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RedPenFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-red-pen.png")
        >>> red_pen_filter = RedPenFilter()
        >>> image_no_red = red_pen_filter(image_rgb)
    """

    def __call__(self, img: Image.Image):
        return red_pen_filter(img)


class GreenPenFilter(ImageFilter):
    """Filter out green pen marks from a diagnostic slide.

    The resulting mask is a composition of green filters with different thresholds
    for the RGB channels.

    .. figure:: https://user-images.githubusercontent.com/31658006/116548722-f290e200-a8f4-11eb-9780-0ce5844295dd.png

    Parameters
    ---------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Image the green pen marks filtered out.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import GreenPenFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-green-pen.png")
        >>> green_pen_filter = GreenPenFilter()
        >>> image_no_green = green_pen_filter(image_rgb)
    """  # noqa

    def __call__(self, img: Image.Image) -> Image.Image:
        return green_pen_filter(img)


class BluePenFilter(ImageFilter):
    """Filter out blue pen marks from a diagnostic slide.

    The resulting mask is a composition of green filters with different thresholds
    for the RGB channels.

    Parameters
    ---------
    img : Image.Image
        Input RGB image

    Returns
    -------
    np.ndarray
        NumPy array representing the mask with the blue pen marks filtered out.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import BluePenFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/wsi-blue-pen.png")
        >>> blue_pen_filter = BluePenFilter()
        >>> image_no_blue = blue_pen_filter(image_rgb)
    """  # noqa

    def __call__(self, img: Image.Image) -> Image.Image:
        return blue_pen_filter(img)
