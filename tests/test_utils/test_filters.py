import unittest

import numpy as np
from pathopatch.utils.filters import (
    apply_mask_image,
    blue_pen_filter,
    green_pen_filter,
    np_to_pil,
    red_pen_filter,
)
from PIL import Image, ImageDraw


def create_color_shades_image():
    # Create an empty image
    width, height = 100, 30
    image = Image.new("RGB", (width, height), "white")

    # Define 10 distinct colors for each category
    green_colors = [
        "#00FF00",
        "#00CC00",
        "#009900",
        "#006600",
        "#003300",
        "#00FF33",
        "#00CC33",
        "#009933",
        "#006633",
        "#003333",
    ]
    blue_colors = [
        "#0000FF",
        "#0000CC",
        "#000099",
        "#000066",
        "#000033",
        "#3333FF",
        "#3333CC",
        "#333399",
        "#333366",
        "#333333",
    ]
    red_colors = [
        "#FF0000",
        "#CC0000",
        "#990000",
        "#660000",
        "#330000",
        "#FF3333",
        "#CC3333",
        "#993333",
        "#663333",
        "#333333",
    ]

    # Paste the colors onto the image
    for i, color in enumerate(green_colors):
        ImageDraw.Draw(image).rectangle([i * 10, 0, (i + 1) * 10, 10], fill=color)

    for i, color in enumerate(blue_colors):
        ImageDraw.Draw(image).rectangle([i * 10, 10, (i + 1) * 10, 20], fill=color)

    for i, color in enumerate(red_colors):
        ImageDraw.Draw(image).rectangle([i * 10, 20, (i + 1) * 10, 30], fill=color)

    return image


def get_green_filtered():
    return np.load("./tests/static_test_files/preprocessing/filters/green.npy")


def get_red_filtered():
    return np.load("./tests/static_test_files/preprocessing/filters/red.npy")


def get_blue_filtered():
    return np.load("./tests/static_test_files/preprocessing/filters/blue.npy")


class TestFilters(unittest.TestCase):
    def test_np_to_pil_with_uint8_array(self):
        """Test np_to_pil with an uint8 array."""
        np_img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        pil_img = np_to_pil(np_img)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, "L")
        self.assertTrue(np.array_equal(np.array(pil_img), np_img))

    def test_np_to_pil_with_bool_array(self):
        """Test np_to_pil with a boolean array."""
        np_img = np.array([[True, False], [False, True]], dtype=bool)
        pil_img = np_to_pil(np_img)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, "L")
        self.assertTrue(
            np.array_equal(np.array(pil_img), np_img.astype(np.uint8) * 255)
        )

    def test_np_to_pil_with_float_array(self):
        """Test np_to_pil with a float array."""
        np_img = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        pil_img = np_to_pil(np_img)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, "L")

    def test_np_to_pil_with_float_array_max_greater_than_one(self):
        """Test np_to_pil with a float array where max value is greater than 1."""
        np_img = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
        pil_img = np_to_pil(np_img)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, "L")
        self.assertTrue(np.array_equal(np.array(pil_img), np_img.astype(np.uint8)))

    def test_apply_mask_image_with_rgb_image_and_2d_mask(self):
        """Test apply_mask_image with an RGB image and a 2D mask."""
        np_img = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
            ],
            dtype=np.uint8,
        )
        pil_img = Image.fromarray(np_img)
        mask = np.array(
            [[True, False, True], [False, True, False]], dtype=bool
        )  # Die Maske hat jetzt die gleiche Form wie das Bild
        masked_img = apply_mask_image(pil_img, mask)
        expected_img = np.array(
            [[[255, 0, 0], [0, 0, 0], [0, 0, 255]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            dtype=np.uint8,
        )
        self.assertTrue(np.array_equal(np.array(masked_img), expected_img))

    def test_green_pen_filter(self):
        pil_img = create_color_shades_image()
        filtered_image = green_pen_filter(pil_img)
        expected_mask = get_green_filtered()
        self.assertTrue(np.array_equal(np.array(filtered_image), expected_mask))

    def test_blue_pen_filter(self):
        pil_img = create_color_shades_image()
        filtered_image = blue_pen_filter(pil_img)
        expected_mask = get_blue_filtered()
        self.assertTrue(np.array_equal(np.array(filtered_image), expected_mask))

    def test_red_pen_filter(self):
        """Test red pen filter"""
        pil_img = create_color_shades_image()
        filtered_image = red_pen_filter(pil_img)
        expected_mask = get_red_filtered()
        self.assertTrue(np.array_equal(np.array(filtered_image), expected_mask))
