# import unittest

# import numpy as np
# from PIL import Image

# from pathopatch.utils.filters import (apply_mask_image, blue_filter,
#                                       blue_pen_filter, green_filter,
#                                       green_pen_filter, np_to_pil, red_filter,
#                                       red_pen_filter)


# class TestFilters(unittest.TestCase):
#     def test_np_to_pil_with_uint8_array(self):
#         """Test np_to_pil with an uint8 array."""
#         np_img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
#         pil_img = np_to_pil(np_img)
#         self.assertIsInstance(pil_img, Image.Image)
#         self.assertEqual(pil_img.mode, 'L')
#         self.assertTrue(np.array_equal(np.array(pil_img), np_img))

#     def test_np_to_pil_with_bool_array(self):
#         """Test np_to_pil with a boolean array."""
#         np_img = np.array([[True, False], [False, True]], dtype=bool)
#         pil_img = np_to_pil(np_img)
#         self.assertIsInstance(pil_img, Image.Image)
#         self.assertEqual(pil_img.mode, 'L')
#         self.assertTrue(np.array_equal(np.array(pil_img), np_img.astype(np.uint8) * 255))

#     def test_np_to_pil_with_float_array(self):
#         """Test np_to_pil with a float array."""
#         np_img = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
#         pil_img = np_to_pil(np_img)
#         self.assertIsInstance(pil_img, Image.Image)
#         self.assertEqual(pil_img.mode, 'L')

#     def test_np_to_pil_with_float_array_max_greater_than_one(self):
#         """Test np_to_pil with a float array where max value is greater than 1."""
#         np_img = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
#         pil_img = np_to_pil(np_img)
#         self.assertIsInstance(pil_img, Image.Image)
#         self.assertEqual(pil_img.mode, 'L')
#         self.assertTrue(np.array_equal(np.array(pil_img), np_img.astype(np.uint8)))

#     def test_apply_mask_image_with_rgb_image_and_2d_mask(self):
#         """Test apply_mask_image with an RGB image and a 2D mask."""
#         np_img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[255, 255, 255], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
#         pil_img = Image.fromarray(np_img)
#         mask = np.array([[True, False, True], [False, True, False]], dtype=bool)  # Die Maske hat jetzt die gleiche Form wie das Bild
#         masked_img = apply_mask_image(pil_img, mask)
#         expected_img = np.array([[[255, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [128, 128, 128], [0, 0, 0]]], dtype=np.uint8)
#         self.assertTrue(np.array_equal(np.array(masked_img), expected_img))

#     def test_green_filter_with_rgb_image(self):
#         """Test green_filter with an RGB image."""
#         np_img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[255, 255, 255], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
#         pil_img = Image.fromarray(np_img)
#         mask = green_filter(pil_img, 128, 128, 128)
#         expected_mask = np.array([[True, False, True], [True, True, False]], dtype=bool)
#         self.assertTrue(np.array_equal(mask, expected_mask))

#     def test_green_filter_with_non_rgb_image(self):
#         """Test green_filter with a non-RGB image."""
#         np_img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
#         pil_img = Image.fromarray(np_img)
#         with self.assertRaises(ValueError):
#             green_filter(pil_img, 128, 128, 128)

#     def test_green_filter_with_invalid_thresholds(self):
#         """Test green_filter with invalid thresholds."""
#         np_img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[255, 255, 255], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
#         pil_img = Image.fromarray(np_img)
#         with self.assertRaises(ValueError):
#             green_filter(pil_img, -1, 128, 128)
#         with self.assertRaises(ValueError):
#             green_filter(pil_img, 128, 256, 128)
#         with self.assertRaises(ValueError):
#             green_filter(pil_img, 128, 128, -1)

#     def test_red_filter_with_rgb_image(self):
#         """Test red_filter with an RGB image."""
#         np_img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[255, 255, 255], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
#         pil_img = Image.fromarray(np_img)
#         mask = red_filter(pil_img, 128, 128, 128)
#         expected_mask = np.array([[False, True, True], [False, True, False]], dtype=bool)
#         self.assertTrue(np.array_equal(mask, expected_mask))

#     def test_red_filter_with_non_rgb_image(self):
#         """Test red_filter with a non-RGB image."""
#         np_img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
#         pil_img = Image.fromarray(np_img)
#         with self.assertRaises(ValueError):
#             red_filter(pil_img, 128, 128, 128)

#     def test_red_filter_with_invalid_thresholds(self):
#         """Test red_filter with invalid thresholds."""
#         np_img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[255, 255, 255], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
#         pil_img = Image.fromarray(np_img)
#         with self.assertRaises(ValueError):
#             red_filter(pil_img, -1, 128, 128)
#         with self.assertRaises(ValueError):
#             red_filter(pil_img, 128, 256, 128)
#         with self.assertRaises(ValueError):
#             red_filter(pil_img, 128, 128, -1)

# if __name__ == "__main__":
#     unittest.main()
