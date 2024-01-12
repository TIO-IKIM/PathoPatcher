import unittest
import numpy as np
from pathopatch.utils.masking import mask_rgb


class TestMaskRGB(unittest.TestCase):
    def setUp(self):
        """Set up RGB and mask arrays for testing."""
        self.rgb = np.array(
            [[[255, 255, 255], [0, 0, 0]], [[255, 255, 255], [0, 0, 0]]]
        )
        self.mask = np.array([[1, 0], [0, 1]])

    def test_mask_rgb_valid(self):
        """Test mask_rgb with valid inputs."""
        expected_output = np.array(
            [[[255, 255, 255], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
        )
        actual_output = mask_rgb(self.rgb, self.mask)
        np.testing.assert_array_equal(actual_output, expected_output)

    def test_mask_rgb_invalid_shapes(self):
        """Test mask_rgb with mask and RGB arrays of different shapes."""
        invalid_mask = np.array([[1, 0]])
        with self.assertRaises(AssertionError):
            mask_rgb(self.rgb, invalid_mask)

    # def test_mask_rgb_invalid_values(self):
    #     """Test mask_rgb with mask array containing values other than 0 and 1."""
    #     invalid_mask = np.array([[1, 2], [3, 4]])
    #     with self.assertRaises(ValueError):
    #         mask_rgb(self.rgb, invalid_mask)


if __name__ == "__main__":
    unittest.main()
