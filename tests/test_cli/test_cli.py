import unittest
import yaml

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pathopatch.cli import (
    PreProcessingConfig,
    PreProcessingYamlConfig,
)


class TestPreProcessConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Setup configuration"""
        cls.config = "./tests/static_test_files/preprocessing/cli.yaml"
        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = PreProcessingYamlConfig(**yaml_config)

        cls.opt_dict = dict(yaml_config)

    def test_negative_patches(self) -> None:
        test_opt_dict = self.opt_dict
        test_opt_dict["patch_overlap"] = -10
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)

    def test_negative_overlap(self) -> None:
        test_opt_dict = self.opt_dict
        test_opt_dict["patch_overlap"] = -10
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)

    def test_invalid_overlap(self) -> None:
        test_opt_dict = self.opt_dict
        test_opt_dict["patch_size"] = 101
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)

    def test_negative_processes(self) -> None:
        test_opt_dict = self.opt_dict
        self.opt_dict["processes"] = -10
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)

    def test_background_ratio_range(self) -> None:
        test_opt_dict = self.opt_dict
        test_opt_dict["max_background_ratio"] = 2.0
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)
        test_opt_dict["max_background_ratio"] = -2.0
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)

    def test_wrong_annotation_extension(self) -> None:
        test_opt_dict = self.opt_dict
        test_opt_dict["annotation_extension"] = -10
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)

    def test_wrong_log_level(self) -> None:
        test_opt_dict = self.opt_dict
        test_opt_dict["log_level"] = -10
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)

    def test_wrong_wsi_extension(self) -> None:
        test_opt_dict = self.opt_dict
        test_opt_dict["wsi_extension"] = "stupid"
        with self.assertRaises(ValueError):
            PreProcessingConfig(**test_opt_dict)


if __name__ == "__main__":
    unittest.main()
