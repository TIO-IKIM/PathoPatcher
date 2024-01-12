import json
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import yaml
from numpy.testing import assert_almost_equal
from PIL import Image

from pathopatch.cli import PreProcessingConfig, PreProcessingYamlConfig
from pathopatch.patch_extraction.patch_extraction import PreProcessor
from pathopatch.utils.logger import Logger
from pathopatch.utils.tools import close_logger
from test_database.download import check_test_database


class TestPreProcessorBaseline(unittest.TestCase):
    """Test the PreProcessor Module with basic (default) parameter setup"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup configuration"""
        check_test_database()
        cls.config = (
            "./tests/static_test_files/preprocessing/annotations_filtering/config.yaml"
        )
        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = PreProcessingYamlConfig(**yaml_config)

        opt_dict = dict(yaml_config)
        cls.opt_dict = {k: v for k, v in opt_dict.items() if v is not None}
        cls.configuration = PreProcessingConfig(**cls.opt_dict)

        cls.gt_folder = Path(
            "./tests/static_test_files/preprocessing/annotations_filtering/results/"
        ).resolve()
        cls.wsi_name1 = "CMU-1"
        cls.wsi_name2 = "JP2K-33003-1"

        preprocess_logger = Logger(
            level=cls.configuration.log_level.upper(),
            log_dir=cls.configuration.log_path,
            comment="preprocessing",
            use_timestamp=True,
        )
        cls.logger = preprocess_logger.create_logger()
        # do preprocessing for result checking
        cls.slide_processor = PreProcessor(slide_processor_config=cls.configuration)
        cls.logger.info(
            "Sucessfully started the setup - Now we calculate the base dataset. May take up to 10 Minutes!"
        )
        cls.slide_processor.sample_patches_dataset()

    @classmethod
    def tearDownClass(cls):
        """Clean output directory"""
        # close logger
        close_logger(cls.logger)

        # clean output directory
        clean_folders = [
            f for f in Path(cls.opt_dict["output_path"]).iterdir() if f.is_dir()
        ]
        for f in clean_folders:
            shutil.rmtree(f.resolve())
        clean_files = [
            f for f in Path(cls.opt_dict["output_path"]).iterdir() if f.is_file()
        ]
        for f in clean_files:
            os.remove(f.resolve())
        shutil.rmtree(f.parent.resolve())

    def test_init_files(self) -> None:
        """For this case 1 WSI files should have been loaded"""
        self.assertEqual(self.slide_processor.num_files, 3)

    def test_init_num_annotations_loaded(self) -> None:
        """For this case 0 annotation files should have been loaded"""
        self.assertEqual(len(self.slide_processor.annotation_files), 2)

    def test_no_mask_folder(self) -> None:
        """Test that no mask folder has been created"""
        mask_folder = self.slide_processor.config.output_path / self.wsi_name1 / "masks"
        self.assertFalse(mask_folder.exists())

    def test_filterer_annotations(self) -> None:
        """Test if annotations have been filtered correctly and no cyst is inside the masks folder"""
        test_path = (
            self.slide_processor.config.output_path
            / self.wsi_name1
            / "annotation_masks"
        )
        test_files = [f for f in test_path.glob("*cyst*")]
        self.assertEqual(len(test_files), 0)

    def test_mask_overlaid_image(self) -> None:
        """Test if the mask overlaid image is correct and no cyst is displayed"""
        gt_path = (
            self.gt_folder
            / self.wsi_name1
            / "annotation_masks"
            / "all_overlaid_clean.png"
        )
        gt_image = np.array(Image.open(gt_path.resolve()))

        test_path = (
            self.slide_processor.config.output_path
            / self.wsi_name1
            / "annotation_masks"
            / "all_overlaid_clean.png"
        )
        test_image = np.array(Image.open(test_path.resolve()))
        assert_almost_equal(test_image, gt_image)

    def test_metadata_wsi(self) -> None:
        """Test if metadata is correct for WSI"""
        gt_path = self.gt_folder / self.wsi_name1 / "metadata.yaml"
        with open(gt_path, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)

        test_path = (
            self.slide_processor.config.output_path / self.wsi_name1 / "metadata.yaml"
        )
        with open(test_path, "r") as config_file:
            test_file = yaml.safe_load(config_file)

        self.assertEqual(yaml_config, test_file)

    def test_patch_results_wsi(self) -> None:
        """Test if patches are extracted the right way for WSI"""
        gt_path = self.gt_folder / self.wsi_name1 / "patch_metadata.json"
        with open(gt_path, "r") as config_file:
            patch_gt = json.load(config_file)
            patch_gt = sorted(patch_gt, key=lambda d: list(d.keys())[0])

        test_path = (
            self.slide_processor.config.output_path
            / self.wsi_name1
            / "patch_metadata.json"
        )
        with open(test_path, "r") as config_file:
            test_file = json.load(config_file)
            test_file = sorted(test_file, key=lambda d: list(d.keys())[0])

        self.assertEqual(patch_gt, test_file)
