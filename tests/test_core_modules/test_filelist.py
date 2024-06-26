import os
import shutil
import unittest
from pathlib import Path

import yaml

from pathopatch.cli import PreProcessingConfig, PreProcessingYamlConfig
from pathopatch.patch_extraction.patch_extraction import PreProcessor
from pathopatch.utils.logger import Logger
from pathopatch.utils.tools import close_logger
from test_database.download import check_test_database


class TestPreProcessorFilelist(unittest.TestCase):
    """Test the PreProcessor Module with basic (default) parameter setup, but with filelist input"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup configuration"""
        check_test_database()
        cls.config = "./tests/static_test_files/preprocessing/filelist/config.yaml"
        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = PreProcessingYamlConfig(**yaml_config)

        opt_dict = dict(yaml_config)
        cls.opt_dict = {k: v for k, v in opt_dict.items() if v is not None}
        cls.configuration = PreProcessingConfig(**cls.opt_dict)

        cls.gt_folder = Path(
            "./tests/static_test_files/preprocessing/filelist/results/"
        ).resolve()
        cls.wsi_name = "CMU-1"

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
        self.assertEqual(self.slide_processor.num_files, 1)

    def test_init_num_annotations_loaded(self) -> None:
        """For this case 0 annotation files should have been loaded"""
        self.assertEqual(len(self.slide_processor.annotation_files), 0)

    def test_metadata_wsi(self) -> None:
        gt_path = self.gt_folder / self.wsi_name / "metadata.yaml"
        with open(gt_path, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)

        test_path = (
            self.slide_processor.config.output_path / self.wsi_name / "metadata.yaml"
        )
        with open(test_path, "r") as config_file:
            test_file = yaml.safe_load(config_file)

        self.assertEqual(yaml_config, test_file)
