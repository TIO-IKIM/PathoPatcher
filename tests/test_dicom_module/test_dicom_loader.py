import json
import shutil
import subprocess
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


class TestPreProcessorDICOM(unittest.TestCase):
    """Test the dicom image loader, must be equal to openslide loader"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup configuration"""
        check_test_database()
        conversion_command = "wsidicomizer -i ./test_database/input/WSI/CMU-1.svs -o ./test_database/dicom_files/CMU-1"
        # dicom conversion
        process = subprocess.Popen(conversion_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # openslide
        cls.wsi_name = "CMU-1"
        cls.config = "./tests/static_test_files/preprocessing/dicom/openslide.yaml"
        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = PreProcessingYamlConfig(**yaml_config)

        opt_dict = dict(yaml_config)
        cls.opt_dict = {k: v for k, v in opt_dict.items() if v is not None}
        cls.configuration = PreProcessingConfig(**cls.opt_dict)
        cls.openslide_config = cls.configuration.dict()
        preprocess_logger = Logger(
            level=cls.configuration.log_level.upper(),
            log_dir=cls.configuration.log_path,
            comment="preprocessing",
            use_timestamp=True,
        )
        cls.logger = preprocess_logger.create_logger()
        # do preprocessing for openslide
        cls.slide_processor = PreProcessor(slide_processor_config=cls.configuration)
        cls.logger.info(
            "Sucessfully started the setup - Now we calculate the base dataset. May take up to 10 Minutes!"
        )
        cls.slide_processor.sample_patches_dataset()

        # dicom
        cls.config = "./tests/static_test_files/preprocessing/dicom/dicom.yaml"
        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = PreProcessingYamlConfig(**yaml_config)

        opt_dict = dict(yaml_config)
        cls.opt_dict = {k: v for k, v in opt_dict.items() if v is not None}
        cls.configuration = PreProcessingConfig(**cls.opt_dict)
        cls.dicom_config = cls.configuration.dict()
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
        shutil.rmtree(cls.dicom_config["output_path"].parent)
        shutil.rmtree(Path("./test_database/dicom_files/CMU-1").resolve())

    def test_metadata_wsi(self) -> None:
        os_path = self.openslide_config["output_path"] / self.wsi_name / "metadata.yaml"
        with open(os_path, "r") as config_file:
            os_config = yaml.safe_load(config_file)

        dcm_path = self.dicom_config["output_path"] / self.wsi_name / "metadata.yaml"
        with open(dcm_path, "r") as config_file:
            dcm_file = yaml.safe_load(config_file)

        self.assertEqual(os_config, dcm_file)

    def test_count_patches(self) -> None:
        """Test if the number of patches is correct"""
        os_path = self.openslide_config["output_path"] / self.wsi_name / "metadata"
        os_patches_count = len([f for f in os_path.glob("*.yaml")])

        dcm_path = self.dicom_config["output_path"] / self.wsi_name / "metadata"
        dcm_patches_count = len([f for f in dcm_path.glob("*.yaml")])

        self.assertEqual(os_patches_count, dcm_patches_count)

    def test_patch_results_wsi(self) -> None:
        """Test if patches are extracted the right way for WSI"""
        os_path = (
            self.openslide_config["output_path"] / self.wsi_name / "patch_metadata.json"
        )
        with open(os_path, "r") as config_file:
            patch_os = json.load(config_file)
            patch_os = sorted(patch_os, key=lambda d: list(d.keys())[0])
        patch_os = {list(elem.keys())[0]: list(elem.values())[0] for elem in patch_os}

        dcm_path = (
            self.dicom_config["output_path"] / self.wsi_name / "patch_metadata.json"
        )
        with open(dcm_path, "r") as config_file:
            test_dcm = json.load(config_file)
            test_dcm = sorted(test_dcm, key=lambda d: list(d.keys())[0])

        test_dcm = {list(elem.keys())[0]: list(elem.values())[0] for elem in test_dcm}

        # Extract unique patch names from both files
        unique_patches = set(patch_os.keys()).union(test_dcm.keys())
        # print(unique_patches)
        differing_patches = 0
        for patch_name in unique_patches:
            if patch_name not in patch_os or patch_name not in test_dcm:
                differing_patches += 1
                print(f"differing_patches: {patch_name}")
            else:
                if not patch_os[patch_name] == test_dcm[patch_name]:
                    differing_patches += 1
                    print(f"differing_patches: {patch_name}")

        self.assertLess(
            differing_patches,
            15,
            "Patches are not equal and differ in more than 15 patches",
        )

    def test_example_images(self) -> None:
        """ """
        patch_list = [
            "CMU-1_8_116.png",
            "CMU-1_10_109.png",
            "CMU-1_28_152.png",
            "CMU-1_102_16.png",
        ]
        os_path = self.openslide_config["output_path"] / self.wsi_name / "patches"
        dcm_path = self.dicom_config["output_path"] / self.wsi_name / "patches"

        for p_name in patch_list:
            os_image = np.array(Image.open((os_path / p_name).resolve()))
            dcm_image = np.array(Image.open((dcm_path / p_name).resolve()))

            assert_almost_equal(os_image, dcm_image)
