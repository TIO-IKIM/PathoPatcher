import shutil
import unittest

import numpy as np
import yaml
from numpy.testing import assert_almost_equal
from PIL import Image
from tqdm import tqdm

from pathopatch.cli import PreProcessingConfig, PreProcessingYamlConfig
from pathopatch.patch_extraction.dataset import (
    LivePatchWSIConfig,
    LivePatchWSIDataloader,
    LivePatchWSIDataset,
)
from pathopatch.patch_extraction.patch_extraction import PreProcessor
from pathopatch.utils.logger import Logger
from pathopatch.utils.tools import close_logger
from test_database.download import check_test_database


class TestPreProcessorComplexDataset(unittest.TestCase):
    """Compare disk dataset with in memory dataset for a complex setup"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup configuration"""
        check_test_database()
        cls.config = (
            "./tests/static_test_files/preprocessing/complex_setup_dataset/config.yaml"
        )
        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = PreProcessingYamlConfig(**yaml_config)

        opt_dict = dict(yaml_config)
        cls.opt_dict = {k: v for k, v in opt_dict.items() if v is not None}
        cls.configuration = PreProcessingConfig(**cls.opt_dict)

        cls.gt_folder = (
            cls.configuration.log_path / cls.configuration.wsi_paths.stem
        ).resolve()
        cls.wsi_name = "CMU-1-Small-Region"

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
        cls.logger.info("Starting to load in-memory dataset...")
        cls.outdir_pt = cls.configuration.log_path / "in_memory_dataset"
        cls.outdir_pt.mkdir(parents=True, exist_ok=True)
        (cls.outdir_pt / "patches").mkdir(parents=True, exist_ok=True)
        (cls.outdir_pt / "metadata").mkdir(parents=True, exist_ok=True)
        cls.patch_config = LivePatchWSIConfig(
            wsi_path=str(cls.configuration.wsi_paths),
            target_mpp=cls.opt_dict["target_mpp"],
            patch_size=cls.opt_dict["patch_size"],
            patch_overlap=cls.opt_dict["patch_overlap"],
        )
        cls.patch_dataset = LivePatchWSIDataset(
            cls.patch_config, cls.logger, transforms=None
        )
        cls.patch_dataloader = LivePatchWSIDataloader(cls.patch_dataset, 1)
        for batch in tqdm(cls.patch_dataloader, total=len(cls.patch_dataloader)):
            image_tensor = batch[0]
            metadata = batch[1][0]
            patch_name = f"{cls.configuration.wsi_paths.stem}_{metadata['row']}_{metadata['col']}.png"
            image_pil = Image.fromarray(image_tensor[0, ...].numpy().astype(np.uint8))
            image_pil.save(cls.outdir_pt / "patches" / patch_name)
            metadata.pop("discard_patch")
            with open(
                cls.outdir_pt / "metadata" / f"{patch_name.replace('.png', '.yaml')}",
                "w",
            ) as f:
                yaml.dump(metadata, f, sort_keys=False)
        cls.logger.info("Finished loading in-memory dataset")

    @classmethod
    def tearDownClass(cls):
        """Clean output directory"""
        # close logger
        close_logger(cls.logger)
        shutil.rmtree(cls.gt_folder.parent.resolve())

    def test_count_patches_metadata(self) -> None:
        """Test if the number of extracted metadata is correct"""
        gt_path = self.gt_folder / "metadata"
        gt_patches_count = len([f for f in gt_path.glob("*.yaml")])

        test_path = self.outdir_pt / "metadata"
        test_patches_count = len([f for f in test_path.glob("*.yaml")])

        self.assertEqual(gt_patches_count, test_patches_count)

    def test_count_patches(self) -> None:
        """Test if the number of patches is correct"""
        gt_path = self.gt_folder / "patches"
        gt_patches_count = len([f for f in gt_path.glob("*.png")])

        test_path = self.outdir_pt / "patches"
        test_patches_count = len([f for f in test_path.glob("*.png")])

        self.assertEqual(gt_patches_count, test_patches_count)

    def test_patches_behaviour(self) -> None:
        """Check if the patches are the same"""
        gt_path = self.gt_folder / "patches"
        test_path = self.outdir_pt / "patches"
        for gt_patch in gt_path.glob("*.png"):
            test_patch = test_path / gt_patch.name
            gt_image = np.array(Image.open(gt_patch.resolve()))
            test_image = np.array(Image.open(test_patch.resolve()))
            assert_almost_equal(gt_image, test_image)

    def test_metadata_content(self) -> None:
        """Check if the metadata is the same"""
        gt_path = self.gt_folder / "metadata"
        test_path = self.outdir_pt / "metadata"
        for gt_patch in gt_path.glob("*.yaml"):
            test_patch = test_path / gt_patch.name
            with open(gt_patch, "r") as f:
                gt_patch_config = yaml.safe_load(f)
            with open(test_patch, "r") as f:
                test_patch_config = yaml.safe_load(f)
            gt_patch_config.pop("metadata_path")
            self.assertDictEqual(gt_patch_config, test_patch_config)
