import json
import os
import shutil
import unittest
from copy import copy
from pathlib import Path

import numpy as np
import yaml
from numpy.testing import assert_allclose

from pathopatcher.cli import MacenkoConfig, MacenkoParser, MacenkoYamlConfig
from pathopatcher.patch_extraction.patch_extraction import PreProcessor
from test_database.download import check_test_database


class TestMacenkoSaving(unittest.TestCase):
    """Test if the Macenko-Normalization Vectors are stored in the right way"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup: Load configuration"""
        check_test_database()
        cls.parser = MacenkoParser()
        cls.config = "./tests/static_test_files/preprocessing/macenko/test_macenko.yaml"

        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = MacenkoYamlConfig(**yaml_config)

        opt_dict = dict(yaml_config)
        cls.opt_dict = {k: v for k, v in opt_dict.items() if v is not None}
        cls.opt_dict["wsi_paths"] = copy(opt_dict["wsi_path"])
        cls.opt_dict.pop("wsi_path")

        # overwrite hard coded options
        for k, v in cls.parser.default_dict.items():
            cls.opt_dict[k] = v

        assert (
            Path(cls.opt_dict["save_json_path"]).suffix == ".json"
        ), "Output path must be a .json file"

        cls.opt_dict["output_path"] = str(Path(cls.opt_dict["save_json_path"]).parent)

        cls.configuration = MacenkoConfig(**cls.opt_dict)

    @classmethod
    def tearDownClass(cls):
        """Clean output directory"""
        # clean output directory
        clean_files = [f for f in Path(cls.opt_dict["output_path"]).iterdir()]
        for f in clean_files:
            os.remove(f.resolve())
        shutil.rmtree(f.parent.resolve())

    def test_macenko_saving(self) -> None:
        """Test if macenko normalization is performed correctly and file saved at the right place"""
        slide_processor = PreProcessor(slide_processor_config=self.configuration)
        slide_processor.save_normalization_vector(
            wsi_file=self.configuration.wsi_paths,
            save_json_path=self.configuration.save_json_path,
        )

        output_file_dir = Path(self.configuration.save_json_path).resolve()

        self.assertTrue(
            output_file_dir.is_file() and output_file_dir.name == "test_macenko.json"
        )

    def test_macenko_values(self) -> None:
        """Test if macenko normalization is performed correctly and right values calculated"""
        slide_processor = PreProcessor(slide_processor_config=self.configuration)
        slide_processor.save_normalization_vector(
            wsi_file=self.configuration.wsi_paths,
            save_json_path=self.configuration.save_json_path,
        )

        output_file_dir = Path(self.configuration.save_json_path).resolve()

        # load saved file
        with open(
            "./tests/static_test_files/preprocessing/macenko/result/test_macenko.json"
        ) as correct_file:
            correc_values = json.load(correct_file)

        with open(str(output_file_dir)) as calc_file:
            calculated_values = json.load(calc_file)

        correc_values["stain_vectors"] = np.array(correc_values["stain_vectors"])
        correc_values["max_sat"] = np.array(correc_values["max_sat"])
        calculated_values["stain_vectors"] = np.array(
            calculated_values["stain_vectors"]
        )
        calculated_values["max_sat"] = np.array(calculated_values["max_sat"])

        assert_allclose(
            correc_values["stain_vectors"], calculated_values["stain_vectors"]
        )
        assert_allclose(correc_values["max_sat"], calculated_values["max_sat"])


if __name__ == "__main__":
    unittest.main()
