import json
import unittest

import numpy as np
import yaml
from numpy.testing import assert_array_equal

from pathopatcher.cli import (
    PreProcessingConfig,
    PreProcessingYamlConfig,
)
from pathopatcher.utils.patch_util import NormalizeParameters
from test_database.download import check_test_database


class TestMacenkoLoading(unittest.TestCase):
    """Check if the Normalization File can be loaded correctly"""

    @classmethod
    def setUpClass(cls) -> None:
        check_test_database()
        cls.config = "./tests/static_test_files/preprocessing/macenko/load_macenko.yaml"
        with open(cls.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config = PreProcessingYamlConfig(**yaml_config)

        opt_dict = dict(yaml_config)
        cls.opt_dict = {k: v for k, v in opt_dict.items() if v is not None}
        cls.configuration = PreProcessingConfig(**cls.opt_dict)

    def test_loadMacenko(self) -> None:
        """Check if the Normalization File is loaded correctly"""
        normalization_vector_patch = NormalizeParameters(
            normalization_vector_path=self.configuration.normalization_vector_json
        )

        # load saved file
        with open(
            "./tests/static_test_files/preprocessing/macenko/result/test_macenko.json"
        ) as correct_file:
            correc_values = json.load(correct_file)

        correc_values["stain_vectors"] = np.array(correc_values["stain_vectors"])
        correc_values["max_sat"] = np.array(correc_values["max_sat"])
        stain_vectors = normalization_vector_patch.get_he_ref()
        max_sat = normalization_vector_patch.get_max_sat()

        assert_array_equal(correc_values["stain_vectors"], stain_vectors)
        assert_array_equal(correc_values["max_sat"], max_sat)


if __name__ == "__main__":
    unittest.main()
