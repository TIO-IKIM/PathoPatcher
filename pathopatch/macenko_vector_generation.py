#!/usr/bin/env python

# -*- coding: utf-8 -*-
# Main entry point for Macenko
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import logging
import os
import sys

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from pathopatch.cli import MacenkoParser
from pathopatch.patch_extraction.patch_extraction import PreProcessor


def main():
    configuration_parser = MacenkoParser()
    configuration, logger = configuration_parser.get_config()

    slide_processor = PreProcessor(slide_processor_config=configuration)
    slide_processor.save_normalization_vector(
        wsi_file=configuration.wsi_paths, save_json_path=configuration.save_json_path
    )

    logger.info("Finished Macenko Vector Calculation!")


if __name__ == "__main__":
    main()
