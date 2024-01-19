#!/usr/bin/env python

# -*- coding: utf-8 -*-
# Main entry point for patch-preprocessing
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)

import logging

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

from pathopatch.cli import PreProcessingParser
from pathopatch.patch_extraction.patch_extraction import PreProcessor
from pathopatch.utils.tools import close_logger


def main():
    configuration_parser = PreProcessingParser()
    configuration, logger = configuration_parser.get_config()
    configuration_parser.store_config()

    slide_processor = PreProcessor(slide_processor_config=configuration)
    slide_processor.sample_patches_dataset()

    logger.info("Finished Preprocessing.")
    close_logger(logger)


if __name__ == "__main__":
    main()
