# -*- coding: utf-8 -*-
# Geojson annotation preprocessing
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os
import json
from tqdm import tqdm
import argparse
import uuid

def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoJSON annotations (from QuPath) to JSON format."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the input folder containing GeoJSON files.",
    )
    parser.add_argument(
        "--output_folder", type=str, help="Path to the output folder for JSON files."
    )
    parser.add_argument(
        "--generate_tissue_outline",
        action="store_true",
        help="Generate tissue outline.",
    )
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
