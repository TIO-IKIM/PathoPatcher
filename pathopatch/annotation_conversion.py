#!/usr/bin/env python

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
from shapely.geometry import shape, mapping
import uuid


def merge_outlines(geojson_string: str) -> str:
    """Read a GeoJSON file, extract outlines, and add a new feature representing the merged outlines.

    Args:
        geojson_path (str): String representation of the geojson
    Raises:
        NotImplementedError: Error if no FeatureCollection is imported

    Returns:
        str: Geojson string with outline (Named: Tissue-Outline)
    """
    json_element = json.loads(geojson_string)

    merged_outlines = None
    individual_outlines = []

    for element in json_element:
        if element["type"] == "FeatureCollection":
            for feature in element["features"]:
                geometry = shape(feature["geometry"])
                individual_outlines.append(geometry)
                if merged_outlines is None:
                    merged_outlines = geometry
                else:
                    merged_outlines = merged_outlines.union(geometry.buffer(0))

            merged_feature = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(merged_outlines),
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": "Tissue-Outline", "color": [0, 0, 0]},
                },
            }
            json_element[0]["features"].append(merged_feature)

            # Save the modified GeoJSON file
            modified_geojson_str = json.dumps(json_element, indent=2)
            return modified_geojson_str

        elif element["type"] == "Feature":
            print("Deteceted single feature, not a FeatureCollection")
            raise NotImplementedError


def convert_geojson_to_json(
    file_path: str, output_path: str, generate_tissue_outline: bool = False
) -> None:
    """Convert a GeoJSON file to a JSON file.

    Args:
        file_path (str): Path to the input GeoJSON file.
        output_path (str): Path to the output JSON file.
        generate_tissue_outline (bool): Flag to indicate whether to generate tissue outline. Defaults to False.
    """
    with open(file_path, "r") as file:
        geojson_string = file.read()
        if geojson_string[0] != "[" and geojson_string[-1] != "]":
            geojson_string = f"[{geojson_string}]"

    if generate_tissue_outline:
        geojson_string = merge_outlines(geojson_string)
    json_element = json.loads(geojson_string)
    all_elements = []
    for element in json_element:
        if element["type"] == "FeatureCollection":
            for feat in element["features"]:
                all_elements.append(feat)
        elif element["type"] == "Feature":
            all_elements.append(element)
    # Save the JSON file with indentation
    with open(output_path, "w") as file:
        json.dump(all_elements, file, indent=2)


def convert_folder_geojson_to_json(
    input_folder: str, output_folder: str, generate_tissue_outline: bool = False
) -> None:
    """Convert all GeoJSON files in a folder to JSON files.

    Args:
        input_folder (str): Path to the input folder containing GeoJSON files.
        output_folder (str): Path to the output folder for JSON files.
        generate_tissue_outline (bool): Flag to indicate whether to generate tissue outline. Defaults to False.
    """
    filelist = [
        f
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]
    for filename in tqdm(filelist, total=len(filelist)):
        if filename.endswith(".geojson"):
            print(f"Converting {filename}")
            try:
                convert_geojson_to_json(
                    file_path=os.path.join(input_folder, filename),
                    output_path=os.path.join(output_folder, filename[:-8] + ".json"),
                    generate_tissue_outline=generate_tissue_outline,
                )
            except Exception as e:
                print(f"Failed: {filename} - {str(e)}")


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

    convert_folder_geojson_to_json(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        generate_tissue_outline=args.generate_tissue_outline,
    )


if __name__ == "__main__":
    main()
