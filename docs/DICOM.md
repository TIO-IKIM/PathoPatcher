# Convert WSI-Files to DICOM

## Basic cli-usage

```bash
usage: wsidicomizer [-h] -i INPUT [-o OUTPUT] [-t TILE_SIZE] [-m METADATA]
                    [-d DEFAULT_METADATA] [-l LEVELS [LEVELS ...]] [--label LABEL]
                    [--no-label] [--no-overview] [--no-confidential] [-w WORKERS]
                    [--chunk-size CHUNK_SIZE] [--format FORMAT] [--quality QUALITY]
                    [--subsampling SUBSAMPLING] [--offset-table OFFSET_TABLE]

Convert compatible wsi file to DICOM

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input wsi file.
  -o OUTPUT, --output OUTPUT
                        Path to output folder. Folder will be created and must not
                        exist. If not specified a folder named after the input file is
                        created in the same path.
  -t TILE_SIZE, --tile-size TILE_SIZE
                        Tile size (same for width and height). Required for ndpi and
                        openslide formats E.g. 512
  -m METADATA, --metadata METADATA
                        Path to json metadata that will override metadata from source
                        image file.
  -d DEFAULT_METADATA, --default-metadata DEFAULT_METADATA
                        Path to json metadata that will be used as default values.
  -l LEVELS [LEVELS ...], --levels LEVELS [LEVELS ...]
                        Pyramid levels to include, if not all. E.g. 0 1 for base and
                        first pyramid layer.
  --label LABEL         Optional label image to use instead of label found in file.
  --no-label            If not to include label
  --no-overview         If not to include overview
  --no-confidential     If not to include confidential metadata
  -w WORKERS, --workers WORKERS
                        Number of worker threads to use
  --chunk-size CHUNK_SIZE
                        Number of tiles to give each worker at a time
  --format FORMAT       Encoding format to use if re-encoding. 'jpeg' or 'jpeg2000'.
  --quality QUALITY     Quality to use if re-encoding. It is recommended to not use >
                        95 for jpeg. Use < 1 or > 1000 for lossless jpeg2000.
  --subsampling SUBSAMPLING
                        Subsampling option if using jpeg for re-encoding. Use '444'
                        for no subsampling, '422' for 2x1 subsampling, and '420' for
                        2x2 subsampling.
  --offset-table OFFSET_TABLE
                        Offset table to use, 'bot' basic offset table, 'eot' extended
                        offset table, 'None' - no offset table.
```

## Acknowledgement for using WSIDICOMIZER

wsidicomizer: Copyright 2021 Sectra AB, licensed under Apache 2.0.

This project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. IMI website: <www.imi.europa.eu>
