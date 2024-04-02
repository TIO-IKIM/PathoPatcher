# -*- coding: utf-8 -*-
# Wrapping Openslide for a common interface
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from openslide.deepzoom import DeepZoomGenerator


# ignore kwargs for OpenSlide DeepZoomGenerator
class DeepZoomGeneratorOS(DeepZoomGenerator):
    def __init__(
        self, image_loader, tile_size=254, overlap=1, limit_bounds=False, **kwargs
    ):
        """Overwrite DeepZoomGenerator of OpenSlide

            DeepZoomGenerator gets overwritten to provide matching API with CuCim
            No Change in functionality

        Args:
            image_loader (OpenSlide): OpenSlide Image. Needed for OS compatibility and for retrieving metadata.
            tile_size (int, optional): the width and height of a single tile.  For best viewer
                          performance, tile_size + 2 * overlap should be a power
                          of two.. Defaults to 254.
            overlap (int, optional): the number of extra pixels to add to each interior edge
                          of a tile. Defaults to 1.
            limit_bounds (bool, optional): True to render only the non-empty slide region. Defaults to False.
        """
        super().__init__(image_loader, tile_size, overlap, limit_bounds)
