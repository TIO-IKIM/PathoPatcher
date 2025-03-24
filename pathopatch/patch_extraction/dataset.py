# -*- coding: utf-8 -*-
# Main Patch Extraction Class for a WSI/Dataset, in memory extraction without storing
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import logging
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from openslide import OpenSlide
from pydantic import BaseModel, validator
from shapely.affinity import scale
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToTensor
from PIL import Image
from pathopatch.utils.exceptions import WrongParameterException
from pathopatch.wsi_interfaces.openslide_deepzoom import DeepZoomGeneratorOS
from pathopatch.wsi_interfaces.wsidicomizer_openslide import (
    DicomSlide,
    DeepZoomGeneratorDicom,
)
from pathopatch.utils.patch_util import (
    calculate_background_ratio,
    compute_interesting_patches,
    get_intersected_labels,
    get_regions_json,
    macenko_normalization,
    pad_tile,
    patch_to_tile_size,
    target_mag_to_downsample,
    target_mpp_to_downsample,
)
from pathopatch.utils.tools import module_exists

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class LivePatchWSIConfig(BaseModel):
    """Storing the configuration for the PatchWSIDataset

    Args:
        wsi_path (str): Path to the WSI
        wsi_properties (dict, optional): Dictionary with manual WSI metadata, but just applies if metadata cannot be derived from OpenSlide (e.g., for .tiff files). Supported keys are slide_mpp and magnification
        patch_size (int, optional): The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px. Defaults to 256.
        patch_overlap (float, optional): The percentage amount pixels that should overlap between two different patches.
            Please Provide as integer between 0 and 100, indicating overlap in percentage.
            Defaults to 0.
        target_mpp (float, optional): If this parameter is provided, the output level of the WSI
            corresponds to the level that is at the target microns per pixel of the WSI.
            Alternative to target_mag, downsaple and level. Highest priority, overwrites all other setups for magnifcation, downsample, or level.
        target_mag (float, optional): If this parameter is provided, the output level of the WSI
            corresponds to the level that is at the target magnification of the WSI.
            Alternative to target_mpp, downsaple and level. High priority, just target_mpp has a higher priority, overwrites downsample and level if provided. Defaults to None.
        downsample (int, optional): Each WSI level is downsampled by a factor of 2, downsample
            expresses which kind of downsampling should be used with
            respect to the highest possible resolution. Defaults to 0.
        level (int, optional): The tile level for sampling, alternative to downsample. Defaults to None.
        target_mpp_tolerance(float, optional): Tolerance for the target_mpp. If wsi mpp is within a range target_mpp +/- tolarance, no rescaling is performed. Defaults to 0.0.
        annotation_path (str, optional): Path to the .json file with the annotations. Defaults to None.
        label_map_file (str, optional): Path to the .json file with the label map. Defaults to None.
        label_map (dict, optional): Dictionary with the label map. Defaults to None.
        exclude_classes (List[str], optional): List of classes to exclude from the annotation. Defaults to [].
        overlapping_labels (bool, optional): If True, overlapping labels are allowed. Defaults to False.
        store_masks (bool, optional): If True, masks are stored. Defaults to False.
        normalize_stains (bool, optional): If True, stains are normalized. Defaults to False.
        normalization_vector_json (str, optional): Path to the .json file with the normalization vector. Defaults to None.
        min_intersection_ratio (float, optional): The minimum intersection between the tissue mask and the patch.
            Must be between 0 and 1. 0 means that all patches are extracted. Defaults to 0.01.
        tissue_annotation (str, optional): Can be used to name a polygon annotation to determine the tissue area
            If a tissue annotation is provided, no Otsu-thresholding is performed. Defaults to None.
        tissue_annotation_intersection_ratio (float, optional): Intersection ratio with tissue annotation. Helpful, if ROI annotation is passed, which should not interfere with background ratio.
            If not provided, the default min_intersection_ratio with the background is used. Defaults to None.
        masked_otsu (bool, optional): Use annotation to mask the thumbnail before otsu-thresholding is used. Defaults to False.
        otsu_annotation (bool, optional): Can be used to name a polygon annotation to determine the area
            for masked otsu thresholding. Seperate multiple labels with ' ' (whitespace). Defaults to None.
        apply_prefilter (bool, optional): Pre-extraction mask filtering to remove marker from mask before applying otsu. Defaults to False.
        filter_patches (bool, optional): Post-extraction patch filtering to sort out artefacts, marker and other non-tissue patches with a DL model. Time consuming.
            Defaults to False.
    """

    # path
    wsi_path: str
    wsi_properties: Optional[dict]

    # basic setup
    patch_size: int
    patch_overlap: float = 0.0
    downsample: Optional[int] = 1
    target_mpp: Optional[float]
    target_mag: Optional[float]
    level: Optional[int]
    target_mpp_tolerance: Optional[float] = 0.0

    # annotation specific settings
    annotation_path: Optional[str]
    label_map_file: Optional[str]
    label_map: Optional[dict]
    exclude_classes: Optional[List[str]] = []
    overlapping_labels: Optional[bool] = False
    store_masks: Optional[bool] = False

    # macenko stain normalization
    normalize_stains: Optional[bool] = False
    normalization_vector_json: Optional[str]

    # finding patches
    min_intersection_ratio: Optional[float] = 0.01
    tissue_annotation: Optional[str]
    tissue_annotation_intersection_ratio: Optional[float]
    otsu_annotation: Optional[str]
    masked_otsu: Optional[bool] = False
    apply_prefilter: Optional[bool] = False
    filter_patches: Optional[bool] = False

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        __pydantic_self__.__post_init_post_parse__()

    # validators
    @validator("patch_size")
    def patch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Patch-Size in pixels must be positive")
        return v

    @validator("patch_overlap")
    def overlap_percentage(cls, v):
        if v < 0 and v >= 100:
            raise ValueError(
                "Patch-Overlap in percentage must be between 0 and 100 (100 not included)"
            )
        return v

    @validator("min_intersection_ratio")
    def min_intersection_ratio_range_check(cls, v):
        if v < 0 and v > 1:
            raise ValueError("Background ratio must be between 0 and 1")
        return v

    def __post_init_post_parse__(self) -> None:
        if self.label_map_file is None or self.label_map is None:
            self.label_map = {"background": 0}
        if self.otsu_annotation is not None:
            self.otsu_annotation = self.otsu_annotation.lower()
        if self.tissue_annotation is not None:
            self.tissue_annotation = self.tissue_annotation.lower()
        if len(self.exclude_classes) > 0:
            self.exclude_classes = [f.lower() for f in self.exclude_classes]
        if self.tissue_annotation_intersection_ratio is None:
            self.tissue_annotation_intersection_ratio = self.min_intersection_ratio
        else:
            if (
                self.tissue_annotation_intersection_ratio < 0
                and self.tissue_annotation_intersection_ratio > 1
            ):
                raise RuntimeError(
                    "Tissue_annotation_intersection_ratio must be between 0 and 1"
                )
        if self.annotation_path is not None:
            if not os.path.exists(self.annotation_path):
                raise FileNotFoundError(
                    f"Annotation path {self.annotation_path} does not exist"
                )
            if Path(self.annotation_path.suffix) != "json":
                raise ValueError("Only JSON annotations are supported")


class LivePatchWSIDataset(Dataset):
    def __init__(
        self,
        slide_processor_config: LivePatchWSIConfig,
        logger: logging.Logger = None,
        transforms: Callable = ToTensor(),
    ) -> None:
        """A class to represent a dataset of patches from whole slide images (WSI).

        This class provides functionality for extracting patches from WSIs using a specified configuration. It also provides
        functionality for loading and processing WSIs.

        Args:
            slide_processor_config (LivePatchWSIConfig): Configuration for preprocessing the dataset.
            logger (logging.Logger, optional): Logger for logging events. Defaults to None.
            transforms (Callable, optional): Transforms to apply to the patches. Defaults to ToTensor().

        Attributes:
            slide_openslide (OpenSlide): OpenSlide object for the slide
            image_loader (Union[OpenSlide, Any]): Image loader for the slide, method for loading the slide
            slide (Union[OpenSlide, Any]): Extraction object for the slide (OpenSlide, CuCIM, wsiDicomizer), instance of image_loader
            wsi_metadata (dict): Metadata of the WSI
            deepzoomgenerator (Union[DeepZoomGeneratorOS, Any]): Class for tile extraction, deepzoom-interface
            tile_extractor (Union[DeepZoomGeneratorOS, Any]): Instance of self.deepzoomgenerator
            config (LivePatchWSIConfig): Configuration for preprocessing the dataset
            logger (logging.Logger): Logger for logging events
            rescaling_factor (int): Rescaling factor for the slide
            interesting_coords (List[Tuple[int, int, float]]): List of interesting coordinates (patches -> row, col, ratio)
            curr_wsi_level (int): Level of the slide during extraction
            res_tile_size (int): Size of the extracted tiles
            res_overlap (int): Overlap of the extracted tiles
            polygons (List[Polygon]): List of polygons
            region_labels (List[str]): List of region labels
            transforms (Callable): Transforms to apply to the patches
            detector_device (str): Device for the detector model
            detector_model (torch.nn.Module): Detector model for filtering patches
            detector_transforms (Callable): Transforms to apply to the detector model
            mask_images (dict[str, PIL.Image.Image]): Dictionary of mask images for the patches, key is the name and value is the mask image

        Methods:
            __init__(slide_processor_config: PreProcessingDatasetConfig, logger: logging.Logger = None) -> None:
                Initializes the PatchWSIDataset.
            _set_hardware() -> None:
                Sets the hardware for the dataset.
            _prepare_slide() -> Tuple[List[Tuple[int, int, float]], int, List[Polygon], List[str]]:
                Prepares the slide for patch extraction.
            __len__() -> int:
                Returns the number of interesting coordinates (patches).
            __getitem__(index: int) -> Tuple[np.ndarray, dict, np.ndarray]:
                Returns the image tile, metadata and mask for a given index.
        """
        # slide specific
        self.slide_openslide: OpenSlide
        self.image_loader: Union[OpenSlide, Any]
        self.slide: Union[OpenSlide, Any]
        self.wsi_metadata: dict
        self.deepzoomgenerator: Union[
            DeepZoomGeneratorOS, Any
        ]  # function for tile extraction
        self.tile_extractor: Union[
            DeepZoomGeneratorOS, Any
        ]  # instance of self.deepzoomgenerator

        # config
        self.config = slide_processor_config
        self.logger = logger
        self.rescaling_factor = 1

        # extraction specific
        self.interesting_coords: List[Tuple[int, int, float]]
        self.curr_wsi_level: int
        self.res_tile_size: int
        self.res_overlap: int
        self.polygons: List[Polygon]
        self.region_labels: List[str]
        self.transforms = transforms
        self.mask_images: dict[str, Image.Image]

        # filter
        self.detector_device: str
        self.detector_model: torch.nn.Module
        self.detector_transforms: Callable

        if logger is None:
            self.logger = logging.getLogger(__name__)

        self._set_hardware()

        if self.config.filter_patches is True:
            self._set_tissue_detector()

        self.config.patch_overlap = int(
            np.floor(self.config.patch_size / 2 * self.config.patch_overlap / 100)
        )
        # set seed
        random.seed(42)

        # prepare slide
        (
            self.interesting_coords,
            self.curr_wsi_level,
            self.polygons,
            self.region_labels,
            self.mask_images,
        ) = self._prepare_slide()

    def _set_hardware(self) -> None:
        """Either load CuCIM (GPU-accelerated) or OpenSlide"""
        wsi_file = Path(self.config.wsi_path)
        if wsi_file.is_dir():
            if len(list(wsi_file.glob("*.dcm"))) != 0:
                self.logger.debug("Detected dicom files")
                try:
                    dcm_files = list(wsi_file.glob("*.dcm"))
                    dcm_files = [(f, os.path.getsize(f.resolve())) for f in dcm_files]
                    dcm_files = sorted(dcm_files, key=lambda x: x[1], reverse=True)
                    OpenSlide(str(dcm_files[0][0]))
                    wsi_file = dcm_files[0][0]
                    self.image_loader = OpenSlide
                    self.deepzoomgenerator = DeepZoomGeneratorOS
                    self.slide_metadata_loader = OpenSlide
                except:
                    try:
                        DicomSlide(wsi_file)
                    except Exception as e:
                        raise e
                    self.deepzoomgenerator = DeepZoomGeneratorDicom
                    self.image_loader = DicomSlide
                    self.slide_metadata_loader = DicomSlide
        elif wsi_file.suffix == ".dcm":
            self.logger.debug("Detected dicom files")
            try:
                OpenSlide(str(wsi_file))
                self.image_loader = OpenSlide
                self.deepzoomgenerator = DeepZoomGeneratorOS
                self.slide_metadata_loader = OpenSlide
            except:
                try:
                    DicomSlide(wsi_file)
                except Exception as e:
                    raise e
                self.deepzoomgenerator = DeepZoomGeneratorDicom
                self.image_loader = DicomSlide
                self.slide_metadata_loader = DicomSlide
        else:
            if module_exists("cucim", error="ignore"):
                self.logger.debug("Using CuCIM")
                from cucim import CuImage

                from pathopatch.wsi_interfaces.cucim_deepzoom import (
                    DeepZoomGeneratorCucim,
                )

                self.deepzoomgenerator = DeepZoomGeneratorCucim
                self.image_loader = CuImage
                self.slide_metadata_loader = OpenSlide
            else:
                self.logger.debug("Using OpenSlide")
                self.deepzoomgenerator = DeepZoomGeneratorOS
                self.image_loader = OpenSlide
                self.slide_metadata_loader = OpenSlide

    def _set_tissue_detector(self) -> None:
        """Set up the tissue detection model and transformations.

        Raises:
            ImportError: If torch or torchvision cannot be imported.
        """
        try:
            import torch.nn as nn
            from torchvision.models import mobilenet_v3_small
            from torchvision.transforms.v2 import (
                Compose,
                Normalize,
                Resize,
                ToDtype,
                ToTensor,
            )
        except ImportError:
            raise ImportError(
                "Torch cannot be imported, Please install PyTorch==2.0 with torchvision for your system (https://pytorch.org/get-started/previous-versions/)!"
            )
        self.detector_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        if self.detector_device == "cpu":
            self.logger.warning(
                "No CUDA device detected - Speed may be very slow. Please consider performing extraction on CUDA device or disable tissue detector!"
            )
        model = mobilenet_v3_small().to(device=self.detector_device)
        model.classifier[-1] = nn.Linear(1024, 4)
        checkpoint = torch.load(
            "/pathopatch/data/tissue_detector.pt",  # this causes errors
            map_location=self.detector_device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.detector_model = model
        self.detector_transforms = Compose(
            [
                Resize(224),
                ToTensor(),
                ToDtype(torch.float32),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ).to(self.detector_device)

    def _prepare_slide(
        self,
    ) -> Tuple[
        List[Tuple[int, int, float]],
        int,
        List[Polygon],
        List[str],
        dict[str, Image.Image],
    ]:
        """Prepare the slide for patch extraction

        This method prepares the slide for patch extraction by loading the slide, extracting metadata,
        calculating the magnification per pixel (MPP), and setting up the tile extractor. It also calculates
        the interesting coordinates for patch extraction based on the configuration.

        Raises:
            NotImplementedError: Raised when the MPP is not defined either by metadata or by the config file.
            WrongParameterException: Raised when the requested level does not exist in the slide.

        Returns:
            Tuple[List[Tuple[int, int, float]], int, List[Polygon], List[str]]:
                * List[Tuple[int, int, float]]: List of interesting coordinates (patches -> row, col, ratio)
                * int: Level of the slide
                * List[Polygon]: List of polygons, downsampled to the target level
                * List[str]: List of region labels
                * dict[str, Image.Image]: Dictionary of mask images for the patches, key is the name and value is the mask image
        """
        self.slide_openslide = self.slide_metadata_loader(str(self.config.wsi_path))
        self.slide = self.image_loader(str(self.config.wsi_path))

        if (
            self.config.wsi_properties is not None
            and "slide_mpp" in self.config.wsi_properties
        ):
            slide_mpp = self.config.wsi_properties["slide_mpp"]
        elif "openslide.mpp-x" in self.slide_openslide.properties:
            slide_mpp = float(self.slide_openslide.properties["openslide.mpp-x"])
        else:  # last option is to use regex
            try:
                pattern = re.compile(r"MPP(?: =)? (\d+\.\d+)")
                # Use the pattern to find the match in the string
                match = pattern.search(
                    self.slide_openslide.properties["openslide.comment"]
                )
                # Extract the float value
                if match:
                    slide_mpp = float(match.group(1))
                    self.logger.warning(
                        f"MPP {slide_mpp:.4f} was extracted from the comment of the WSI (Tiff-Metadata comment string) - Please check for correctness!"
                    )
                else:
                    raise NotImplementedError(
                        "MPP must be defined either by metadata or by config file!"
                    )
            except:
                raise NotImplementedError(
                    "MPP must be defined either by metadata or by config file!"
                )

        if (
            self.config.wsi_properties is not None
            and "magnification" in self.config.wsi_properties
        ):
            slide_mag = self.config.wsi_properties["magnification"]
        elif "openslide.objective-power" in self.slide_openslide.properties:
            slide_mag = float(
                self.slide_openslide.properties.get("openslide.objective-power")
            )
        else:
            raise NotImplementedError(
                "MPP must be defined either by metadata or by config file!"
            )

        slide_properties = {"mpp": slide_mpp, "magnification": slide_mag}

        resulting_mpp = None

        if self.config.target_mpp is not None:
            if (
                not slide_properties["mpp"] - self.config.target_mpp_tolerance
                <= self.config.target_mpp
                <= slide_properties["mpp"] + self.config.target_mpp_tolerance
            ):
                (
                    self.config.downsample,
                    self.rescaling_factor,
                ) = target_mpp_to_downsample(
                    slide_properties["mpp"],
                    self.config.target_mpp,
                )
            else:
                self.config.downsample = 1
                self.rescaling_factor = 1.0
            if self.rescaling_factor != 1.0:
                resulting_mpp = (
                    slide_properties["mpp"]
                    * self.rescaling_factor
                    * self.config.downsample
                )
            else:
                resulting_mpp = slide_properties["mpp"] * self.config.downsample
        # target mag has precedence before downsample!
        elif self.config.target_mag is not None:
            self.config.downsample = target_mag_to_downsample(
                slide_properties["magnification"],
                self.config.target_mag,
            )
            resulting_mpp = slide_properties["mpp"] * self.config.downsample

        self.res_tile_size, self.res_overlap = patch_to_tile_size(
            self.config.patch_size, self.config.patch_overlap, self.rescaling_factor
        )

        self.tile_extractor = self.deepzoomgenerator(
            meta_loader=self.slide_openslide,
            image_loader=self.slide,
            tile_size=self.res_tile_size,
            overlap=self.res_overlap,
            limit_bounds=True,
        )

        if self.config.downsample is not None:
            # Each level is downsampled by a factor of 2
            # downsample expresses the desired downsampling, we need to count how many times the
            # downsampling is performed to find the level
            # e.g. downsampling of 8 means 2 * 2 * 2 = 3 times
            # we always need to remove 1 level more than necessary, so 4
            # so we can just use the bit length of the numbers, since 8 = 1000 and len(1000) = 4
            level = (
                self.tile_extractor.level_count - self.config.downsample.bit_length()
            )
            if resulting_mpp is None:
                resulting_mpp = slide_properties["mpp"] * self.config.downsample
        else:
            self.config.downsample = 2 ** (self.tile_extractor.level_count - level - 1)
            if resulting_mpp is None:
                resulting_mpp = slide_properties["mpp"] * self.config.downsample

        if level >= self.tile_extractor.level_count:
            raise WrongParameterException(
                "Requested level does not exist. Number of slide levels:",
                self.tile_extractor.level_count,
            )

        # store level!
        self.curr_wsi_level = level

        # initialize annotation objects
        region_labels: List[str] = []
        polygons: List[Polygon] = []
        polygons_downsampled: List[Polygon] = []
        tissue_region: List[Polygon] = []

        # load the annotation if provided
        if self.config.annotation_path is not None:
            (
                region_labels,
                polygons,
                polygons_downsampled,
                tissue_region,
            ) = self._get_wsi_annotations(downsample=self.config.downsample)

        # get the interesting coordinates: no background, filtered by annotation etc.
        self.logger.debug("Calculating patches to sample")
        n_cols, n_rows = self.tile_extractor.level_tiles[level]
        if self.config.min_intersection_ratio == 0.0 and tissue_region is None:
            # Create a list of all coordinates of the grid -> Whole WSI with background is loaded
            interesting_coords = [
                (row, col, 1.0) for row in range(n_rows) for col in range(n_cols)
            ]
        else:
            (
                interesting_coords,
                mask_images,
                _,
            ) = compute_interesting_patches(
                slide=self.slide_openslide,
                tiles=self.tile_extractor,
                target_level=level if level is not None else 1,
                target_patch_size=self.res_tile_size,
                target_overlap=self.res_overlap,
                label_map=self.config.label_map,
                region_labels=region_labels,
                polygons=polygons,
                mask_otsu=self.config.masked_otsu,
                apply_prefilter=self.config.apply_prefilter,
                tissue_annotation=tissue_region,
                otsu_annotation=self.config.otsu_annotation,
                tissue_annotation_intersection_ratio=self.config.tissue_annotation_intersection_ratio,
                fast_mode=True,
            )
        self.logger.debug(f"Number of patches sampled: {len(interesting_coords)}")
        if len(interesting_coords) == 0:
            self.logger.warning(f"No patches sampled from {self.config.wsi_path}")

        self.wsi_metadata = {
            "orig_n_tiles_cols": n_cols,
            "orig_n_tiles_rows": n_rows,
            "base_magnification": slide_mag,
            "downsampling": self.config.downsample,
            "label_map": self.config.label_map,
            "patch_overlap": self.config.patch_overlap * 2,
            "patch_size": self.config.patch_size,
            "base_mpp": slide_mpp,
            "target_patch_mpp": resulting_mpp,
            "stain_normalization": self.config.normalize_stains,
            "magnification": slide_mag
            / (self.config.downsample * self.rescaling_factor),
            "level": level,
        }

        return (
            list(interesting_coords),
            level,
            polygons_downsampled,
            region_labels,
            mask_images,
        )

    def _get_wsi_annotations(
        self, downsample: int
    ) -> Tuple[List[str], List[Polygon], List[Polygon], List[Polygon]]:
        """Get the annotations for the WSI

        Args:
            downsample (int): Downsample factor

        Returns:
            Tuple[List[str], List[Polygon], List[Polygon], List[Polygon]]: Tuple containing the region labels, polygons, downsampled polygons, and tissue region
        """
        region_labels: List[str] = []
        polygons: List[Polygon] = []
        polygons_downsampled: List[Polygon] = []
        tissue_region: List[Polygon] = []

        polygons, region_labels, tissue_region = get_regions_json(
            path=Path(self.config.annotation_path),
            exclude_classes=self.config.exclude_classes,
            tissue_annotation=self.config.tissue_annotation,
        )

        polygons_downsampled = [
            scale(
                poly,
                xfact=1 / downsample,
                yfact=1 / downsample,
                origin=(0, 0),
            )
            for poly in polygons
        ]

        return region_labels, polygons, polygons_downsampled, tissue_region

    def __len__(self) -> int:
        """__len__ method for the dataset

        Returns:
            int: Number of interesting coordinates (patches)
        """
        return len(self.interesting_coords)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, dict, np.ndarray]:
        """Returns the image tile, metadata and mask for a given index

        Args:
            index (int): Index of the patch

        Returns:
            Tuple[np.ndarray, dict, np.ndarray]:
                * np.ndarray: Image tile
                * dict: Metadata of the patch
                * np.ndarray: Mask of the patch
        """
        discard_patch = False  # flag for discarding patch
        row, col, _ = self.interesting_coords[index]

        # openslide
        image_tile = np.asarray(
            self.tile_extractor.get_tile(self.curr_wsi_level, (col, row)),
            dtype=np.uint8,
        )
        image_tile = pad_tile(
            image_tile, self.res_tile_size + 2 * self.res_overlap, col, row
        )

        # calculate background ratio
        background_ratio = calculate_background_ratio(
            image_tile, self.config.patch_size
        )

        if background_ratio > 1 - self.config.min_intersection_ratio:
            self.logger.debug(
                f"Removing patch {row}, {col} because of intersection ratio with background is too big"
            )
            discard_patch = True
            image_tile = None
            intersected_labels = []  # Zero means background
            ratio = {}
            patch_mask = np.zeros(
                (self.res_tile_size, self.res_tile_size), dtype=np.uint8
            )
        else:
            intersected_labels, ratio, patch_mask = get_intersected_labels(
                tile_size=self.res_tile_size,
                patch_overlap=self.res_overlap,
                col=col,
                row=row,
                polygons=self.polygons,
                label_map=self.config.label_map,
                min_intersection_ratio=0,
                region_labels=self.region_labels,
                overlapping_labels=self.config.overlapping_labels,
                store_masks=self.config.store_masks,
            )
            if len(ratio) != 0:
                background_ratio = 1 - np.sum(ratio)
            ratio = {k: v for k, v in zip(intersected_labels, ratio)}

            if self.config.normalize_stains:
                image_tile, _, _ = macenko_normalization(
                    [image_tile],
                    normalization_vector_path=self.config.normalization_vector_json,
                )
                image_tile = image_tile[0]

            if image_tile.shape[0] != self.config.patch_size:
                image_tile = Image.fromarray(image_tile)
                if image_tile.size[-1] > self.config.patch_size:
                    image_tile.thumbnail(
                        (self.config.patch_size, self.config.patch_size),
                        getattr(Image, "Resampling", Image).LANCZOS,
                    )
                else:
                    image_tile = image_tile.resize(
                        (self.config.patch_size, self.config.patch_size),
                        getattr(Image, "Resampling", Image).LANCZOS,
                    )
                image_tile = np.array(image_tile)
            try:
                image_tile = self.transforms(image_tile)
            except TypeError:
                pass

        patch_metadata = {
            "row": row,
            "col": col,
            "background_ratio": float(background_ratio),
            "intersected_labels": intersected_labels,
            "label_ratio": ratio,
            "discard_patch": discard_patch,
        }

        return image_tile, patch_metadata, patch_mask


class LivePatchWSIDataloader:
    """Dataloader for LivePatchWSIDataset

    Args:
        dataset (LivePatchWSIDataset): Dataset to load patches from.
        batch_size (int): Batch size for the dataloader.
        shuffle (bool, optional): To shuffle iterations. Defaults to False.
        seed (int, optional): Seed for shuffle. Defaults to 42.
    """

    def __init__(
        self,
        dataset: LivePatchWSIDataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        assert isinstance(dataset, LivePatchWSIDataset)
        assert isinstance(batch_size, int)
        assert isinstance(shuffle, bool)
        assert isinstance(seed, int)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.element_list = list(range(len(self.dataset)))
        self.i = 0
        self.discard_count = 0

        if self.shuffle:
            grtr = np.random.default_rng(seed)
            self.element_list = grtr.permutation(self.element_list)

    def __iter__(self):
        self.i = 0
        self.discard_count = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, List[dict], List[np.ndarray]]:
        """Create one batch of patches

        Raises:
            StopIteration: If the end of the dataset is reached.

        Returns:
            Tuple[torch.Tensor, List[dict], List[np.ndarray]]:
                * torch.Tensor: Batch of patches, shape (batch_size, 3, patch_size, patch_size)
                * List[dict]: List of metadata for each patch
                * List[np.ndarray]: List of masks for each patch
        """
        patches = []
        metadata = []
        masks = []
        if self.i < len(self.element_list):
            batch_item_count = 0
            while batch_item_count < self.batch_size and self.i < len(
                self.element_list
            ):
                patch, meta, mask = self.dataset[self.element_list[self.i]]
                self.i += 1
                if patch is None and meta["discard_patch"]:
                    self.discard_count += 1
                    continue
                elif self.dataset.config.filter_patches:
                    output = self.dataset.detector_model(
                        self.dataset.detector_transforms(patch)[None, ...]
                    )
                    output_prob = torch.softmax(output, dim=-1)
                    prediction = torch.argmax(output_prob, dim=-1)
                    if int(prediction) != 0:
                        self.discard_count += 1
                        continue
                patches.append(patch)
                metadata.append(meta)
                masks.append(mask)
                batch_item_count += 1
            if len(patches) > 1:
                patches = [torch.tensor(f) for f in patches]
                patches = torch.stack(patches)
            elif len(patches) == 1:
                patches = torch.tensor(patches[0][None, ...])
            return patches, metadata, masks
        else:
            raise StopIteration

    def __len__(self):
        return int(np.ceil((len(self.dataset) - self.discard_count) / self.batch_size))
