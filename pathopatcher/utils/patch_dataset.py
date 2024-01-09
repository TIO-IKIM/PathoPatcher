from pathlib import Path
from typing import Callable, Tuple

import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TissueDetectionDataset(Dataset):
    """
    TissueDetectionDataset class for loading patched WSI images for tissue detection.

    Args:
        patched_wsi_path (str): Path to the directory containing patched WSI images.
        transforms (torchvision.transforms.Compose): Transformations to apply to the images.

    Attributes:
        patched_wsi_path (Path): Path to the directory containing patched WSI images.
        image_folder (Path): Path to the directory containing the patched images.
        transforms (torchvision.transforms.Compose): Transformations to apply to the images.
        image_list (List[Path]): List of paths to the patched images.

    Methods:
        __len__() -> int:
            Returns the total number of patched images in the dataset.

        __getitem__(index: int) -> Tuple[torch.Tensor, str]:
            Retrieves the image and its corresponding filename at the given index.

    Raises:
        None

    Returns:
        None

    Example:
        dataset = TissueDetectionDataset(patched_wsi_path="/path/to/patched_wsi", transforms=transforms)
        image, image_name = dataset[0]
    """

    def __init__(self, patched_wsi_path: str, transforms: Callable):
        self.patched_wsi_path = Path(patched_wsi_path)
        self.image_folder = self.patched_wsi_path / "patches"
        self.transforms = transforms
        self.image_list = natsorted(
            [x for x in self.image_folder.iterdir() if x.is_file()]
        )

    def __len__(self) -> int:
        """Returns the total number of patched images in the dataset."""
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """Retrieves the image and its corresponding filename at the given index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, str]: The image tensor and its corresponding filename.
        """
        image_filepath = self.image_list[index].resolve()
        image_name = self.image_list[index].name

        image = Image.open(image_filepath)
        image = self.transforms(image)

        return image, image_name


def load_tissue_detection_dl(patched_wsi_path: str, transforms: Callable) -> DataLoader:
    """
    Loads the TissueDetectionDataset into a DataLoader for tissue detection inference.

    Args:
        patched_wsi_path (str): Path to the directory containing patched WSI images.
        transforms (torchvision.transforms.Compose): Transformations to apply to the images.

    Returns:
        DataLoader: DataLoader object for the TissueDetectionDataset.

    Example:
        dataloader = load_tissue_detection_dl(patched_wsi_path="/path/to/patched_wsi", transforms=transforms)
    """
    inference_ds = TissueDetectionDataset(patched_wsi_path, transforms)
    inference_dl = DataLoader(
        dataset=inference_ds,
        batch_size=256,
        num_workers=8,
        prefetch_factor=4,
        shuffle=False,
    )

    return inference_dl
