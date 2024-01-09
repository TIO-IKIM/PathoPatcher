from pathlib import Path
import requests
from tqdm import tqdm


def file_exists(directory_path: Path, file_name: str) -> bool:
    """Check if a file exists in a specific directory.

    Args:
        directory_path (Path): The path of the directory to check.
        file_name (str): The name of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_path = directory_path / file_name
    return file_path.exists()


def download_file(download_link: str, file_path: Path) -> None:
    """Download a file from a link and save it to a specific path.

    Args:
        download_link (str): The link to download the file from.
        file_path (Path): The path to save the downloaded file to.

    Raises:
        HTTPError: If the download request fails.
    """
    response = requests.get(download_link, stream=True)

    # Ensure the request was successful
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KiloByte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def check_and_download(
    directory_path: Path, file_name: str, download_link: str
) -> None:
    """Check if a file exists, and download it if it does not exist.

    Args:
        directory_path (Path): The path of the directory to check.
        file_name (str): The name of the file to check.
        download_link (str): The link to download the file from if it does not exist.
    """
    if not file_exists(directory_path, file_name):
        file_path = directory_path / file_name
        print("Downloading file...")
        download_file(download_link, file_path)
        print(
            f"The file {file_name} has been successfully downloaded and is located in {directory_path}."
        )
    else:
        print(f"The file {file_name} already exists in {directory_path}.")


def check_test_database() -> None:
    """Check if the test database exists, and download it if it does not exist."""
    print("Checking Test Database")
    base_path = Path(__file__).parent / "input/WSI/"
    check_and_download(
        base_path,
        "CMU-1.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs",
    )
    check_and_download(
        base_path,
        "CMU-1-Small-Region.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs",
    )
    check_and_download(
        base_path,
        "JP2K-33003-1.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/JP2K-33003-1.svs",
    )
    print("Test Database is now cached on local machine.")


if __name__ == "__main__":
    check_test_database()
