import zipfile
from pathlib import Path
from typing import List
from urllib import request

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def download_dataset(remote_url: str, filename_path: str) -> None:
    """
    Downloads the dataset from a given URL saving the file to the given filname path

    Args:
        remote_url (str): URL
        filename (str): path and filename to which the file needs to be save
    """

    print("Downloading", remote_url)
    request.urlretrieve(remote_url, filename_path)
    print("Saved", filename_path)


def extract_images(dataset_zip: str, ouput_dir: str, file_pattern: str) -> None:
    """
    Extracts matched files from a zip archive to a given output director. The extraction ignores the zip archive file structure.

    Args:
        dataset_zip (str): file path of the zip archive which hold the images
        ouput_dir (str): path to the extraction directory. If they directory doesn't exist it will be created.
        file_pattern (str): Glob pattern of the files to extract from the dataset_zip archive
    """

    my_zip = Path(dataset_zip)
    output_dirpath = Path(ouput_dir)

    if not Path.exists(output_dirpath):
        print("Creating output directory:", output_dirpath)
        Path.mkdir(output_dirpath)

    with zipfile.ZipFile(my_zip, "r") as zip_file:
        for filename in zip_file.namelist():
            if Path(filename).match(file_pattern):
                data = zip_file.read(filename)
                myfile_path = output_dirpath / Path(filename).name
                print("Extracting:", filename, "to", myfile_path, sep=" ")
                myfile_path.write_bytes(data)
        print("Extraction completed.")


def load_images(directory_path: str) -> List[npt.NDArray]:
    """
    Loads PNG images into a list of numpy NDArrays.

    Args:
        directory_path (str): Directory from which to load the PNG images.

    Returns:
        List[npt.NDArray]: List of images in the form of numpy NDArrays
    """

    HEIGHT = []
    WIDTH = []
    img_array = []

    images_directory = Path(directory_path)
    files = images_directory.iterdir()

    for file in tqdm(files):
        if file.suffix == ".png":
            img = cv2.imread(str(file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape

            img_array.append(np.array(img))
            HEIGHT.append(height)
            WIDTH.append(width)

    print("Image count:", len(img_array))
    print()
    print("MIN HEIGHT:", min(HEIGHT))
    print("MAX HEIGHT:", max(HEIGHT))
    print()
    print("MIN WIDTH:", min(WIDTH))
    print("MAX WIDTH:", max(WIDTH))

    return img_array
