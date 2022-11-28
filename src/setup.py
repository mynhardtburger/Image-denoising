import math
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


def add_noise(noise_typ: str, image: npt.NDArray):
    # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """
    One of the following strings, selecting the type of noise to add:
    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

    Args:
        noise_typ (str): _description_
        image (npt.NDArray): _description_

    Returns:
        _type_: image with noise added
    """
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = cv2.add(image, gauss)
        return noisy

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = cv2.add(image, image * gauss)
        return noisy


def add_gaussian_noise(
    image: npt.NDArray, mean: float = 0, variance: float = 0.01
) -> npt.NDArray:
    # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """
    Adds Gaussian white noise with the provided mean and variance

    Args:
        image (npt.NDArray): Input image
        mean (float): Mean of gaussian white noise
        variance (float): Variance of gaussian white noise

    Returns:
        npt.NDArray: image with noise added
    """

    row, col, ch = image.shape
    stddev = math.sqrt(variance)
    gauss = np.random.normal(mean, stddev, (row, col, ch))
    gauss = gauss.reshape(row, col, ch).astype(int)
    noisy = cv2.add(image, gauss)
    return noisy


def add_sp_noise(image: npt.NDArray, noise_density: float = 0.05):
    # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """
    Each pixel is assigned a random probability value from a standard uniform distribution on the open interval (0, 1).
    For pixels with probability value in the range (0, noise_density/2), the pixel value is set to 0.
    For pixels with probability value in the range [noise_density/2, noise_density), the pixel value is set to the maximum value of the image data type
    For pixels with probability value in the range [d, 1), the pixel value is unchanged.

    Args:
        image (npt.NDArray): Input image
        noise_density (float): Proportion of pixels to be randomly affected by salt & pepper noise

    Returns:
        npt.NDArray: image with noise added
    """

    sp_prob = np.random.random_sample(image.shape)
    pepper_mask = sp_prob < (noise_density / 2)
    salt_mask = (sp_prob < noise_density) & ~pepper_mask

    noisy = np.copy(image)
    noisy[pepper_mask] = 0
    noisy[salt_mask] = 1
    return noisy


def add_noise(image: npt.NDArray, variance: float = 0.05) -> npt.NDArray:
    # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """


    Args:
        image (npt.NDArray): _description_
        variance (float): Variance of uniformly distributed random noise with mean 0

    Returns:
        npt.NDArray: image with noise added

    """

    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = cv2.add(image, (image * gauss).astype(int))
    return noisy
