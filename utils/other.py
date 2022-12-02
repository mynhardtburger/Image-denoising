from ast import Call
from typing import Callable, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from skimage.util import img_as_ubyte, random_noise
from tqdm import tqdm

ImageShape = Tuple[int, int, int]


def pixel_range(image: npt.NDArray) -> None:
    """
    Prints out the value ranges for each channel of an image

    Args:
        image (npt.NDArray): Input image
    """
    for ch in range(image.shape[2]):
        print(
            "Channel:",
            ch,
            "Min:",
            image[:, :, ch].min(),
            "Max:",
            image[:, :, ch].max(),
        )
    print("Total min:", image.min(), "Total max:", image.max())


def img_compare(img1: npt.NDArray, img2: npt.NDArray) -> Dict[str, float]:
    """
    Calculates and returns the MSE and SSIM similarity metrics between two images.

    Args:
        img1 (npt.NDArray): Image 1 to compare
        img2 (npt.NDArray): Image 2 to compare

    Returns:
        Dict[str, float]: MSE, SSIM, PSNR
    """
    mse = mean_squared_error(img1, img2)
    ssim = structural_similarity(
        img1,
        img2,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    psnr = peak_signal_noise_ratio(img1, img2)

    return {
        "mse": mse,
        "ssim": ssim,
        "psnr": psnr,
    }


def make_noisy_images_gaussian(img: npt.NDArray, count: int = 10) -> npt.NDArray:
    """
    Generate X copies of the image with additive gaussian noise applied.
    Gaussian: 0 mean, 0.01 variance.
    Output pixel range of the noisy images is uint8 [0,255].

    Args:
        img (npt.NDArray): Input image
        count (int, optional): Number of copies Defaults to 10.

    Returns:
        npt.NDArray: Array of images
    """
    noisy_images = []
    for _ in range(count):
        noisy = random_noise(img, mode="gaussian", clip=True, mean=0, var=0.01)
        noisy = img_as_ubyte(noisy)
        noisy_images.append(noisy)

    noisy_arr = np.array(noisy_images)

    return noisy_arr


def make_noisy_images_sp(img: npt.NDArray, count: int = 10) -> npt.NDArray:
    """
    Generate X copies of the image with salt & pepper noise applied.
    Salt & Pepper: 0.05 amount.
    Output pixel range of the noisy images is uint8 [0,255].

    Args:
        img (npt.NDArray): Input image
        count (int, optional): Number of copies Defaults to 10.

    Returns:
        npt.NDArray: Array of images
    """
    noisy_images = []
    for _ in range(count):
        noisy = random_noise(img, mode="s&p", amount=0.05)
        noisy = img_as_ubyte(noisy)
        noisy_images.append(noisy)

    noisy_arr = np.array(noisy_images)

    return noisy_arr


def make_noisy_images_speckle(img: npt.NDArray, count: int = 10) -> npt.NDArray:
    """
    Generate X copies of the image with multiplicative speckle noise applied.
    Speckle: 0 mean, 0.01 variance.
    Output pixel range of the noisy images is uint8 [0,255].

    Args:
        img (npt.NDArray): Input image
        count (int, optional): Number of copies Defaults to 10.

    Returns:
        npt.NDArray: Array of images
    """
    noisy_images = []
    for _ in range(count):
        noisy = random_noise(img, mode="speckle", clip=True, mean=0, var=0.01)
        noisy = img_as_ubyte(noisy)
        noisy_images.append(noisy)

    noisy_arr = np.array(noisy_images)

    return noisy_arr


def reshape_1d_to_img(
    img: npt.NDArray, original_shape: ImageShape, axis: int = 0
) -> npt.NDArray:
    """
    Reshapes a 1D images into the shape provided.

    Args:
        img (npt.NDArray): 1D image array
        original_shape (ImageShape): desired output shape
        axis (int): Indicates along which axis the individual images occur: 0=row wise, 1=column wise.

    Returns:
        npt.NDArray: Fattened image in the desired output shape
    """
    if len(img.shape) == 2:
        # fatten array of 1D images.
        if axis == 0:
            images = np.reshape(img, (img.shape[0], *original_shape))
        if axis == 1:
            images = np.reshape(img.T, (img.T.shape[0], *original_shape))
        return images

    if len(img.shape) == 1:
        return np.reshape(img, original_shape)

    raise TypeError("Img must be either 1D or 2D array of images")


def reshape_img_to_1d(img: npt.NDArray) -> Dict[npt.NDArray, ImageShape]:
    """
    Flattens a 2D image to a 1D array.

    Args:
        img (npt.NDArray): 2D image in shape of (x, y, channel) or (#, x, y, channel)

    Returns:
        Dict[npt.NDArray, ImageShape]: Dictonary of 1d images and its original shape
    """

    if isinstance(img, list):
        img = np.stack(img)

    if len(img.shape) == 4:
        # flatten array of images
        flat_images = np.reshape(img, (img.shape[0], -1))
        return {
            "1d_img": flat_images,
            "shape": img[0].shape,
        }

    if len(img.shape) == 3:
        # flatten single image
        return {
            "1d_img": img.flatten(),
            "shape": img.shape,
        }

    raise TypeError("Img shape must be (x, y, channel) or (#, x, y, channel)")


def baseline_measurements_old(
    img_arr: npt.NDArray, measurements: int = 3
) -> pd.DataFrame:
    """
    Calculate MSE, SSIM and PSNR for images against default noisy versions.
    Gaussian, Salt & Pepper and Speckle noise is used.

    Args:
        img_arr (npt.NDArray): Array of images
        measurements (int, optional): Number of measurements for each image & noise type. Defaults to 3.

    Returns:
        pd.DataFrame: Results of each measurement (img_id, measurement, mse, ssim, psnr, noise_type)
    """
    results = []
    for img_idx, img in enumerate(tqdm(img_arr)):
        gauss = make_noisy_images_gaussian(img, measurements)
        sp = make_noisy_images_sp(img, measurements)
        speckle = make_noisy_images_speckle(img, measurements)

        for noisy_idx, noisy_img in enumerate(gauss):
            metrics = img_compare(img, noisy_img)
            metrics.update(
                img_id=img_idx, noise_type="gaussian", measurement=noisy_idx + 1
            )
            results.append(metrics)

        for noisy_idx, noisy_img in enumerate(sp):
            metrics = img_compare(img, noisy_img)
            metrics.update(img_id=img_idx, noise_type="s&p", measurement=noisy_idx + 1)
            results.append(metrics)

        for noisy_idx, noisy_img in enumerate(speckle):
            metrics = img_compare(img, noisy_img)
            metrics.update(
                img_id=img_idx, noise_type="speckle", measurement=noisy_idx + 1
            )
            results.append(metrics)
    return pd.DataFrame.from_records(results, index=["img_id", "measurement"])


def batch_img_compare(
    img_arr: List[npt.NDArray],
    noise_arr: List[npt.NDArray],
    noise_type_label: str,
    denoise_func: str = "None",
) -> pd.DataFrame:
    """
    Calculate MSE, SSIM and PSNR for images against the provided noisy versions.

    Args:
        img_arr (npt.NDArray): Array of images
        noisy_imgs (List[List[npt.NDArray]]): List of lists containing img arrays
        noise_type_label (str): Noise type label

    Returns:
        pd.DataFrame: Results of each measurement (img_id, measurement, mse, ssim, psnr, noise_type)
    """

    # test lengths of inputs
    if len(img_arr) != len(noise_arr):
        raise ValueError("Image lists all of the same length")

    results = []
    for img_idx, img in enumerate(tqdm(img_arr)):
        for noise_idx, noise_img in enumerate(noise_arr[img_idx]):
            metrics = img_compare(img, noise_img)
            metrics.update(
                img_id=img_idx,
                noise_type=noise_type_label,
                measurement=noise_idx + 1,
                denoise_function=denoise_func,
            )
            results.append(metrics)

    return pd.DataFrame.from_records(
        results, index=["denoise_function", "img_id", "measurement"]
    )


def batch_generate_noise(
    img_arr: List[npt.NDArray], noisefunc: Callable[[npt.NDArray], List[npt.NDArray]]
) -> List[npt.NDArray]:
    """
    For each input image generate noisy copies using the provided noise generating function.

    Args:
        img_arr (npt.NDArray): list of images
        noisefunc (Callable[[npt.NDArray], List[npt.NDArray]]): Callable function which takes an img input and outputs a list of noisy images.

    Returns:
        List[npt.NDArray]: List of image arrays.
    """
    noisy_img_arr = []
    for _, img in enumerate(tqdm(img_arr)):
        noisy_imgs = np.stack(noisefunc(img), axis=0)
        noisy_img_arr.append(noisy_imgs)

    return noisy_img_arr


def batch_evaluate_denoise(
    img_arr: List[npt.NDArray],
    noise_arr: List[npt.NDArray],
    noise_type_label: str,
    denoisefunc: Callable[[npt.NDArray], npt.NDArray],
    denoise_func_label: str,
) -> pd.DataFrame:
    """
    Wrapper function which for each image in the array of images:
    1. Denoises each noisy image using the provided denoise function.
    2. Calculates the mse, ssim, psnr metrics vs the original clean image for each noisy image.

    Args:
        img_arr (List[npt.NDArray]): Array of images
        noise_arr (List[npt.NDArray]): Array of noise images
        noise_type_label (str): Noise type label
        denoisefunc (Callable[[npt.NDArray],npt.NDArray]): Callable function which accepts an image and outputs an image
        denoise_func_label (str): Denoise function label

    Returns:
        pd.DataFrame: Results of each measurement (denoise_function, img_id, measurement, mse, ssim, psnr, noise_type)
    """
    results = []
    for img_idx, img in enumerate(tqdm(img_arr)):
        for noisy_idx, noisy_img in enumerate(noise_arr[img_idx]):
            denoised_img = denoisefunc(noisy_img)
            metrics = img_compare(denoised_img, img)
            metrics.update(
                {
                    "img_id": img_idx,
                    "noise_type": noise_type_label,
                    "measurement": noisy_idx + 1,
                    "denoise_function": denoise_func_label,
                }
            )
            results.append(metrics)
    return pd.DataFrame.from_records(
        results, index=["denoise_function", "img_id", "measurement"]
    )


def batch_evaluate_series_denoise(
    img_arr: List[npt.NDArray],
    noise_arr: List[npt.NDArray],
    noise_type_label: str,
    denoisefunc: Callable[[npt.NDArray], npt.NDArray],
    denoise_func_label: str,
) -> pd.DataFrame:
    """
    Wrapper function which for each image in the array of images:
    1. Denoises each noisy image using the provided denoise function.
    2. Calculates the mse, ssim, psnr metrics vs the original clean image for each noisy image.

    Args:
        img_arr (List[npt.NDArray]): Array of images
        noise_arr (List[npt.NDArray]): Array of noise images
        noise_type_label (str): Noise type label
        denoisefunc (Callable[[npt.NDArray],npt.NDArray]): Callable function which accepts an image and outputs an image
        denoise_func_label (str): Denoise function label

    Returns:
        pd.DataFrame: Results of each measurement (denoise_function, img_id, measurement, mse, ssim, psnr, noise_type)
    """
    results = []
    for img_idx, img in enumerate(tqdm(img_arr)):
        denoised_img = denoisefunc(list(noise_arr[img_idx]))
        metrics = img_compare(denoised_img, img)
        metrics.update(
            {
                "img_id": img_idx,
                "noise_type": noise_type_label,
                "measurement": pd.NA,
                "denoise_function": denoise_func_label,
            }
        )
        results.append(metrics)
    return pd.DataFrame.from_records(
        results, index=["denoise_function", "img_id", "measurement"]
    )
