from typing import List, Tuple

import numpy.typing as npt
from skimage.exposure import rescale_intensity
from skimage.metrics import mean_squared_error
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.other import reshape_1d_to_img, reshape_img_to_1d


def run_pca(
    noisy_images: npt.NDArray,
    whiten: bool = False,
    return_stats: bool = False,
    mse_threshold: int = 5000,
) -> npt.NDArray | Tuple[npt.NDArray, List[float], List[float]]:
    """
    Run PCA

    Args:
        noisy_images (npt.NDArray): Input NDArray where each column represents an image
        img_shape (Tuple[int, int, int]): Shape of original image. Used to reshape output components back to images.

    Returns:
        Tuple[List[npt.NDArray], List[float], List[float]]: transformed principal components, PC explained variances, PC explained variance ratios
    """
    imgarr_1d = reshape_img_to_1d(noisy_images)

    pca = PCA(n_components=1, whiten=whiten)
    pcas = pca.fit_transform(imgarr_1d["1d_img"].T)

    pca_imgs = reshape_1d_to_img(pcas, imgarr_1d["shape"], axis=1)
    pca_img = rescale_intensity(pca_imgs[0], out_range="uint8")

    # PC's sign is likely flipped if the MSE is too large compared to the original image.
    mse = mean_squared_error(pca_imgs[0], noisy_images[0])
    if mse > mse_threshold:
        pca_imgs = reshape_1d_to_img(pcas * -1, imgarr_1d["shape"], axis=1)
        pca_img = rescale_intensity(pca_imgs[0], out_range="uint8")

    if return_stats:
        return pca_img, pca.explained_variance_, pca.explained_variance_ratio_
    return pca_img


def run_fastica(
    noisy_images: npt.NDArray,
) -> Tuple[List[npt.NDArray], List[float], List[float]]:
    """
    Run FastICA

    Args:
        noisy_images (npt.NDArray): Input NDArray where each column represents an image
        img_shape (Tuple[int, int, int]): Shape of original image. Used to reshape output components back to images.

    Returns:
        Tuple[List[npt.NDArray], List[float], List[float]]: transformed independent components, PC explained variances, PC explained variance ratios
    """
    imgarr_1d = reshape_img_to_1d(noisy_images)

    ica = FastICA(n_components=1, whiten="arbitrary-variance")
    pipe = Pipeline([("standardscaler", StandardScaler()), ("ica", ica)])
    icas = pipe.fit_transform(imgarr_1d["1d_img"].T)

    ica_imgs = reshape_1d_to_img(icas, imgarr_1d["shape"], axis=1)
    ica_img = rescale_intensity(ica_imgs[0], out_range="uint8")

    return ica_img
