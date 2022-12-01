from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy.typing as npt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

from utils.other import img_compare


def plot_zoom_image(img: npt.NDArray, title: str) -> None:
    """
    Plots image with zoomed in sample patch (picture in picture).

    Args:
        img (npt.NDArray): Image
        title (str): Plot title
    """

    # calculate coordinates of zoomed in portion
    width = img.shape[1]
    height = img.shape[0]
    inset_size_ratio = 0.1
    x1 = int(width * 0.6 - width * inset_size_ratio)
    x2 = x1 + int(width * inset_size_ratio)
    y1 = int(height * 0.4 - height * inset_size_ratio)
    y2 = y1 + int(width * inset_size_ratio)

    # common parameters
    edge_color = "red"
    line_width = 3

    # Plot main image
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(title)

    # Plot zoomed in picture
    axins: plt.Axes = zoomed_inset_axes(parent_axes=ax, zoom=4, loc="lower left")
    axins.imshow(img)

    # Set boundry of main image to zoom in
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)  # https://github.com/matplotlib/matplotlib/issues/13649

    # Switch off axis tick marks
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)

    # Set axes line width and line color
    axins.spines["bottom"].set_color(edge_color)
    axins.spines["bottom"].set_linewidth(line_width)
    axins.spines["top"].set_color(edge_color)
    axins.spines["top"].set_linewidth(line_width)
    axins.spines["right"].set_color(edge_color)
    axins.spines["right"].set_linewidth(line_width)
    axins.spines["left"].set_color(edge_color)
    axins.spines["left"].set_linewidth(line_width)

    # Create guide lines to source patch which is being enlarged
    _, pp1, pp2 = mark_inset(
        parent_axes=ax,
        inset_axes=axins,
        loc1=2,  # connect line to corner. manually set later.
        loc2=4,  # connect line to corner. manually set later.
        edgecolor=edge_color,
        linewidth=line_width,
    )
    pp1.loc1, pp1.loc2 = (
        2,
        3,
    )  # inset corner 2 to origin corner 3 (these are non-intuitive due to the flipping of coordinates)
    pp2.loc1, pp2.loc2 = (
        4,
        1,
    )  # inset corner 4 to origin corner 1 (these are non-intuitive due to the flipping of coordinates)

    plt.show()


def plot_results(
    truth: npt.NDArray, input: npt.NDArray, result: npt.NDArray
) -> Tuple[float, float]:
    """
    Plots the 3 images (ground truth image, example of noisy image, processed resulting image) for comparison together with MSE and SSIM similarity metrics.

    Args:
        truth (npt.NDArray): Ground truth image
        input (npt.NDArray): Example of noisy image
        result (npt.NDArray): Processed resulting image

    Returns:
        Tuple[float, float]: (MSE, SSIM)
    """
    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    truth_plt = fig.add_subplot(grid[:2, :2])
    input_plt = fig.add_subplot(grid[:2, 2:])

    result_plt = fig.add_subplot(grid[2:, 1:3])

    truth_plt.set_title("Ground Truth")
    truth_plt.imshow(truth)

    input_plt.set_title("Input")
    input_plt.imshow(input)

    result_plt.set_title("Result")
    result_plt.imshow(result)

    return img_compare(truth, result)


def plot_intensity_dist(img: npt.NDArray, title: str = ""):
    """
    Plot colour intensity for a single image

    Args:
        img (npt.NDArray): Single image in shape (x, y, channels)
    """

    histr_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    histr_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    histr_b = cv2.calcHist([img], [2], None, [256], [0, 256])

    plt.plot(histr_r / histr_r.sum(), color="r", label="Red")
    plt.plot(histr_g / histr_g.sum(), color="g", label="Green")
    plt.plot(histr_b / histr_b.sum(), color="b", label="Blue")
    plt.legend()
    plt.ylabel("Frequency %")
    plt.xlabel("Pixel intensity")
    plt.title(title)
    plt.ylim(bottom=0, top=0.02)

    plt.show()
