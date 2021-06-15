import math
from typing import Iterable, List, Union, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


def grayscale(img: np.ndarray) -> np.ndarray:
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def is_valid_kernel_size(num: int) -> bool:
    if num == 0:                # Can be 0
        ret_val = True
    elif num < 0:               # Must be positive
        ret_val = False
    else:                       # Must be odd
        ret_val = num % 2 != 0
    return ret_val


def canny(img, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Applies a Gaussian Noise kernel"""
    if not is_valid_kernel_size(kernel_size):
        raise ValueError(
            "kernel_size must either be 0 or a positive, odd integer")

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    return img

def draw_lanes(
    img: np.ndarray,
    lines: List[np.ndarray],
    color=[255, 0, 0],
    thickness=2,
    slope_cutoff=0.5,
    lane_top=None
) -> np.ndarray:
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    if lines is None:
        return None

    img = img.copy()
    img_height, _, _ = img.shape
    # There should be 2 lanes for each image - left and right
    # The slope of each line should be approximately the same.
    # Presumably, there are Hough line for each lane line

    # Left lane points will have a negative slope and right lane points will have a positive slope.
    # Because origin is upper-left of photo

    # It is possible (during lane changes, for the slope to approach infinite/undefined)
    lanes = {
        "right": [],
        "left": [],
        'other': []
    }
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = calculate_slope(x1, y1, x2, y2)
            line_slope = [x1, y1, x2, y2, slope]
            # Right lane check - slope between cutoff and infinity
            if slope_cutoff <= slope < np.inf:
                lane_side = 'right'
            elif -slope_cutoff >= slope > -np.inf:
                lane_side = 'left'
            else:
                lane_side = 'other'

            lanes[lane_side].append(line_slope)

    for line_side in ['left', 'right']:
        if lanes[line_side]:
            x1, y1, _, _ = find_lane_end(lanes[line_side])
            slope = calc_average_slope(lanes[line_side])
            x2, y2 = interpolate_line_bottom(x1, y1, img_height, slope)

            if lane_top:
                x1, y1 = interpolate_line_top(x2, y2, lane_top, slope)

            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def find_lane_end(lines):
    # The lane endpoint will be where y1 value is the minimum
    y_min = np.inf
    line_end = None
    for line in lines:
        x1, y1, x2, y2, slope = line
        if y2 < y_min:
            y_min = y1
            line_end = (x1, y1, x2, y2)
    return line_end


def interpolate_line_top(x2: int, y2: int, lane_top, slope) -> Tuple[int, int]:

    x1 = x2 - ((y2 - lane_top)/slope)
    return int(x1), lane_top


def interpolate_line_bottom(x1: int, y1: int, img_height: int, slope: float):
    x2 = ((img_height - y1) / slope) + x1
    return int(x2), img_height


def calc_average_slope(lines: List) -> float:
    return np.average([line[4] for line in lines])


def calculate_slope(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Calculate the slope as ((y2-y1)/(x2-x1))

    However, it is possible (during lane changes, etc.) that the slope of the line
    would be infinity (a vertical line), in that case, return np.Inf
    """
    try:
        return ((y2-y1)/(x2-x1))
    except ZeroDivisionError:
        return np.inf


def hough_lines(
    img: np.ndarray,
    rho: float,
    theta: float,
    threshold: int,
    min_line_len: float,
    max_line_gap: float,
    line_color:List[int]=[255, 0, 0],
    line_thickness: int=2,
    lane_top: int=None
) -> np.ndarray:
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = draw_lanes(line_img, lines, color=line_color,
                          thickness=line_thickness, lane_top=lane_top)
    return line_img


def weighted_img(
    img: np.ndarray,
    initial_img: np.ndarray,
    alpha: float = 0.8,
    beta: float = 1.0,
    gamma: float = 0.0
):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    if img is None:
        return initial_img
    else:
        return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def draw_vertices_on_image(img: np.ndarray, vertices: np.ndarray, thickness: int) -> np.ndarray:
    v1 = np.roll(vertices, -1, axis=1)
    lines = np.dstack((vertices, v1)).reshape(4, 1, 4)

    return draw_lines(img, lines, thickness=thickness)


def find_plot_shape(num: int):
    """
    Create a rectangular shape with enough spots for the number of images. 
    Try to get as close to a square as possible given the number.

    find_plot_shape(9) will return (3, 3)
    find_plot_shape(9) will return (3, 4)
    find_plot_shape(12) will return (3, 4)
    find_plot_shape(13) will return (4, 4)

    """
    sqrt = np.sqrt(num)
    rows = int(np.ceil(sqrt))
    cols = int(np.ceil(num / rows))
    return (rows, cols)


def get_vertices(img: np.ndarray, height: int, top_width: int):
    """
    Get the vertices of the quadrilateral mask for lane detector

    Assumes that lower right and lower left are always in the bottom corners 
    of the image

    Then takes in the height value (in pixels) and top_width (in pixels) 
    and returns the vertices of the quadrilateral
    """

    img_height, img_width, _ = img.shape

    bottom_left = (0, img_height)
    bottom_right = (img_width, img_height)

    midpoint = img_width//2

    top_left = (midpoint-(top_width//2), img_height-height)
    top_right = (midpoint+(top_width//2), img_height-height)

    return np.array([[bottom_left, top_left, top_right, bottom_right]])


def plot_all_images(images: List[np.ndarray], figsize=(12, 12)) -> plt.Figure:
    rows, cols = find_plot_shape(len(images))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for ix, image in enumerate(images):
        axes.flatten()[ix].imshow(image)

    return fig
