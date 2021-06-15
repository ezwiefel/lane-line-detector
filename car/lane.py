from typing import List
import numpy as np
import car.utils as utils


class LaneFinder(object):

    def __init__(
        self,
        low_canny_threshold: int,
        high_canny_threshold: int,
        vertices: np.ndarray,
        rho: float = 2,
        theta: float = np.pi/180,
        hough_threshold: int = 10,
        min_line_len: float= 1.0,
        max_line_gap: float = 0.0,
        kernel_size: int = 1,
        line_color: List[int]=[255,0,0],
        line_thickness: int =2,
        calc_lane_top: bool = False
    ) -> None:

        if not utils.is_valid_kernel_size(kernel_size):
            raise ValueError(
            "kernel_size must either be 0 or a positive, odd integer")

        self._kernel_size = kernel_size

        self._canny_params = {
            "low_threshold": low_canny_threshold,
            "high_threshold": high_canny_threshold
        }

        self._hough_parameters = {
            "rho": rho,
            "theta": theta,
            "threshold": hough_threshold,
            "min_line_len": min_line_len,
            "max_line_gap": max_line_gap,
            "line_color": line_color,
            "line_thickness": line_thickness,
            "lane_top": vertices[0][1][1] if calc_lane_top else None
        }

        self._vertices = vertices

    def find_lines(self, img: np.ndarray) -> np.ndarray:

        gray = utils.grayscale(img)
        blur_gray = utils.gaussian_blur(gray, self._kernel_size)
        edges = utils.canny(blur_gray, **self._canny_params)
        masked_edges = utils.region_of_interest(edges, self._vertices)
        lines = utils.hough_lines(masked_edges, **self._hough_parameters)

        return lines

    def show_polygon(self, img: np.ndarray) -> np.ndarray:
        return utils.draw_vertices_on_image(img, self._vertices, thickness=3)