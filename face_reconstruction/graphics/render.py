from typing import Union, List

import numpy as np


class SimpleImageRenderer:

    def __init__(self, projection_matrix: np.ndarray, img_width: int, img_height: int):
        """

        :param projection_matrix:
            Projection matrix of the camera to transform 3D camera space points into 2D screen coordinates
        :param img_width:
            width of the image. Just acts as a threshold to define which projected points are in a "visible" area
        :param img_height:
            height of the image.
        """
        self.projection_matrix = projection_matrix
        self.img_width = img_width
        self.img_height = img_height

    def project_points(self, camera_pose: np.ndarray, points: Union[np.ndarray, List]):
        """
        Projects the given points to (continuous) raster space
        :param camera_pose: (4,4)
            4x4 pose matrix of camera. Note: this is actually the pose matrix of the object in the world
        :param points: (n, 3) or (n, 4) or (3,) or (4,)
            world space coordinates of the points to render
        :return: (n, 2)
            raster space coordinates of the given points. All points will be returned even if they are not in the
            visible area. The visible area is defined as [0, img_width] and [0, img_height].
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if points.shape[1] == 3:
            points = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))

        # Project points to camera space
        points_projected = self.projection_matrix @ camera_pose @ points.T

        # Divide by z-coordinate
        points_screen = points_projected / points_projected[2, :]

        # Invert y-coordinate (raster space origin is top left)
        points_screen[1, :] = -points_screen[1, :]

        # Scale x- and y-coordinate such that visible points range between [0, img_width] and [0, img_height]
        points_pixels = (points_screen + 1) / 2 * np.array([[self.img_width], [self.img_height], [0], [0]])
        points_pixels = points_pixels[:2, :].T

        return points_pixels
