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
        if projection_matrix.shape[0] == 3 and projection_matrix.shape[1] == 3:
            _projection_matrix = np.eye(4)
            _projection_matrix[0:3, 0:3] = projection_matrix
            projection_matrix = _projection_matrix
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

    def backproject_points(self, depth_values: np.ndarray):
        # TODO: seems not to be 100% correct. Check with the other (working) functions below
        a = self.projection_matrix[0, 0]
        b = self.projection_matrix[0, 2]
        c = self.projection_matrix[1, 1]
        d = self.projection_matrix[1, 2]
        projection_matrix_inv = np.array([[1 / a, 0, - b / a],
                                          [0, 1 / c, -d / c],
                                          [0, 0, 1]])

        screen_points = []
        positions = []
        for y in range(self.img_height):
            for x in range(self.img_width):
                depth = depth_values[y, x]
                if depth > 0:
                    screen_points.append([float(x * depth), float((self.img_height - y) * depth), float(depth)])
                    positions.append((y, x))
        screen_points = np.array(screen_points, dtype=np.float64)
        projected_points = projection_matrix_inv @ screen_points.reshape((-1, 3)).T

        return projected_points.T, positions


def backproject_points(intrinsics: np.ndarray, depth_values: np.ndarray, pixels: np.ndarray):
    """
    Projects the given pixels back to 3D camera space using the given intrinsics matrix and corresponding depth values.

    Parameters
    ----------
        intrinsics:
            the 3x3 matrix defining the camera intrinsics. For backprojecting, the inverse is needed which
            is computed automatically
        depth_values:
            a list of depth values that correspond to the pixels (one per pixel)
        pixels:
            (n, 2) matrix of pixel values that are to be back projected

    Returns
    -------
        (n, 3) numpy array of the projected points
    """
    if not isinstance(depth_values, np.ndarray):
        depth_values = np.array(depth_values)
    if not isinstance(pixels, np.ndarray):
        pixels = np.array(pixels)

    if len(pixels.shape) == 2:
        pixels = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    intrinsics_inv = np.linalg.inv(intrinsics)

    points_camera = intrinsics_inv @ pixels.T
    points_camera = points_camera * np.expand_dims(depth_values, 0)
    points_camera[1, :] = -points_camera[1, :]  # Invert y-axis
    return points_camera.T


def backproject_image(intrinsics, depth_image):
    """
    Projects the whole image back to 3D camera space using the given intrinsics matrix and depth image.

    Parameters
    ----------
        intrinsics:
            the 3x3 matrix defining the camera intrinsics. For backprojecting, the inverse is needed which
            is computed automatically
        depth_image:
            (h, w) matrix containing the depth values for each pixel

    Returns
    -------
        (h*w, 4) list of the projected 4d homogenous points
    """
    img_width = depth_image.shape[1]
    img_height = depth_image.shape[0]
    depth_intrinsics_inv = np.linalg.inv(intrinsics)

    points_depth_screen = np.array(
        [[[float(x), float(y), float(1)] for x in range(img_width)] for y in range(img_height)], dtype=np.float64)
    points_depth_screen *= np.expand_dims(depth_image, -1)
    points_depth_screen = points_depth_screen.reshape(-1, 3)

    points_depth_projected = depth_intrinsics_inv @ points_depth_screen.T
    points_depth_projected = np.vstack((points_depth_projected, np.ones((1, points_depth_screen.shape[0]))))
    return points_depth_projected.T


def register_rgb_depth(depth_data: np.ndarray, rgb_data: np.ndarray,
                       depth_intrinsics: np.ndarray, rgb_intrinsics: np.ndarray,
                       extrinsics: np.ndarray):
    """
    For RGB-D images, the depth and color channel are often not aligned, as they stem from different cameras. As such
    the depth information has to be transformed to match the color information. This process is called registration.
        1) Project depth image to camera space
        2) Apply camera extrinsics of color image
        3) Project 3D points onto color screen space
        4) Obtain corresponding color for each projected 3D point from 2D color image
    Not all pixels in the depth image will have a counterpart in the color image due to slight offsets of the cameras.
    This method only returns back-projected 3D points that have a color counterpart.

    Parameters
    ----------
        depth_data:
            (h, w) depth image
        rgb_data:
            (h, w, 3) color image
        depth_intrinsics:
            (3, 3) matrix with intrinsics of depth camera
        rgb_intrinsics:
            (3, 3) matrix with intrinsics of color camera
        extrinsics:
            (4, 4) matrix with extrinsics of color camera

    Returns
    -------
        points: (n, 3) list of back-projected 3D points
        colors: (n, 3) list of corresponding colors
        positions: (n, 2) list of corresponding pixel locations in the color image.

    """
    rgb_height = rgb_data.shape[0]
    rgb_width = rgb_data.shape[1]

    # backproject to 3D and transform to color camera space
    points_depth_projected = backproject_image(depth_intrinsics, depth_data)
    points_depth_projected = extrinsics @ points_depth_projected.T
    points_depth_projected = points_depth_projected.T
    points_depth_projected = points_depth_projected[:, :3]

    # Project to RGB screen
    points_rgb_screen = rgb_intrinsics @ (points_depth_projected / points_depth_projected[:, 2, None]).T

    points = []
    colors = []
    positions = []
    # Collect color information of corresponding pixels in RGB image
    for i, point_rgb_screen in enumerate(points_rgb_screen.T):
        x = point_rgb_screen[0]
        y = point_rgb_screen[1]

        if 0 <= x < rgb_width and 0 <= y < rgb_height:
            x = round(x)
            y = round(y)
            positions.append((x, y))

            point_3d = points_depth_projected[i][:3]
            points.append([point_3d[0], -point_3d[1], point_3d[2]])  # Invert y-axis
            colors.append(rgb_data[y, x])

    return np.array(points), np.array(colors), positions
