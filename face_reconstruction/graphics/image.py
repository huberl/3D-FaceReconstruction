from statistics import mean
import numpy as np


def draw_pixels_to_image(img, pixels, color=1):
    img_width = img.shape[1]
    img_height = img.shape[0]

    for pixel in pixels:

        x = round(pixel[0])
        y = round(pixel[1])
        if x >= 0 and x < img_width and y >= 0 and y < img_height:
            img[y, x] = color


def cv2_to_plt(img):
    # Swap red and blue channels
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    return img


def interpolate_around(img: np.ndarray, pixel, size=1):
    """
    Returns the value at the given pixel coordinate if it is defined, i.e. not 0. Otherwise, tries to interpolate
    from neighbouring pixels by just averaging over defined values there.

    Parameters
    ----------
        img:
            the image to query values from (usually an incomplete depth image)
        pixel:
            where to get a potentially interpolated value
        size:
            how large the interpolation window should be. size=1 means that 1 pixel to the left/right/top/bottom will be
            considered for interpolation

    Returns
    -------
        the pixel value at the given pixel coordinate, if defined
        an average of the closest defined values (depending on size), if there are neighbouring pixels with values
        0, otherwise
    """
    img_width = img.shape[1]
    img_height = img.shape[0]

    center_value = img[pixel[1], pixel[0]]
    if not center_value == 0:
        return center_value
    values = []
    for x_offset in range(-size, size + 1):
        for y_offset in range(-size, size + 1):
            x = pixel[0] + x_offset
            y = pixel[1] + y_offset
            if 0 <= x < img_width and 0 <= y < img_height:
                value = img[y, x]
                if not value == 0:
                    values.append(value)
    if len(values) > 0:
        return mean(values)
    else:
        return 0
