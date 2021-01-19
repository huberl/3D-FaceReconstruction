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
