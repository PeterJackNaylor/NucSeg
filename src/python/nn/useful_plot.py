import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.color import hsv2rgb
from skimage import img_as_ubyte


def f(tup):
    return np.expand_dims(np.expand_dims(np.array(tup, dtype="float"), 0), 0)


def inv_f(pix):
    return np.squeeze(pix)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(float(i) / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: inv_f(hsv2rgb(f(c))), hsv))
    np.random.shuffle(colors)
    colors = np.array(colors)
    n, p = colors.shape
    new_colors = np.zeros(shape=(n + 1, p))
    new_colors[1:, :] = colors
    return new_colors


def apply_mask(image, labeled, color, alpha=0.5):
    """Apply the given mask to the image."""
    for i in range(1, labeled.max() + 1):
        for c in range(3):
            image[:, :, c] = np.where(
                labeled == i,
                image[:, :, c] * (1 - alpha) + alpha * color[i, c] * 255,
                image[:, :, c],
            )
    return image


def apply_mask_with_highlighted_borders(image, labeled, color, alpha=0.5):
    """Apply the given mask to the image."""
    for i in range(1, labeled.max() + 1):
        for c in range(3):
            image = add_contours(image, labeled == i, color=color[i])
            image[:, :, c] = np.where(
                labeled == i,
                image[:, :, c] * (1 - alpha) + alpha * color[i, c] * 255,
                image[:, :, c],
            )
    return image


def add_contours(image, label, color=(0, 1, 0)):

    # mask = find_boundaries(label)
    # res = np.array(image).copy()
    # res[mask] = np.array([0, 255, 0])
    res = mark_boundaries(image, label, color=color)
    res = img_as_ubyte(res)
    return res


def coloring_bin(labeled):
    n = labeled.max()
    if n != 0:
        rdm_col_orig = random_colors(n)
        rdm_col = np.array([np.array((0, 0, 0), dtype="int")] + rdm_col_orig)
    else:
        rdm_col = np.array([np.array((0, 0, 0), dtype="int")])
    colored = rdm_col[labeled]

    return colored, rdm_col
