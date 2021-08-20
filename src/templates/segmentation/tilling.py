#!/usr/bin/env python
"""
Input variables:
    - sample: path of a tif WSI image.
    - model: path of the tissue segmentation file.
Output files:
    - {sample}_mask.npy
"""
import os

import numpy as np
from skimage.transform import resize
from skimage import io

from utils import setup_data, load_model_v2, get_backbone_model
from validation import load_meta # is preppended in the makefile

from useful_wsi import (open_image, get_size, get_x_y_from_0,
                        get_whole_image, get_image, white_percentage,
                        patch_sampling)

from useful_plot import coloring_bin, apply_mask_with_highlighted_borders

from tqdm import tqdm

def check_for_white(img):
    """
    Function to give to wsi_analyse to filter out images that
    are too white. 
    Parameters
    ----------
    img: numpy array corresponding to an rgb image.
    Returns
    -------
    A bool, indicating to keep or remove the img.
    """
    return white_percentage(img, 210, 0.3)

def wsi_analysis(image, model, size, list_roi, opt):
    """
    Tiles a tissue and encodes each tile.
    Parameters
    ----------
    image: string or wsi image, 
        image to tile.
    model: keras model,
        model to encode each tile.
    list_roi: list of list of ints,
        information to tile correctly image.
    Returns
    -------
    Encoded tiles in matrix form. In row the number of tiles 
    and in columns their respective features.
    """

    n = len(list_roi)
    raw = np.zeros(shape=(n, size, size, 3), dtype="uint8")
    for (i, para) in tqdm(enumerate(list_roi), total=n):
        raw[i] = get_image(image, para)
    
    ds = setup_data(raw, opt.mean, opt.std, opt.backbone)
    res = model.predict(ds)
    return res


def tile(image, level, mask, size, mask_level=5):
    """
    Loads a folder of numpy array into a dictionnary.
    Parameters
    ----------
    image: string or wsi image, 
        image to tile.
    level: int,
        level to which apply the analysis.
    mask_level: int,
        level to which apply the mask tissue segmentation.
    Returns
    -------
    A list of parameters corresponding to the tiles in image.
    """
    def load_gt(ignore):
        return mask
    ## Options regarding the mask creationg, which level to apply the function.
    options_applying_mask = {'mask_level': mask_level, 'mask_function': load_gt}

    ## Options regarding the sampling. Method, level, size, if overlapping or not.
    ## You can even use custom functions. Tolerance for the mask segmentation.
    ## allow overlapping is for when the patch doesn't fit in the image, do you want it?
    ## n_samples and with replacement are for the methods random_patch
    options_sampling = {'sampling_method': "grid", 'analyse_level': level, 
                        'patch_size': (size, size), 'overlapping': 0, 
                        'list_func': [check_for_white], 'mask_tolerance': 0.3,
                        'allow_overlapping': False, 'n_samples': 100, 'with_replacement': False}

    roi_options = dict(options_applying_mask, **options_sampling)

    list_roi = patch_sampling(image, **roi_options)  
    return list_roi


def main():
    size = 224
    # Load sample
    slide = open_image("${sample}")
    mask = io.imread("${mask}")[:,:,0]

    original_size = get_whole_image(slide, slide.level_count - 2).shape
    mask = resize(
            mask,
            original_size[:2],
            preserve_range=True,
            anti_aliasing=True,
            order=0
        )

    # Load segmentation_model
    opt = type('', (), {})()
    opt.meta = os.path.join("${model}", "meta.pkl")
    opt.backbone, opt.model = get_backbone_model(os.path.join("${model}", "final_score.csv"))
    opt.weights = os.path.join("${model}", "model_weights.h5")
    opt.mean, opt.std = load_meta(opt.meta)

    model = load_model_v2(opt.backbone, opt.model, opt.weights)

    # Divide and analyse
    list_positions = tile(slide, 0, mask, size, slide.level_count - 2)
    segmented_tiles = wsi_analysis(slide, model, size, list_positions, opt)
    np.savez("segmented_tiles.npz", tiles=segmented_tiles, positions=list_positions)


if __name__ == "__main__":
    main()
