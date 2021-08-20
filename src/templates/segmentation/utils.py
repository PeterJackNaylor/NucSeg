
import numpy as np
import tensorflow as tf

import segmentation_models as sm
from useful_wsi.utils import open_image, get_whole_image
from skimage.transform import resize
from skimage import io

from augmentation import processing_data_functions, partial
from validation import load_meta, load_model
from useful_plot import coloring_bin, apply_mask_with_highlighted_borders

def get_backbone_model(path):
    with open(path) as f:
        contents = f.readlines()
        for line in contents:
            if "backbone" in line:
                backbone = line[:-1].split(',')[1]
            if "model," in line:
                model = line[:-1].split(',')[1]
    return backbone, model

def load_model_v2(backbone, model, weights):
    opt = type('', (), {})() #empty object
    opt.backbone = backbone

    if model == "Unet":
        model_f = sm.Unet
    elif model == "FPN":
        model_f = sm.FPN
    elif model == "Linknet":
        model_f = sm.Linknet
    elif model == "PSPNet":
        model_f = sm.PSPNet
    else:
        raise ValueError(f"unknown model: {model}")
    opt.model_f = model_f    
    
    opt.classes = 1 #because binary accuracy
    opt.activation = "relu"
    opt.weights = weights

    return load_model(opt)

def prepare_sample(path, size):
    wsi = open_image(path)
    raw = get_whole_image(wsi, level=wsi.level_count - 2)
    raw = resize(
            raw,
            (size, size),
            preserve_range=True,
            anti_aliasing=True,
            order=0
        )
    raw = np.expand_dims(raw, axis=0)
    return raw, raw.shape[1]

def setup_data(x, mean, std, backbone, batch_size=1, image_size=224):

    preprocess_input = sm.get_preprocessing(backbone)

    x = preprocess_input(x.copy())
    fake_y = np.zeros_like(x)[:,:,:,0]

    ds = (
        tf.data.Dataset.from_tensor_slices((x, fake_y))
        .batch(batch_size, drop_remainder=True)
        .map(
            partial(
                processing_data_functions(
                    key="validation",
                    size=image_size,
                    p=None,
                    mean=mean,
                    std=std
                ),
                bs=batch_size,
            )
        )
    )
    return ds

def plot(img, prob, name, original_size):
    img = img[0]
    prob = prob.copy()[0,:,:,0]
    img = resize(
            img,
            (original_size, original_size, 3),
            preserve_range=True,
            anti_aliasing=True,
        )
    prob = resize(
            prob,
            (original_size, original_size),
            preserve_range=True,
            anti_aliasing=True,
        )
    img = img.astype('uint8')
    pred = (prob > 0.5).astype('uint8')
    mask_pred, pred_color = coloring_bin(pred)
    rgb_mask_pred = apply_mask_with_highlighted_borders(
                                                    img,
                                                    pred,
                                                    pred_color
                                                    )
    fname = name + "_{}.png"
    io.imsave(fname.format("_img"), img)
    io.imsave(fname.format("_overlay"), rgb_mask_pred)
    io.imsave(fname.format("mask"), mask_pred)
    io.imsave(fname.format("prob"), prob)
