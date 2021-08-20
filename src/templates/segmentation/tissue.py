#!/usr/bin/env python
"""
Input variables:
    - sample: path of a tif WSI image.
    - model: path of the tissue segmentation file.
Output files:
    - {sample}_mask.npy
"""
import os

from utils import prepare_sample, setup_data, load_model_v2, get_backbone_model, plot
from validation import load_meta # is preppended in the makefile

def main():
    # prepare meta data
    meta_path = os.path.join("${model}", "meta.pkl")
    backbone, model = get_backbone_model(os.path.join("${model}", "final_score.csv"))
    weights = os.path.join("${model}", "model_weights.h5")
    mean, std = load_meta(meta_path)

    # prepare sample to predict
    raw, original_size = prepare_sample("${sample}", 224)
    ds = setup_data(raw, mean, std, backbone)

    model = load_model_v2(backbone, model, weights)
    pred = model.predict(ds)
    
    # save segmentations for later check
    name = "${sample}".split(".")[0]
    plot(raw, pred, name, original_size)

if __name__ == "__main__":
    main()
