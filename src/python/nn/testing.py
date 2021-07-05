import os
import argparse
import numpy as np
import pandas as pd
from validation import load_meta, load_model, setup_data

from google.protobuf.descriptor import Error
from skimage.io import imsave
from useful_plot import coloring_bin, apply_mask_with_highlighted_borders

import segmentation_models as sm
from dynamic_watershed import post_process
from metric.object_metrics import aji_fast
from sklearn.metrics import accuracy_score, f1_score


def get_max(path):
    table = pd.read_csv(path, delimiter=",")
    max_aji = table["aji"].max()
    table = table.loc[
        table["aji"] == max_aji,
    ]
    max_f1 = table["val_f1"].max()
    table = table.loc[
        table["val_f1"] == max_f1,
    ]
    max_acc = table["val_acc"].max()
    table = table.loc[
        table["val_acc"] == max_acc,
    ]
    table = table.loc[table.index[0]]
    return table


def options():
    parser = argparse.ArgumentParser(description="setting up training")
    parser.add_argument("--path", type=str)
    parser.add_argument("--scores", type=str, default="meta.pkl")
    args = parser.parse_args()

    table = get_max(args.scores)
    args.table = table
    args.alpha = table["alpha"]
    args.beta = table["beta"]
    args.model = table["model"]
    args.backbone = table["backbone"]
    args.type = table["type"]
    args.weights = table["weights"]
    args.meta = table["meta"]

    os.symlink(args.weights, os.path.basename(args.weights))
    os.symlink(args.meta, os.path.basename(args.meta))

    if args.type == "binary":
        activation = "sigmoid"
    elif args.type == "distance":
        activation = "relu"
    else:
        raise Error(f"unknown type: {args.type}, not implemented")

    args.activation = activation
    args.classes = 1

    if args.backbone == "Unet":
        model_f = sm.Unet
    elif args.backbone == "FPN":
        model_f = sm.FPN
    elif args.backbone == "Linknet":
        model_f = sm.Linknet
    elif args.backbone == "PSPNet":
        model_f = sm.PSPNet
    else:
        raise Error(f"unknown backbone: {args.backbone}, not implemented")
    args.model_f = model_f

    return args

def plot_and_save(i, rgb, pred_i, gt_i, opt, pred):
    rgb_i = rgb[i, 13:-13, 13:-13]
    name = "samples/id_{:03d}_{}.png"
    
    imsave(name.format(i, "rgb"), rgb_i)
    if pred_i.max() != 0:
        mask_pred, pred_color = coloring_bin(pred_i)
        rgb_mask_pred = apply_mask_with_highlighted_borders(rgb_i, pred_i, pred_color)
        imsave(name.format(i, "pred_class"), mask_pred)
        imsave(name.format(i, "pred_class_rgb"), rgb_mask_pred)
    else:
        imsave(name.format(i, "pred_class"), np.zeros_like(rgb_i))
        imsave(name.format(i, "pred_class_rgb"), rgb_i)
    if gt_i.max() != 0:
        mask_gt, gt_color = coloring_bin(gt_i)
        rgb_mask_gt = apply_mask_with_highlighted_borders(rgb_i, gt_i, gt_color)
        imsave(name.format(i, "gt_class"), (mask_gt*255).astype('uint8'))
        imsave(name.format(i, "gt_class_rgb"), rgb_mask_gt)
    else:
        imsave(name.format(i, "gt_class"), np.zeros_like(rgb_i))
        imsave(name.format(i, "gt_class_rgb"), rgb_i)
    if opt.type == 'binary':
        p_name = name.format(i, "proba")
        raw_out = (pred[i].copy() * 255).astype('uint8')
    else:
        p_name = name.format(i, "distance")
        raw_out = (pred[i].copy()).astype('uint8')
    imsave(p_name, raw_out)

def main():
    opt = options()
    mean, std = load_meta(opt.meta)
    ds_val, y_labeled = setup_data(opt.path, mean, std)
    rgb = np.load(opt.path)["x"]
    model = load_model(opt)
    pred = model.predict(ds_val)

    # aji computation
    ajis = []
    n = pred.shape[0]
    os.mkdir("samples")
    import pdb; pdb.set_trace()
    for i in range(n):
        if opt.type == 'binary':
            pred_i = post_process(pred[i,:,:,0], opt.alpha / 255, thresh=opt.beta)
        else:
            pred_i = post_process(pred[i,:,:,0], opt.alpha, thresh=opt.beta)
        gt_i = y_labeled[i]
        plot_and_save(i, rgb, pred_i, gt_i, opt, pred)
        ajis.append(aji_fast(gt_i, pred_i))
    aji = np.mean(ajis)
    # accuracy, f1

    y_true = (y_labeled > 0).flatten()
    y_pred = (pred > opt.beta).flatten()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    df = opt.table
    df["test_acc"] = acc
    df["test_f1"] = f1
    df["test_aji"] = aji
    df.to_csv("final_score.csv")


if __name__ == "__main__":
    main()
