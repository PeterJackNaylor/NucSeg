import os
import argparse
import numpy as np
import pandas as pd

from google.protobuf.descriptor import Error
from skimage.io import imsave
from skimage.measure import label 
from sklearn.metrics import accuracy_score, f1_score
from tqdm import trange

from validation import load_meta, load_model, setup_data
from useful_plot import coloring_bin, apply_mask_with_highlighted_borders
import segmentation_models as sm
from dynamic_watershed import post_process
from metric.from_hover import get_fast_aji_plus


def get_max(path, aji=True):
    table = pd.read_csv(path, delimiter=",")
    table = table.fillna(0)

    if aji:
        var = "aji"
        max_var = table[var].max()
        table = table.loc[
            table[var] == max_var,
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
    # options
    parser = argparse.ArgumentParser(description="setting up training")
    parser.add_argument("--path", type=str)
    parser.add_argument("--scores", type=str, default="meta.pkl")
    parser.add_argument('--aji', dest='aji', action='store_true')
    parser.add_argument('--no_aji', dest='aji', action='store_false')
    args = parser.parse_args()

    table = get_max(args.scores, args.aji)
    args.table = table
    args.alpha = table["alpha"]
    args.beta = table["beta"]
    args.model = table["model"]
    args.backbone = table["backbone"]
    args.type = table["type"]
    args.weights = table["weights"]
    args.meta = table["meta"]

    # bring file to working directory
    if not os.path.isfile(os.path.basename(args.weights)):
        os.symlink(args.weights, os.path.basename(args.weights))
    if not os.path.isfile(os.path.basename(args.meta)):
        os.symlink(args.meta, os.path.basename(args.meta))

    if args.type == "binary":
        activation = "sigmoid"
    elif args.type == "distance":
        activation = "relu"
    else:
        raise Error(f"unknown type: {args.type}, not implemented")

    args.activation = activation
    args.classes = 1

    if args.model == "Unet":
        model_f = sm.Unet
    elif args.model == "FPN":
        model_f = sm.FPN
    elif args.model == "Linknet":
        model_f = sm.Linknet
    elif args.model == "PSPNet":
        model_f = sm.PSPNet
    else:
        raise Error(f"unknown model: {args.model}, not implemented")
    args.model_f = model_f

    return args


def plot_and_save(i, rgb, pred_i, gt_i, opt, pred):
    rgb_i = rgb[i]
    if not (rgb_i.shape[:2] == gt_i.shape):
        rgb_i = rgb_i[13:-13, 13:-13]
    name = "samples/id_{:03d}_{}.png"
    imsave(name.format(i, "rgb"), rgb_i)
    if pred_i.max() != 0:
        mask_pred, pred_color = coloring_bin(pred_i)
        rgb_mask_pred = apply_mask_with_highlighted_borders(
                                                rgb_i,
                                                pred_i,
                                                pred_color
                                                )
        imsave(name.format(i, "pred_class"), mask_pred)
        imsave(name.format(i, "pred_class_rgb"), rgb_mask_pred)
    else:
        imsave(name.format(i, "pred_class"), np.zeros_like(rgb_i))
        imsave(name.format(i, "pred_class_rgb"), rgb_i)
    if gt_i.max() != 0:
        mask_gt, gt_color = coloring_bin(gt_i)
        rgb_mask_gt = apply_mask_with_highlighted_borders(
                                                rgb_i,
                                                gt_i,
                                                gt_color
                                                )
        imsave(name.format(i, "gt_class"), (mask_gt * 255).astype("uint8"))
        imsave(name.format(i, "gt_class_rgb"), rgb_mask_gt)
    else:
        imsave(name.format(i, "gt_class"), np.zeros_like(rgb_i))
        imsave(name.format(i, "gt_class_rgb"), rgb_i)
    if opt.type == "binary":
        p_name = name.format(i, "proba")
        raw_out = (pred[i].copy() * 255).astype("uint8")
    else:
        p_name = name.format(i, "distance")
        raw_out = (pred[i].copy()).astype("uint8")
    imsave(p_name, raw_out)


def main():
    opt = options()
    mean, std = load_meta(opt.meta)
    ds_val, y_labeled = setup_data(opt.path, mean, std, opt.backbone)
    rgb = np.load(opt.path)["x"]
    model = load_model(opt)
    pred = model.predict(ds_val)

    # aji computation
    n = pred.shape[0]
    if not os.path.isdir('samples'):
        os.mkdir("samples")

    if opt.aji:
        ajis = []
    for i in trange(n):
        if opt.aji:
            if opt.type == "binary":
                pred_i = post_process(
                    pred[i, :, :, 0],
                    opt.alpha / 255,
                    thresh=opt.beta
                    )
            else:
                pred_i = post_process(
                    pred[i, :, :, 0],
                    opt.alpha,
                    thresh=opt.beta
                    )
        else:
            pred_i = (pred[i, :, :, 0] > opt.beta).astype('uint8')
        gt_i = y_labeled[i]
        plot_and_save(i, rgb, pred_i, gt_i, opt, pred)
        if opt.aji:
            if gt_i.max() == 0 and pred_i.max() == 0:
                ajis.append(1.)
            else:
                ajis.append(get_fast_aji_plus(label(gt_i), pred_i))
    if opt.aji:
        aji = np.mean(ajis)
    # accuracy, f1
    y_true = (y_labeled > 0).flatten()
    y_pred = (pred > opt.beta).flatten()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    df = opt.table
    df["test_acc"] = acc
    df["test_f1"] = f1
    if opt.aji:
        df["test_aji"] = aji
    df.to_csv("final_score.csv")


if __name__ == "__main__":
    main()
