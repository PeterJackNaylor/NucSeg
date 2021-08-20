import os
import argparse
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from augmentation import processing_data_functions, AUTO, partial

from skimage.measure import label 
from google.protobuf.descriptor import Error
from sklearn.metrics import accuracy_score, f1_score

import segmentation_models as sm
from training import get_Xy
from dynamic_watershed import post_process
from metric.from_hover import get_fast_aji_plus


def options():
    parser = argparse.ArgumentParser(description="setting up training")
    parser.add_argument("--path", type=str)
    parser.add_argument("--meta", type=str, default="meta.pkl")
    parser.add_argument("--weights", type=str, default="model_weights.h5")
    parser.add_argument("--alpha", type=float, default=5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--param", type=str, required=False)
    parser.add_argument('--aji', dest='aji', action='store_true')
    parser.add_argument('--no_aji', dest='aji', action='store_false')
    args = parser.parse_args()

    if args.param:
        f = open(args.param, "r").read()
        d = dict(x.split("=") for x in f.split(": ")[1].split("; "))
        args.param = d

    args.type = args.param["type"]
    args.model = args.param["model"]
    args.backbone = args.param["backbone"]

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
        raise Error(f"unknown backbone: {args.model}, not implemented")
    args.model_f = model_f

    return args


def load_meta(file_name):
    with open(file_name, "rb") as handle:
        d = pickle.load(handle)
        mean, std = d["mean"], d["std"]
    return mean, std


def setup_data(path, mean, std, backbone, batch_size=1, image_size=224):

    preprocess_input = sm.get_preprocessing(backbone)

    x_val, y_val = get_Xy(path)
    x_val = preprocess_input(x_val)

    y_labeled = np.load(path)["labeled_y"]
    validation_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
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
            ),
            num_parallel_calls=AUTO,
        )
        .prefetch(AUTO)
    )
    if y_labeled.shape[1] != image_size:
        pad = (y_labeled.shape[1] - image_size) // 2
        y_labeled = y_labeled[:, pad:-pad, pad:-pad]
    return validation_ds, y_labeled


def load_model(opt):
    model = opt.model_f(
        opt.backbone,
        classes=opt.classes,
        activation=opt.activation,
        encoder_weights=None
    )
    model.load_weights(opt.weights)
    return model


def main():
    opt = options()
    mean, std = load_meta(opt.meta)
    ds_val, y_labeled = setup_data(opt.path, mean, std, opt.param['backbone'])
    model = load_model(opt)
    pred = model.predict(ds_val)

    # aji computation
    if opt.aji:
        ajis = []
        n = pred.shape[0]
        for i in range(n):
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
            gt_i = y_labeled[i]
            
            ajis.append(get_fast_aji_plus(label(gt_i), pred_i))
        aji = np.mean(ajis)
    # accuracy, f1,

    y_true = (y_labeled > 0).flatten()
    y_pred = (pred > opt.beta).flatten()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    table_training = pd.read_csv("history.csv", index_col=0)
    if opt.type == "binary":
        name_acc_train = "binary_accuracy"
        name_f1_train = "f1-score"
        name_acc_val = "val_binary_accuracy"
        name_f1_val = "val_f1-score"

    elif opt.type == "distance":
        name_acc_train = "accuracy_d"
        name_f1_train = "f1_score_d"
        name_acc_val = "val_accuracy_d"
        name_f1_val = "val_f1_score_d"

    acc_from_history = table_training[name_acc_train].max()
    f1_from_history = table_training[name_f1_train].max()
    val_acc_from_history = table_training[name_acc_val].max()
    val_f1_from_history = table_training[name_f1_val].max()

    dic = {
        "val_acc": acc,
        "val_f1": f1,
        "history_acc": acc_from_history,
        "history_f1": f1_from_history,
        "history_val_acc": val_acc_from_history,
        "history_val_f1": val_f1_from_history,
        "alpha": opt.alpha,
        "beta": opt.beta,
        "weights": os.readlink(opt.weights),
        "meta": os.readlink(opt.meta),
    }

    if opt.aji:
        dic["aji"] = aji

    if opt.type == "binary":
        dic["history_val_auc"] = table_training["val_auc"].max()
        dic["history_val_iou"] = table_training["val_iou_score"].max()
    else:
        dic["history_val_auc"] = np.nan
        dic["history_val_iou"] = np.nan
    if opt.param:
        dic.update(opt.param)

    df = pd.DataFrame(dic, index=[0])
    df.to_csv("score.csv", index=False)


if __name__ == "__main__":
    main()
