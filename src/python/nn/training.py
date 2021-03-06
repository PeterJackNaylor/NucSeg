from google.protobuf.descriptor import Error

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import segmentation_models as sm

from augmentation import setup_datahandler
from metric.dist_metrics import AccuracyDistance, F1_ScoreDistance


def options():
    parser = argparse.ArgumentParser(description="setting up training")
    parser.add_argument("--path_train", type=str)
    parser.add_argument("--path_validation", type=str)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--model", type=str, default="Unet")
    parser.add_argument("--encoder", type=str, default="imagenet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--loss", type=str, default="CE")
    args = parser.parse_args()
    args.encoder = args.encoder if args.encoder == "imagenet" else None
    if args.loss == "CE":
        loss = "binary_crossentropy"
        activation = "sigmoid"
        metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
            "binary_accuracy",
            tf.keras.metrics.AUC()
        ]
        early_stopping_var = "val_f1-score"
    elif args.loss == "focal":
        loss = sm.losses.CategoricalFocalLoss()
        activation = "sigmoid"
        metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
            "binary_accuracy",
            tf.keras.metrics.AUC(),
        ]
        early_stopping_var = "val_f1-score"
    elif args.loss == "mse":
        loss = "mse"
        activation = "relu"
        metrics = ["mse", AccuracyDistance(), F1_ScoreDistance()]
        early_stopping_var = "val_f1_score_d"
    else:
        raise Error("unknown loss, not implemented")
    args.k_loss = loss
    args.classes = 1
    args.activation = activation
    args.metrics = metrics
    args.early_stopping_var = early_stopping_var
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


def get_Xy(path):
    data = np.load(path)
    x, y = data["x"], data["y"]
    # y = tf.keras.utils.to_categorical(y, num_classes=2)
    x = x.astype("uint8")
    y = y.astype("uint8")
    return x, y


def load_data(opts):
    x_t, y_t = get_Xy(opts.path_train)
    x_v, y_v = get_Xy(opts.path_validation)
    return x_t, y_t, x_v, y_v


def main():
    opt = options()

    x_train, y_train, x_val, y_val = load_data(opt)

    image_size = 224
    epochs = opt.epochs

    ds_train, ds_val = setup_datahandler(
        x_train,
        y_train,
        x_val,
        y_val,
        opt.batch_size,
        opt.backbone,
        image_size,
    )
    # define model

    model = opt.model_f(
        opt.backbone,
        classes=opt.classes,
        activation=opt.activation,
        encoder_weights=opt.encoder,
    )

    model = sm.utils.set_regularization(
        model,
        kernel_regularizer=tf.keras.regularizers.l2(opt.weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(opt.weight_decay),
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)

    model.compile(
        optimizer,
        loss=opt.k_loss,
        metrics=opt.metrics,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=epochs / 5,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=epochs / 10, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            './model_weights.h5', save_weights_only=True,
            save_best_only=True, mode='min'
        ),
    ]

    # fit model
    history = model.fit(
        ds_train,
        batch_size=opt.batch_size,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=callbacks,
        max_queue_size=10,
        workers=10,
        use_multiprocessing=True,
        verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    with open("history.csv", mode="w") as f:
        hist_df.to_csv(f)


if __name__ == "__main__":
    main()
