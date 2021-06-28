import tensorflow as tf
import numpy as np

import segmentation_models as sm
import albumentations as A

from functools import partial

AUTO = tf.data.AUTOTUNE
P = 0.3


def img_transformer(p=P, size=224, key="train", mean=None, std=None):
    if key == "train":
        transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=p
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=p
                ),
                A.HueSaturationValue(
                    hue_shift_limit=1, sat_shift_limit=20,
                    val_shift_limit=20, p=p
                ),
                A.HorizontalFlip(p=p),
                A.VerticalFlip(p=p),
                A.ElasticTransform(
                    alpha=5, sigma=50, alpha_affine=20, interpolation=1, p=p
                ),
                A.RandomCrop(size, size, always_apply=True),
                A.Normalize(
                    mean=mean / 255.0,
                    std=std / 255.0,
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
            ]
        )
    else:
        transform = A.Compose(
            [
                A.CenterCrop(size, size, always_apply=True),
                A.Normalize(
                    mean=mean / 255.0,
                    std=std / 255.0,
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
            ]
        )
    return transform


def processing_data_functions(p=P, size=224, key="train", mean=None, std=None):
    transform = img_transformer(p, size=size, key=key, mean=mean, std=std)

    def aug_fn(image, label, bs, image_size=size):
        img_aug = np.zeros((bs, image_size, image_size, 3), dtype=np.float32)
        lbl_aug = np.zeros((bs, image_size, image_size), dtype=np.float32)
        for i in range(bs):
            aug_data = transform(image=image[i], mask=label[i])
            img_aug[i] = aug_data["image"]
            lbl_aug[i] = aug_data["mask"]
        return img_aug, lbl_aug

    def process_data(image, label, bs, image_size=size):
        aug_img, label = tf.numpy_function(
            func=aug_fn, inp=[image, label, bs], Tout=[tf.float32, tf.float32]
        )
        return aug_img, label

    return process_data


def setup_datahandler(
    x_train, y_train, x_val, y_val, batch_size,
    backbone, image_size, p=P, auto=AUTO
):
    preprocess_input = sm.get_preprocessing(backbone)
    mean = x_train.mean(axis=(0, 1, 2))
    std = x_train.std(axis=(0, 1, 2))
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)
    train_ds_rand = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size, drop_remainder=True)
        .map(
            partial(
                processing_data_functions(
                    key="train", size=image_size, p=p, mean=mean, std=std
                ),
                batch_size=batch_size,
            ),
            num_parallel_calls=auto,
        )
        .prefetch(auto)
    )

    validation_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size, drop_remainder=True)
        .map(
            partial(
                processing_data_functions(
                    key="train", size=image_size, p=p, mean=mean, std=std
                ),
                batch_size=batch_size,
            ),
            num_parallel_calls=auto,
        )
        .prefetch(auto)
    )
    return train_ds_rand, validation_ds
