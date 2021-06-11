import os
import argparse

from typing import Tuple

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import cv2

import tensorflow as tf
from tensorflow import keras

import imgaug as ia
from imgaug import augmenters as iaa


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default="Painting")
    parser.add_argument("--prefix_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_height", type=int, default=256)
    parser.add_argument("--target_width", type=int, default=256)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    args = parser.parse_args()

    return args


def decode_img(img, target_height=256, target_width=256):
    img = tf.image.decode_jpeg(img, channels=3)
    img_shape = tf.shape(img)
    img_height = img_shape[0]
    img_width = img_shape[1]
    target_height_const = tf.constant(target_height)
    target_width_const = tf.constant(target_width)
    img = tf.cond(tf.logical_and(tf.equal(img_height, target_height_const),
                                 tf.equal(img_width, target_width_const)),
                  lambda: img,
                  lambda: tf.cast(tf.image.resize(img, size=(target_height, target_width), method=tf.image.ResizeMethod.BICUBIC), dtype=tf.uint8))
    #img = tf.cast(img, dtype=tf.uint8)
    return img


def process_path(file_path, target_height=256, target_width=256):
    img = tf.io.read_file(file_path)
    img = decode_img(img, target_height=target_height, target_width=target_width)
    return img


sometimes = lambda aug: iaa.Sometimes(0.8, aug)

augment_seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
                    percent=(-0.3, 0.3),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-25, 25), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        #iaa.Cutout(nb_iterations=(3, 8), squared=False, fill_mode=["gaussian", "constant"], size=(0.15, 0.25), cval=(0, 255), fill_per_channel=True),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        #iaa.SomeOf((2, 7),
        #    [
        #        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
        #        iaa.OneOf([
        #            iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
        #            iaa.AverageBlur(k=3), # blur image using local means with kernel sizes between 2 and 7
        #            iaa.MedianBlur(k=3), # blur image using local medians with kernel sizes between 2 and 7
        #        ]),
        #        #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        #        #iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.5)), # emboss images
        #        # search either for all edges or for directed edges,
        #        # blend the result with the original image using a blobby mask
        #        #iaa.SimplexNoiseAlpha(iaa.OneOf([
        #        #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
        #        #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
        #        #])),
        #        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.5), # add gaussian noise to images
        #        #iaa.OneOf([
        #        #    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
        #        #    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        #        #]),
        #        #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.1, 0.2), per_channel=0.2),
        #        #iaa.Invert(0.05, per_channel=True), # invert color channels
        #        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
        #        #iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
        #        # either change the brightness of the whole image (sometimes
        #        # per channel) or change the brightness of subareas
        #        iaa.OneOf([
        #            iaa.Multiply((0.5, 1.5), per_channel=0.5),
        #            iaa.FrequencyNoiseAlpha(
        #                exponent=(-4, 0),
        #                first=iaa.Multiply((0.5, 1.5), per_channel=True),
        #                second=iaa.LinearContrast((0.5, 2.0))
        #            )
        #        ]),
        #        #iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
        #        iaa.Grayscale(alpha=(0.0, 1.0)),
        #        #sometimes(iaa.ElasticTransformation(alpha=(2.0, 10.0), sigma=0.25)), # move pixels locally around (with random strengths) # Nên bỏ đi
        #        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        #        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        #        #sometimes(iaa.Jigsaw(nb_rows=(3, 5), nb_cols=(3, 5)))
        #    ],
        #    random_order=True
        #)
    ],
    random_order=True
)


def augment_fn(image: tf.Tensor, label: tf.Tensor):
  image = tf.numpy_function(augment_seq.augment_images,
                            [image],
                            image.dtype)
  image = tf.cast(image, dtype=tf.uint8)
  #image = keras.applications.resnet50.preprocess_input(image) # Với EfficientNet phải comment dòng này
  return image, label


def get_dataset(csv_path: str, data_dir: str, batch_size: int, target_height: int, target_width: int, is_training=True) -> Tuple[tf.data.Dataset, int]:
    df = pd.read_csv(csv_path)
    images_list = df["image_id"].to_list()
    images_list = list(map(lambda x: os.path.join(data_dir, x), images_list))
    labels_list = df["class"].to_numpy() - 1 # Huhu, quên trừ đi 1
    images_data = tf.data.Dataset.from_tensor_slices(images_list)
    labels_data = tf.data.Dataset.from_tensor_slices(labels_list)
    dataset = tf.data.Dataset.zip((images_data, labels_data))
    num_elements = dataset.cardinality().numpy()
    dataset = dataset.cache()
    dataset = dataset.map(lambda image, label: (process_path(image, target_height=target_height, target_width=target_width), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(2048)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if is_training:
        dataset = dataset.map(lambda image, label: augment_fn(image=image, label=label))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, num_elements


if __name__ == '__main__':
    args = get_args()

    print("Arguments: {}".format(args))

    prefix = args.prefix
    prefix_dir = args.prefix_dir
    training_csv = os.path.join(prefix_dir, prefix + "_train.csv")
    validation_csv = os.path.join(prefix_dir, prefix + "_val.csv")
    test_csv = os.path.join(prefix_dir, prefix + "_test.csv")
    data_dir = os.path.join(args.data_dir, prefix)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.prefix)

    batch_size = args.batch_size
    target_height = args.target_height
    target_width = args.target_width
    learning_rate = args.learning_rate

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    training = pd.read_csv(training_csv)
    training_images = training["image_id"].to_list()
    training_labels = training["class"].to_numpy()

    num_classes = np.max(training_labels)
    print("Number of classes {}".format(num_classes))

    train_dataset, num_train = get_dataset(csv_path=training_csv, data_dir=data_dir, target_height=target_height,
                                           target_width=target_width, batch_size=batch_size, is_training=True)
    val_dataset, num_val = get_dataset(csv_path=validation_csv, data_dir=data_dir, target_height=target_height,
                                       target_width=target_width, batch_size=batch_size, is_training=False)
    test_dataset, num_test = get_dataset(csv_path=test_csv, data_dir=data_dir, target_height=target_height,
                                         target_width=target_width, batch_size=batch_size, is_training=False)

    print("Train: {}, Val: {}, Test: {}".format(num_train, num_val, num_test))

    inputs = keras.Input(shape=(target_height, target_width, 3))

    pretrained_model = keras.applications.EfficientNetB5(input_shape=(target_height, target_width, 3),
                                                         include_top=False, weights="imagenet")
    pretrained_model.trainable = False

    x = pretrained_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Quan trọng, phải có from_logits=True
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.summary()

    callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "best_model.h5"),
                                                 monitor="val_sparse_categorical_accuracy",
                                                 verbose=1, save_best_only=True,
                                                 save_weights_only=True,
                                                 mode="max"),
                 keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", min_delta=0.001,
                                               patience=20, verbose=1, mode="max",
                                               restore_best_weights=True),
                 # keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001,
                 #                                  patience=5, verbose=1, mode="min",
                 #                                  restore_best_weights=True),
                 # keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1,
                 #                                  verbose=1, mode="min", min_lr=1e-6),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_sparse_categorical_accuracy", factor=0.8, patience=1,
                                                   verbose=1, mode="max", min_lr=1e-7)]

    print("Training the last layer")
    num_epochs = 5
    num_train_steps = num_train // batch_size
    history = model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=num_train_steps, epochs=num_epochs,
                        callbacks=callbacks)

    pretrained_model.trainable = True
    learning_rate = learning_rate / 20
    num_epochs = 100

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.summary()

    print("Finetuning whole network")
    initial_epoch = history.epoch[-1]
    history = model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=num_train_steps, epochs=num_epochs,
                        callbacks=callbacks, initial_epoch=initial_epoch)

    result = model.evaluate(test_dataset)
    result = dict(zip(model.metrics_names, result))

    print("Evaluate on test set: {}".format(result))
