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
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

from utils import augment_seq


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
    img = tf.image.resize(img, size=(target_height, target_width), method=tf.image.ResizeMethod.BICUBIC)
    img = tf.cast(img, dtype=tf.uint8)
    return img


def process_path(file_path, target_height=256, target_width=256):
    img = tf.io.read_file(file_path)
    img = decode_img(img, target_height=target_height, target_width=target_width)
    return img


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


class MyHyperModel(HyperModel):
    def __init__(self, target_height, target_width, num_classes):
        super(MyHyperModel, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.num_classes = num_classes

    def build(self, hp: HyperParameters):
        inputs = keras.Input(shape=(self.target_height, self.target_width, 3))

        pretrained_model = keras.applications.EfficientNetB3(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
        pretrained_model.trainable = False

        x = pretrained_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self.num_classes)(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Quan trọng, phải có from_logits=True
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )

        model.summary()

        return model


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

    hyper_model = MyHyperModel(target_height=target_height, target_width=target_width, num_classes=num_classes)

    tuner = RandomSearch(hypermodel=hyper_model,
                         objective="val_sparse_categorical_accuracy",
                         max_trials=5,
                         executions_per_trial=3)

    tuner.search_space_summary()

    num_train_steps = num_train // batch_size

    tuner.search(train_dataset,
                 epochs=5,
                 validation_data=val_dataset,
                 steps_per_epoch=num_train_steps)

    tuner.results_summary()
