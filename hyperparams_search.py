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

from utils import get_args, get_dataset, get_network


class MyHyperModel(HyperModel):
    def __init__(self, arch, target_height, target_width, num_classes):
        super(MyHyperModel, self).__init__()
        self.arch = arch
        self.target_height = target_height
        self.target_width = target_width
        self.num_classes = num_classes

    def build(self, hp: HyperParameters):
        inputs = keras.Input(shape=(self.target_height, self.target_width, 3))

        preprocessed_inputs, pretrained_model = get_network(inputs=inputs, arch=self.arch, target_height=self.target_height,
                                                            target_width=self.target_width)
        pretrained_model.trainable = False

        x = pretrained_model(preprocessed_inputs, training=False)
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

    arch = args.arch

    memory_fraction = args.memory_fraction

    assert 0 < memory_fraction <= 1.

    physical_devices = tf.config.list_physical_devices("GPU")

    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=5120)])

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

    hyper_model = MyHyperModel(arch=arch, target_height=target_height, target_width=target_width, num_classes=num_classes)

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
