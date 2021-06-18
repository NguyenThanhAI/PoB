import os
import argparse

from typing import Tuple

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import cv2

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from utils import SpatialAttentionLayer, get_args, get_dataset, get_network


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

    preprocessed_inputs, pretrained_model = get_network(inputs=inputs, arch=arch, target_height=target_height,
                                                        target_width=target_width)
    pretrained_model.trainable = False

    x = pretrained_model(preprocessed_inputs, training=True)
    x = SpatialAttentionLayer()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.ReLU()(x)
    outputs = keras.layers.Dense(num_classes)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Quan trọng, phải có from_logits=True
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.summary()

    callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "{}_best_model.h5".format(arch)),
                                                 monitor="val_sparse_categorical_accuracy",
                                                 verbose=1, save_best_only=True,
                                                 save_weights_only=True,
                                                 mode="max"),
                 keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", min_delta=0.001,
                                               patience=100, verbose=1, mode="max",
                                               restore_best_weights=True),
                 # keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001,
                 #                                  patience=5, verbose=1, mode="min",
                 #                                  restore_best_weights=True),
                 # keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1,
                 #                                  verbose=1, mode="min", min_lr=1e-6),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_sparse_categorical_accuracy", factor=0.95, patience=1,
                                                   verbose=1, mode="max", min_lr=1e-7)
                 ]

    print("Training the last layer")
    num_epochs = 5
    num_train_steps = num_train // batch_size
    history = model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=num_train_steps, epochs=num_epochs,
                        callbacks=callbacks)

    weights_1 = model.get_weights()

    pretrained_model.trainable = True
    learning_rate = learning_rate / 20
    #learning_rate = tfa.optimizers.ExponentialCyclicalLearningRate(initial_learning_rate=1e-7,
    #                                                               maximal_learning_rate=1e-3,
    #                                                               step_size=float(num_train_steps),
    #                                                               scale_mode="cycle",
    #                                                               gamma=0.95)
    num_epochs = 400

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.summary()

    weights_2 = model.get_weights()

    print("Compare model weight", end="")
    compare = [np.all(np.equal(a, b)) for a, b in zip(weights_1, weights_2)]
    print(compare)
    assert all(compare), "Two weights are not equal"

    print("Finetuning whole network")
    initial_epoch = history.epoch[-1]
    history = model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=num_train_steps, epochs=num_epochs,
                        callbacks=callbacks, initial_epoch=initial_epoch)

    result = model.evaluate(test_dataset)
    result = dict(zip(model.metrics_names, result))

    print("Evaluate on test set: {}".format(result))
