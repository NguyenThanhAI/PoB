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

import imgaug as ia
from imgaug import augmenters as iaa


class SpatialAttentionLayer(keras.layers.Layer):
    def __init__(self):
        super(SpatialAttentionLayer, self).__init__()
        self.gamma = tf.Variable(initial_value=0., shape=[], name="gamma")

    def build(self, input_shape):
        ch = input_shape[-1]
        self.f_conv = tfa.layers.SpectralNormalization(layer=keras.layers.Conv2D(filters=ch // 8, kernel_size=1))
        self.g_conv =  tfa.layers.SpectralNormalization(layer=keras.layers.Conv2D(filters=ch // 8, kernel_size=1))
        self.h_conv = tfa.layers.SpectralNormalization(layer=keras.layers.Conv2D(filters=ch, kernel_size=1))

    def call(self, inputs, **kwargs):
        self.f = self.f_conv(inputs)
        self.g = self.g_conv(inputs)
        self.h = self.h_conv(inputs)

        self.f = tf.reshape(self.f, shape=[tf.shape(self.f)[0], -1, tf.shape(self.f)[-1]])
        self.g = tf.reshape(self.g, shape=[tf.shape(self.g)[0], -1, tf.shape(self.g)[-1]])
        self.h = tf.reshape(self.h, shape=[tf.shape(self.h)[0], -1, tf.shape(self.h)[-1]])

        s = tf.matmul(self.g, self.f, transpose_b=True)

        beta = tf.nn.softmax(s)

        o = tf.matmul(beta, self.h)

        o = tf.reshape(o, shape=tf.shape(inputs))

        output = self.gamma * o + inputs

        return output


sometimes = lambda aug: iaa.Sometimes(0.8, aug)

augment_seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
                    percent=(-0.1, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        #    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        #    rotate=(-45, 45), # rotate by -45 to +45 degrees
        #    shear=(-25, 25), # shear by -16 to +16 degrees
        #    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        #    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        iaa.Cutout(nb_iterations=(0, 3), squared=False, fill_mode=["gaussian", "constant"], size=(0.05, 0.1), cval=(0, 255), fill_per_channel=True),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((5, 10),
            [
        #        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=3), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=3), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 0.2), lightness=(0, 0.8)), # sharpen images
                iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.8)), # emboss images
        #       # search either for all edges or for directed edges,
        #       # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.0, 0.8)),
                    iaa.DirectedEdgeDetect(alpha=(0.0, 0.8), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
        #        #iaa.OneOf([
        #        #    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
        #        #    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        #        #]),
        #        #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.1, 0.2), per_channel=0.2),
        #        iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
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
        #        iaa.LinearContrast((0.1, 0.5), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(1.0, 3.0), sigma=0.25)), # move pixels locally around (with random strengths) # Nên bỏ đi
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.05))),
        #        #sometimes(iaa.Jigsaw(nb_rows=(3, 5), nb_cols=(3, 5)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


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

    parser.add_argument("--arch", type=str, default=None)

    parser.add_argument("--memory_fraction", type=float, default=1.)

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


def get_network(inputs: keras.Input, arch: str, target_height: int, target_width: int):
    inputs = tf.cast(inputs, dtype=tf.float32)
    if arch.lower() == "efficientnetb0":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB0(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "efficientnetb1":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB1(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "efficientnetb2":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB2(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "efficientnetb3":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB3(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "efficientnetb4":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB4(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "efficientnetb5":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB5(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "efficientnetb6":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB6(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "efficientnetb7":
        preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        pretrained_model = keras.applications.EfficientNetB7(input_shape=(target_height, target_width, 3),
                                                             include_top=False, weights="imagenet")
    elif arch.lower() == "resnet50":
        preprocessed_inputs = keras.applications.resnet50.preprocess_input(inputs)
        pretrained_model = keras.applications.ResNet50(input_shape=(target_height, target_width, 3),
                                                       include_top=False, weights="imagenet")
    elif arch.lower() == "resnet101":
        preprocessed_inputs = keras.applications.resnet.preprocess_input(inputs)
        pretrained_model = keras.applications.ResNet101(input_shape=(target_height, target_width, 3),
                                                        include_top=False, weights="imagenet")
    elif arch.lower() == "resnet152":
        preprocessed_inputs = keras.applications.resnet.preprocess_input(inputs)
        pretrained_model = keras.applications.ResNet152(input_shape=(target_height, target_width, 3),
                                                        include_top=False, weights="imagenet")
    elif arch.lower() == "resnet50v2":
        preprocessed_inputs = keras.applications.resnet_v2.preprocess_input(inputs)
        pretrained_model = keras.applications.ResNet50V2(input_shape=(target_height, target_width, 3),
                                                         include_top=False, weights="imagenet")
    elif arch.lower() == "resnet101v2":
        preprocessed_inputs = keras.applications.resnet_v2.preprocess_input(inputs)
        pretrained_model = keras.applications.ResNet101V2(input_shape=(target_height, target_width, 3),
                                                          include_top=False, weights="imagenet")
    elif arch.lower() == "resnet152v2":
        preprocessed_inputs = keras.applications.resnet_v2.preprocess_input(inputs)
        pretrained_model = keras.applications.ResNet152V2(input_shape=(target_height, target_width, 3),
                                                          include_top=False, weights="imagenet")
    elif arch.lower() == "vgg16":
        preprocessed_inputs = keras.applications.vgg16.preprocess_input(inputs)
        pretrained_model = keras.applications.VGG16(input_shape=(target_height, target_width, 3),
                                                    include_top=False, weights="imagenet")
    elif arch.lower() == "vgg19":
        preprocessed_inputs = keras.applications.vgg19.preprocess_input(inputs)
        pretrained_model = keras.applications.VGG19(input_shape=(target_height, target_width, 3),
                                                    include_top=False, weights="imagenet")
    elif arch.lower() == "densenet121":
        preprocessed_inputs = keras.applications.densenet.preprocess_input(inputs)
        pretrained_model = keras.applications.DenseNet121(input_shape=(target_height, target_width, 3),
                                                          include_top=False, weights="imagenet")
    elif arch.lower() == "densenet169":
        preprocessed_inputs = keras.applications.densenet.preprocess_input(inputs)
        pretrained_model = keras.applications.DenseNet169(input_shape=(target_height, target_width, 3),
                                                          include_top=False, weights="imagenet")
    elif arch.lower() == "densenet201":
        preprocessed_inputs = keras.applications.densenet.preprocess_input(inputs)
        pretrained_model = keras.applications.DenseNet201(input_shape=(target_height, target_width, 3),
                                                          include_top=False, weights="imagenet")
    elif arch.lower() == "inceptionv3":
        preprocessed_inputs = keras.applications.inception_v3.preprocess_input(inputs)
        pretrained_model = keras.applications.InceptionV3(input_shape=(target_height, target_width, 3),
                                                          include_top=False, weights="imagenet")
    elif arch.lower() == "inceptionresnetv2":
        preprocessed_inputs = keras.applications.inception_resnet_v2.preprocess_input(inputs)
        pretrained_model = keras.applications.InceptionResNetV2(input_shape=(target_height, target_width, 3),
                                                                include_top=False, weights="imagenet")
    elif arch.lower() == "xception":
        preprocessed_inputs = keras.applications.xception.preprocess_input(inputs)
        pretrained_model = keras.applications.Xception(input_shape=(target_height, target_width, 3),
                                                       include_top=False, weights="imagenet")
    else:
        raise ValueError("Unknown architecture: {}".format(arch.lower()))

    return  preprocessed_inputs, pretrained_model
