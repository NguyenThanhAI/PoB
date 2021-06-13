import imgaug as ia
from imgaug import augmenters as iaa


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
        #sometimes(iaa.Affine(
        #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        #    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        #    rotate=(-45, 45), # rotate by -45 to +45 degrees
        #    shear=(-25, 25), # shear by -16 to +16 degrees
        #    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        #    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        #    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        #)),
        #iaa.Cutout(nb_iterations=(3, 8), squared=False, fill_mode=["gaussian", "constant"], size=(0.15, 0.25), cval=(0, 255), fill_per_channel=True),
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