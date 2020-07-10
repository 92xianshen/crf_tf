# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cv2 import cv2
from CRFLayer import CRFLayer

import tensorflow as tf

def unary_from_labels(labels, n_labels, gt_prob, zero_unsure=True):
    """
    Simple classifier that is 50% certain that the annotation is correct.
    (same as in the inference example).


    Parameters
    ----------
    labels: numpy.array
        The label-map, i.e. an array of your data's shape where each unique
        value corresponds to a label.
    n_labels: int
        The total number of labels there are.
        If `zero_unsure` is True (the default), this number should not include
        `0` in counting the labels, since `0` is not a label!
    gt_prob: float
        The certainty of the ground-truth (must be within (0,1)).
    zero_unsure: bool
        If `True`, treat the label value `0` as meaning "could be anything",
        i.e. entries with this value will get uniform unary probability.
        If `False`, do not treat the value `0` specially, but just as any
        other class.
    """
    assert 0 < gt_prob < 1, "`gt_prob must be in (0,1)."

    labels = labels.flatten()

    n_energy = -np.log((1.0 - gt_prob) / (n_labels - 1))
    p_energy = -np.log(gt_prob)

    # Note that the order of the following operations is important.
    # That's because the later ones overwrite part of the former ones, and only
    # after all of them is `U` correct!
    U = np.full((n_labels, len(labels)), n_energy, dtype='float32')
    U[labels - 1 if zero_unsure else labels, np.arange(U.shape[1])] = p_energy

    # Overwrite 0-labels using uniform probability, i.e. "unsure".
    if zero_unsure:
        U[:, labels == 0] = -np.log(1.0 / n_labels)

    return U

class CRFModel(tf.keras.Model):
    def __init__(self, num_classes, theta_alpha, theta_beta, theta_gamma, sigma_spatial, sigma_bilateral, spatial_compat, bilateral_compat, num_iterations):
        super(CRFModel, self).__init__()
        self.crf_layer = CRFLayer(num_classes, theta_alpha, theta_beta, theta_gamma, sigma_spatial, sigma_bilateral, spatial_compat, bilateral_compat, num_iterations)

    def call(self, unary, image):
        Q = self.crf_layer(unary, image)

        return Q

@tf.function
def inference(model, unary, image):
    return model(unary, image)

if __name__ == "__main__":
    image = cv2.imread('examples/im1.png').astype(np.float32)
    image = image.astype(np.float32)
    anno_rgb = cv2.imread('examples/anno1.png').astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + \
        (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    colors, labels = np.unique(anno_lbl, return_inverse=True)

    HAS_UNK = 0 in colors
    if HAS_UNK:
        print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels",
          (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    unary = unary_from_labels(labels, n_labels, 0.7, HAS_UNK)
    unary = np.rollaxis(unary.reshape(n_labels, *image.shape[:2]), 0, 3)
    print(image.max(), image.min(), unary.max(), unary.min())

    model = CRFModel(n_labels, theta_alpha=80., theta_beta=13,
                     theta_gamma=3., sigma_spatial=80., sigma_bilateral=13., spatial_compat=3., bilateral_compat=10., num_iterations=10)
    pred = inference(model, unary[np.newaxis, ...], image[np.newaxis, ...])

    pred = pred.numpy()[0]
    np.savez('pred.npz', pred)
    MAP = np.argmax(pred, axis=-1)
    print(np.unique(MAP, return_counts=True))
    MAP = np.broadcast_to(MAP[..., np.newaxis], image.shape)
    MAP = MAP / MAP.max()
    image = image / image.max()
    plt.imshow(np.hstack([image[..., ::-1], MAP]))
    plt.show()