# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2020 Libin Jiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cv2 import cv2

import tensorflow as tf

from BilateralLayer import BilateralLayer
from GaussianLayer import GaussianLayer
from JointBilateralUpsampleLayer import JointBilateralUpsampleLayer

def _diagonal_compatibility(shape):
    return tf.eye(shape[0], shape[1], dtype=np.float32)

def _potts_compatibility(shape):
    return -1 * _diagonal_compatibility(shape)

class CRFLayer(tf.keras.layers.Layer):
    """ A layer implementing Dense CRF """

    def __init__(self, num_classes, theta_alpha, theta_beta, theta_gamma, sigma_spatial, sigma_bilateral, spatial_compat, bilateral_compat, num_iterations):
        super(CRFLayer, self).__init__()
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations

        self.spatial_weights = spatial_compat * \
            _diagonal_compatibility((num_classes, num_classes))
        self.bilateral_weights = bilateral_compat * \
            _diagonal_compatibility((num_classes, num_classes))
        self.compatibility_matrix = _potts_compatibility(
            (num_classes, num_classes))

        self.bilateral = BilateralLayer(theta_alpha, theta_beta)
        self.gaussian = GaussianLayer(theta_gamma)
        self.jbu = JointBilateralUpsampleLayer(sigma_spatial, sigma_bilateral)

    def call(self, unary, image):
        assert len(image.shape) == 4 and len(unary.shape) == 4

        batch_size, height, width, _ = unary.shape
        all_ones = tf.ones((batch_size, height, width, self.num_classes), dtype=tf.float32)

        spatial_norm_vals = self.gaussian(all_ones) - all_ones
        bilateral_norm_vals = self.bilateral(all_ones, image) - all_ones

        # Initialize Q
        Q = tf.nn.softmax(-unary)

        for i in range(self.num_iterations):
            tmp1 = -unary

            # Message passing - spatial
            spatial_out = self.gaussian(Q) - Q
            spatial_out /= spatial_norm_vals

            # Message passing - bilateral
            bilateral_out = self.bilateral(Q, image) - Q
            bilateral_out = self.jbu(bilateral_out, image)
            bilateral_out /= bilateral_norm_vals

            # Message passing
            spatial_out = tf.reshape(spatial_out, [-1, self.num_classes])
            spatial_out = tf.matmul(spatial_out, self.spatial_weights)
            bilateral_out = tf.reshape(bilateral_out, [-1, self.num_classes])
            bilateral_out = tf.matmul(bilateral_out, self.bilateral_weights)
            message_passing = spatial_out + bilateral_out

            # Compatibility transform
            pairwise = tf.matmul(message_passing, self.compatibility_matrix)
            pairwise = tf.reshape(pairwise, [batch_size, height, width, self.num_classes])

            # Local update
            tmp1 -= pairwise

            # Normalize
            Q = tf.nn.softmax(tmp1)

        return Q
