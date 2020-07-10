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

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

class JointBilateralUpsampleLayer(tf.keras.layers.Layer):
    """ Joint Bilateral Upsample Layer """

    def __init__(self, sigma_spatial, sigma_bilateral, radius=1):
        super(JointBilateralUpsampleLayer, self).__init__()
        self.sigma_spatial = sigma_spatial
        self.sigma_bilateral = sigma_bilateral
        self.radius = radius

        x, y = np.mgrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
        w1 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_spatial ** 2))
        self.w1 = tf.constant(w1, dtype=tf.float32)

    def call(self, src, im):
        batch_size, height, width, src_channels = src.shape
        _, _, _, im_channels = im.shape
        patch_size = self.radius * 2 + 1

        # Color weight
        w = tf.pad(im, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]], mode='reflect')
        w = tf.image.extract_patches(w, sizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        w = tf.reshape(w, shape=[batch_size, height, width, patch_size, patch_size, im_channels])

        # Bilateral weight
        im = tf.reshape(im, shape=[batch_size, height, width, 1, 1, im_channels])
        w -= im
        w = tf.reduce_sum(w ** 2, axis=-1, keepdims=True)
        w = tf.exp(-w / (2 * self.sigma_bilateral ** 2))
        w = w * tf.reshape(self.w1, shape=[1, 1, 1, patch_size, patch_size, 1])

        # Source patches
        src = tf.pad(src, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]], mode='reflect')
        src = tf.image.extract_patches(src, sizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        src = tf.reshape(src, shape=[batch_size, height, width, patch_size, patch_size, src_channels])

        src *= w
        src = tf.reduce_sum(src, axis=[3, 4]) / tf.reduce_sum(w, axis=[3, 4])

        return src
