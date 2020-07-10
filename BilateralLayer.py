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

class BilateralLayer(tf.keras.layers.Layer):
    """ Bilateral layer """

    def __init__(self, theta_alpha, theta_beta, downsample=4):
        super(BilateralLayer, self).__init__()
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.downsample = downsample

        r1 = int(theta_alpha * 4)
        x, y = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]
        w1 = np.exp(-(x ** 2 + y ** 2) / (2 * theta_alpha ** 2))
        self.radius = int(theta_beta * 4)
        self.w1 = tf.constant(w1[r1 - self.radius:self.radius - r1, r1 - self.radius:self.radius - r1], dtype=tf.float32)

    def call(self, src, im):
        batch_size, height, width, src_channels = src.shape
        _, _, _, im_channels = im.shape
        assert height % self.downsample == 0 and width % self.downsample == 0
        r_height, r_width = height // self.downsample, width // self.downsample
        
        # Downsample
        src = tf.image.resize(src, [r_height, r_width])
        im = tf.image.resize(im, [r_height, r_width])

        patch_size = self.radius * 2 + 1

        # Color weight
        w = tf.pad(im, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]], mode='reflect')
        w = tf.image.extract_patches(w, sizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        w = tf.reshape(w, shape=[batch_size, r_height, r_width, patch_size, patch_size, im_channels])

        # Bilateral weight
        im = tf.reshape(im, shape=[batch_size, r_height, r_width, 1, 1, im_channels])
        w -= im
        w = tf.reduce_sum(w ** 2, axis=-1, keepdims=True)
        w = tf.exp(-w / (2 * self.theta_beta ** 2))
        w = w * tf.reshape(self.w1, shape=[1, 1, 1, patch_size, patch_size, 1])

        # Source patches
        src = tf.pad(src, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]], mode='constant', constant_values=0)
        src = tf.image.extract_patches(src, sizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        src = tf.reshape(src, shape=[batch_size, r_height, r_width, patch_size, patch_size, src_channels])

        src *= w
        src = tf.reduce_sum(src, axis=[3, 4])

        src = tf.image.resize(src, [height, width])

        return src

# @tf.function
# def filt(bilateral_filter, src, im):
#     filtered = bilateral_filter(src=src, im=im)
#     return filtered

# if __name__ == "__main__":
#     # a = np.array(Image.open('lena.jpg')).astype(np.float32) / 255.0
#     src = np.array(Image.open('examples/anno1-b.png').convert('RGB')).astype(np.float32)
#     im = np.array(Image.open('examples/im1.png').convert('RGB')).astype(np.float32)
#     bilateral_filter = BilateralLayer(theta_alpha=80, theta_beta=13)
#     filtered = filt(bilateral_filter=bilateral_filter, src=src[np.newaxis, ...], im=im[np.newaxis, ...])
#     filtered = filtered.numpy()[0].astype(np.uint8)
#     plt.imshow(np.hstack([src, im, filtered]))
#     plt.show()