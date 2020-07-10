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

class GaussianLayer(tf.keras.layers.Layer):
    """ Gaussian layer """

    def __init__(self, theta_gamma):
        super(GaussianLayer, self).__init__()
        self.theta_gamma = theta_gamma
        self.radius = int(theta_gamma * 4)

        x, y = np.mgrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
        k = np.exp(-(x ** 2 + y ** 2) / (2 * theta_gamma ** 2))
        # k = k / np.sum(k)
        self.kernel = tf.constant(k, dtype=tf.float32)

    def call(self, src):
        batch_size, height, width, channels = src.shape

        kernel = tf.repeat(self.kernel[..., tf.newaxis, tf.newaxis], repeats=channels, axis=2)
        src = tf.pad(src, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]], mode='constant', constant_values=0)

        dst = tf.nn.depthwise_conv2d(src, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return dst

# @tf.function
# def filt(gaussian_filter, src):
#     filtered = gaussian_filter(src)
#     return filtered

# if __name__ == "__main__":
#     # a = np.array(Image.open('lena.jpg')).astype(np.float32) / 255.0
#     src = np.array(Image.open('examples/anno1-b.png').convert('RGB')).astype(np.float32) / 255.0
#     gaussian_filter = GaussianLayer(radius=10, theta_gamma=5)
#     filtered = filt(gaussian_filter=gaussian_filter, src=src[np.newaxis])
#     filtered = filtered.numpy()[0]
#     plt.imshow(np.hstack([src, filtered]))
#     plt.show()