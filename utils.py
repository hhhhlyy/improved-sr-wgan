import tensorflow as tf

from numpy.random import permutation
import numpy as np
def conv2d(x, output_dim, kernel=5, stride=2, stddev=0.02, padding='SAME', name=None, reuse=False):
    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables()
        weights = tf.get_variable(name='weights',
                                  shape=[kernel, kernel, x.get_shape()[-1], output_dim], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, filter=weights, strides=[1, stride, stride, 1], padding=padding)
        biases = tf.get_variable(name='biases', shape=[output_dim],
                                 dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)

    return out

def batch_norm(x, epsilon=1e-5, momentum = 0.999, scale=False, is_training=True,
        name=None, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        return tf.contrib.layers.batch_norm(x, decay=momentum, scale=scale, epsilon=epsilon,
            updates_collections=None, is_training=is_training, scope=name)

def lrelu(x, leak=0.2): #0.01 for dcgan
    return tf.maximum(x, leak*x)

def ful_connect(x, output_size, stddev=0.02, biases_start=0.0, name=None, reuse=False):
    '''Fully connected layer.'''
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        weights = tf.get_variable(name='weights',
                                  shape=[x.get_shape()[1], output_size], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=stddev))

        biases = tf.get_variable(name='biases', shape=[output_size], dtype=tf.float32,
                                 initializer=tf.constant_initializer(biases_start))

    #return tf.nn.xw_plus_b(x, weights, biases)
    return tf.matmul(x, weights) + biases


def downsample(data, method='conv', DIM = 128, K = 4):
    arr = np.zeros([K, K, 3, 3])
    arr[:, :, 0, 0] = 1.0 / (K * K)
    arr[:, :, 1, 1] = 1.0 / (K * K)
    arr[:, :, 2, 2] = 1.0 / (K * K)
    _downsample_weight = tf.constant(arr, dtype=tf.float32)
    data = tf.reshape(data, [-1, 3, DIM, DIM])
    # BCHW -> BHWC
    data = tf.transpose(data, [0, 2, 3, 1])
    if method == 'conv':
        data = tf.nn.conv2d(data, _downsample_weight,
                            strides=[1, K, K, 1], padding='SAME')
    elif method == 'area':
        data = tf.image.resize_area(data, [DIM//K, DIM//K])
    # BHWC -> BCHW
    data = tf.transpose(data, [0, 3, 1, 2])
    data = tf.reshape(data, [-1, 3 * DIM//K * DIM//K])
    return data

class BatchGenerator:
    '''Generator class returning list of indexes at every iteration.'''
    def __init__(self, batch_size, dataset_size):
        self.batch_size   = batch_size
        self.dataset_size = dataset_size

        assert (self.dataset_size > 0)               , 'Dataset is empty.'
        assert (self.dataset_size >= self.batch_size), 'Invalid bathc_size.'
        assert (self.batch_size > 0)                 , 'Invalid bathc_size.'

        self.last_idx = -1
        self.idxs     = permutation(dataset_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_idx + self.batch_size <= self.dataset_size - 1:
            start = self.last_idx + 1
            self.last_idx += self.batch_size

            return self.idxs[start: self.last_idx + 1]

        else:
            if self.last_idx == self.dataset_size - 1:
                raise StopIteration

            start = self.last_idx + 1
            self.last_idx = self.dataset_size - 1

            return self.idxs[start: self.last_idx + 1]
