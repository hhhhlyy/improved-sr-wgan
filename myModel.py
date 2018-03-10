import tensorflow as tf

from utils import ful_connect
from utils import batch_norm
from utils import  conv2d
from utils import lrelu

class Model:
    ''' to do for wgan:
       * no batch_norm layers
       * weights set to 0~c
       * loss function
       * no sigmoid layers in discriminative network'''
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def generative(self, x, reuse = False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            if self.run_flags.run == 'train':
                is_training = True
            else:
                is_training = False

            conv1 = lrelu(batch_norm(conv2d(x, output_dim=32, stride=1, name='g_conv1'), \
                                     is_training=is_training, name='g_conv1_bn')) # 64 x 64 x 32

            conv2 = lrelu(batch_norm(conv2d(conv1, output_dim=128, stride=1, name='g_conv2'), \
                                     is_training=is_training, name='g_conv2_bn')) # 64 x 64 x 128

            conv3 = lrelu(batch_norm(conv2d(conv2, output_dim=128, stride=1, name='g_conv3'), \
                                     is_training=is_training, name='g_conv3_bn')) # 64 x 64 x 128

            conv3_up = tf.image.resize_images(conv3, size=[128, 128])

            conv4 = lrelu(batch_norm(conv2d(conv3_up, output_dim=128, stride=1, name='g_conv4'), \
                                     is_training=is_training, name='g_conv4_bn')) # 128 x 128 x 128

            conv5 = lrelu(batch_norm(conv2d(conv4, output_dim=64, stride=1, name='g_conv5'), \
                                     is_training=is_training, name='g_conv5_bn'))  # 128 x 128 x 64

            conv6 = tf.nn.sigmoid(conv2d(conv5, output_dim=3, stride=1, name='g_conv6')) #128 x 128 x 3

        return conv6

    def discriminative(self, images, reuse = False):
        with tf.variable_scope('discriminative') as scope:
            if reuse:
                scope.reuse_variables()

            if self.run_flags.run == 'train':
                is_training = True
            else:
                is_training = False

            output = tf.reshape(images,[-1,3,64,64], name='d_reshape')

            output = conv2d(output,64, 5, 2, name='d_conv1')
            output = lrelu(output)

            output = conv2d(output, 2*64, 5, 2, name='d_conv2')
            output = lrelu(output)

            output = conv2d(output, 4*64, 5, 2, name='d_conv3')
            output = lrelu(output)

            output = tf.reshape(output, [-1, 4 * 4 * 4 * 64])
            output = ful_connect(output, 1, name='d_output')

        return tf.reshape(output, [-1])

    def costs_and_vars(self, real, generated, real_disc, gener_disc, is_training=True):
        '''Return generative and discriminator networks\' costs,
        and variables to optimize them if is_training=True.'''
        d_real_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_disc,
                                                                             labels=tf.ones_like(real_disc)))
        d_gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc,
                                                                            labels=tf.zeros_like(gener_disc)))

        g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc,
                                                                        labels=tf.ones_like(gener_disc))) * 0.1 + \
                 tf.reduce_mean(tf.abs(tf.subtract(generated, real)))

        d_cost = d_real_cost + d_gen_cost

        if is_training:
            t_vars = tf.trainable_variables()

            d_vars = [var for var in t_vars if 'd_' in var.name]
            g_vars = [var for var in t_vars if 'g_' in var.name]

            return g_cost, d_cost, t_vars, g_vars, d_vars

        else:
            return g_cost, d_cost

    def wgan_loss(self, real, generated, real_disc, gener_disc, is_training=True):
        '''wgan loss function'''
        d_cost = tf.reduce_mean(gener_disc) - tf.reduce_mean(real_disc)

        g_cost = tf.reduce_mean(gener_disc)

        if is_training:
            t_vars = tf.trainable_variables()

            d_vars = [var for var in t_vars if 'd_' in var.name]
            g_vars = [var for var in t_vars if 'g_' in var.name]

            return g_cost, d_cost, t_vars, g_vars, d_vars

        else:
            return g_cost, d_cost

