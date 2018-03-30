import tensorflow as tf
import numpy as np
from numpy import load, array, concatenate, mean
from os import makedirs
from scipy.misc import imsave, imresize

from myModel import Model
from utils import BatchGenerator
from utils import downsample
import psnr
import scipy.ndimage
from edgeDetect import EdgeDetector
from newEdgeDetect import NewEdgeDetector
class Tester(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        print("dictionary here:")
        print(dictionary)
        print('Importing testing set ...')
        run_flags = dictionary['flags']
        self.dataset = load(file=run_flags.data, allow_pickle=False)
        self.datasize = self.dataset.shape[0]
        print('Done.')

    def test(self):
        run_flags = self.flags
        sess = self.sess
        sw = self.sw
        img_size = run_flags.scale * run_flags.input_length

        real_data = tf.placeholder(tf.float32, [None, run_flags.input_width * run_flags.scale,
                                             run_flags.input_width * run_flags.scale, 3]) #128*128*3 as default
        lr_img = tf.placeholder(tf.float32, [None, run_flags.input_width,
                                             run_flags.input_length, 3])#64*64*3 as default
        myModel = Model(locals())

        if run_flags.train_model == 'wgangp':
            fake_data = Model.generative_res(myModel, lr_img)
            real_out_dis = Model.discriminative(myModel, real_data)
            fake_out_dis = Model.discriminative(myModel, fake_data, reuse=True)
            t_vars = tf.trainable_variables()
            var_gen = [var for var in t_vars if 'g_' in var.name]
            var_dis = [var for var in t_vars if 'd_' in var.name]
            cost_gen = -tf.reduce_mean(fake_out_dis)
            cost_dis = tf.reduce_mean(fake_out_dis) - tf.reduce_mean(real_out_dis)
            alpha = tf.random_uniform(
                shape=[run_flags.batch_size, 1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            differences = tf.reshape(differences,[run_flags.batch_size,img_size*img_size*3])
            interpolates = tf.reshape(real_data,[run_flags.batch_size,img_size*img_size*3]) + (alpha * differences)
            gradients = tf.gradients(Model.discriminative(myModel,interpolates,reuse=True), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            cost_dis += 10 * gradient_penalty

            # add L1 difference to penalty
            fake_data_downsampled = downsample(fake_data)
            real_data_downsampled = downsample(real_data)
            gen_l1_cost = tf.reduce_mean(
                tf.abs(fake_data_downsampled - real_data_downsampled))
            cost_gen = run_flags.loss_weight * gen_l1_cost + (1-run_flags.loss_weight) * cost_gen

            # #edge detection
            edge_detector1 = EdgeDetector('configs/hed.yaml', real_data)
            edge_detector1.setup(sess)
            real_edge = edge_detector1.run(sess, real_data)

            edge_detector2 = NewEdgeDetector('configs/hed.yaml', fake_data)
            edge_detector2.setup(sess)
            fake_edge = edge_detector2.run(sess, fake_data, reuse=True)

            real_edge_downsampled = downsample(real_edge)
            fake_edge_downsampled = downsample(fake_edge)
            edge_cost = tf.reduce_mean(tf.abs(fake_edge_downsampled - real_edge_downsampled))

            cost_gen = 0.8 * cost_gen + edge_cost * 20
            optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=run_flags.lr_gen). \
                minimize(cost_gen, var_list=var_gen)
            optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=run_flags.lr_dis). \
                  minimize(cost_dis, var_list=var_dis)
            #optimizer_dis = tf.train.AdamOptimizer(learning_rate=run_flags.lr_dis, beta1=0.5, beta2=0.9). \
             #   minimize(cost_dis, var_list=var_dis)

        if run_flags.train_model == 'gan':
            fake_data = Model.generative_res(myModel, lr_img)
            real_out_dis = Model.discriminative_gan(myModel, real_data)
            fake_out_dis = Model.discriminative_gan(myModel, fake_data, reuse=True)
            cost_gen, cost_dis, _,  var_gen, var_dis = Model.costs_and_vars(myModel, real_data, fake_data, real_out_dis, fake_out_dis)
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=run_flags.lr_gen). \
                minimize(cost_gen, var_list=var_gen)
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=run_flags.lr_dis). \
                minimize(cost_dis, var_list=var_dis)

        # init = tf.global_variables_initializer()
        var_all = tf.global_variables()
        var_gan = [var for var in var_all if 'g_' in var.name or 'd_' in var.name]
        init = tf.variables_initializer(var_gan)
        with tf.Session() as sess:
            sess.run(init)

            saver = tf.train.Saver()

            try:
                saver.restore(sess, '/'.join(['models', run_flags.model, run_flags.model]))
            except:
                print('Model coult not be restored. Exiting.')
                exit()

            makedirs(run_flags.out_path)

            print('Saving test results ...')

            start = 0

            for batch in BatchGenerator(run_flags.batch_size, self.datasize):
                batch_big = self.dataset[batch] / 255.0
                batch_sml = array([imresize(img, size=(run_flags.input_width, run_flags.input_length, 3)) \
                        for img in batch_big])


                if batch.shape[0] != run_flags.batch_size:
                    break

                superres_imgs  = sess.run(fake_data, feed_dict={lr_img: batch_sml})

                gc, dc , re, fe = sess.run([cost_gen, cost_dis, real_edge, fake_edge], \
                        feed_dict={real_data : batch_big, lr_img : batch_sml})

                for i,img in enumerate(superres_imgs):
                    img = img *255
                    img = img.astype(int)
                    img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
                    img_sobel = scipy.ndimage.filters.sobel(img_gray)
                    imsave('%s/%d_edge_fake.png' % (run_flags.out_path, start+i), img_sobel)

                for i,img in enumerate(batch_big):
                    img = img *255
                    img = img.astype(int)
                    img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
                    img_sobel = scipy.ndimage.filters.sobel(img_gray)
                    imsave('%s/%d_edge_real.png' % (run_flags.out_path, start+i), img_sobel)

                for i,img in enumerate(batch_sml):
                    img = imresize(img, size=(img_size, img_size, 3))
                    img = img *255
                    img = img.astype(int)
                    img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
                    img_sobel = scipy.ndimage.filters.sobel(img_gray)
                    imsave('%s/%d_edge_low.png' % (run_flags.out_path, start+i), img_sobel)

                images = concatenate( \
                    ( \
                        array([imresize(img, size=(img_size, img_size, 3)) / 255.0 for img in batch_sml]), \
                        superres_imgs,
                        batch_big
                    ), 2)

                criteria_psnr = mean(array(
                    [psnr.psnr(sr_img * 255, real_img * 255) for sr_img in superres_imgs for real_img in batch_big]))
                bicubic_criteria_psnr = mean(array(
                    [psnr.psnr(imresize(lr_img, size=(img_size, img_size, 3))* 255, real_img * 255) for lr_img in batch_sml for real_img in batch_big]))

                for idx, image in enumerate(images):
                    imsave('%s/%d.png' % (run_flags.out_path, start+idx), image)

                start += run_flags.batch_size

                print('%d/%d saved successfully: Generative cost=%.9f, Discriminative cost=%.9f, psnr=%.9f, bicubic_psnr = %.9f' % \
                        (min(start, self.datasize), self.datasize, gc, dc, criteria_psnr, bicubic_criteria_psnr))

