import tensorflow as tf
from numpy import load, array, concatenate
from os import makedirs
from scipy.misc import imsave, imresize

from myModel import Model
from utils import BatchGenerator

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

        hr_img = tf.placeholder(tf.float32, [None, run_flags.input_width * run_flags.scale,
                                             run_flags.input_width * run_flags.scale, 3])  # 128*128*3 as default
        lr_img = tf.placeholder(tf.float32, [None, run_flags.input_width,
                                             run_flags.input_length, 3])  # 64*64*3 as default
        myModel = Model(locals())

        out_gen = Model.generative(myModel, lr_img)

        real_out_dis = Model.discriminative(myModel, hr_img)

        fake_out_dis = Model.discriminative(myModel, out_gen, reuse=True)

        cost_gen, cost_dis, var_train, var_gen, var_dis = \
            Model.costs_and_vars(myModel, hr_img, out_gen, real_out_dis, fake_out_dis)
        # cost_gen, cost_dis, var_train, var_gen, var_dis = \
        #     Model.wgan_loss(myModel, hr_img, out_gen, real_out_dis, fake_out_dis)

        init = tf.global_variables_initializer()

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
                batch_sml = array([imresize(img, size=(64, 64, 3)) \
                        for img in batch_big])

                superres_imgs = sess.run(out_gen, feed_dict={lr_img: batch_sml})

                gc, dc  = sess.run([cost_gen, cost_dis], \
                        feed_dict={hr_img : batch_big, lr_img : batch_sml})

                images = concatenate( \
                    ( \
                        array([imresize(img, size=(128, 128, 3)) / 255.0 \
                                for img in batch_sml]), \
                        superres_imgs,
                        batch_big \
                    ), 2)

                for idx, image in enumerate(images):
                    imsave('%s/%d.png' % (run_flags.out_path, start+idx), image)

                start += run_flags.batch_size

                print('%d/%d saved successfully: Generative cost=%.9f, Discriminative cost=%.9f' % \
                        (min(start, self.datasize), self.datasize, gc, dc))