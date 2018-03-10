import tensorflow as tf
from os import makedirs
from numpy import array, load
from scipy.misc import imresize
from os.path import exists

from myModel import Model
from utils import BatchGenerator
from utils import downsample

class Trainer(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        print("dictionary here:")
        print(dictionary)
        print('Importing training set ...')
        run_flags = dictionary['flags']
        self.dataset = load(file=run_flags.data, allow_pickle=False)
        self.datasize = self.dataset.shape[0]
        print('Done.')

    def train(self):
        run_flags = self.flags
        sess = self.sess
        sw = self.sw

        real_data = tf.placeholder(tf.float32, [None, run_flags.input_width * run_flags.scale,
                                             run_flags.input_width * run_flags.scale, 3]) #128*128*3 as default
        lr_img = tf.placeholder(tf.float32, [None, run_flags.input_width,
                                             run_flags.input_length, 3])#64*64*3 as default
        myModel = Model(locals())

        fake_data = Model.generative(myModel, lr_img)

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
        differences = tf.reshape(differences,[run_flags.batch_size,128*128*3])
        interpolates = tf.reshape(real_data,[run_flags.batch_size,128*128*3]) + (alpha * differences)
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

        optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=run_flags.lr_gen). \
            minimize(cost_gen, var_list=var_gen)
        optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=run_flags.lr_dis). \
              minimize(cost_dis, var_list=var_dis)
        #optimizer_dis = tf.train.AdamOptimizer(learning_rate=run_flags.lr_dis, beta1=0.5, beta2=0.9). \
         #   minimize(cost_dis, var_list=var_dis)

        init = tf.global_variables_initializer()

        with sess:
            sess.run(init)

            saver = tf.train.Saver()

            if not exists('models'):
                makedirs('models')

            passed_iters = 0

            for epoch in range(1, run_flags.epochs + 1):
                print('Epoch:', str(epoch))

                for batch in BatchGenerator(run_flags.batch_size, self.datasize):
                    batch_hr = self.dataset[batch] / 255.0
                    batch_lr = array([imresize(img, size=(64, 64, 3)) \
                                      for img in batch_hr])

                    if batch.shape[0] != 16:
                        break
                    if passed_iters%5==0:
                        _, gc, dc = sess.run([optimizer_gen, cost_gen, cost_dis],
                                         feed_dict={real_data : batch_hr, lr_img: batch_lr})

                    sess.run([optimizer_dis],
                             feed_dict={real_data : batch_hr, lr_img: batch_lr})

                    passed_iters += 1

                    if passed_iters % run_flags.sample_iter == 0:
                        print('Passed iterations=%d, Generative cost=%.9f, Discriminative cost=%.9f' % \
                              (passed_iters, gc, dc))
                    if passed_iters == 859:
                        print('here!')

                if run_flags.checkpoint_iter and epoch % run_flags.checkpoint_iter == 0:
                    saver.save(sess, '/'.join(['models', run_flags.model, run_flags.model]))

                    print('Model \'%s\' saved in: \'%s/\'' \
                          % (run_flags.model, '/'.join(['models', run_flags.model])))

            print('Optimization finished.')

            saver.save(sess, '/'.join(['models', run_flags.model, run_flags.model]))

            print('Model \'%s\' saved in: \'%s/\'' \
                  % (run_flags.model, '/'.join(['models', run_flags.model])))