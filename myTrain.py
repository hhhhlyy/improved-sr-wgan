import tensorflow as tf
from os import makedirs
from numpy import array, load
from scipy.misc import imresize
from os.path import exists
import time

from myModel import Model
from utils import BatchGenerator
from utils import downsample
import plot
from edgeDetect import EdgeDetector


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
        img_size = run_flags.scale*run_flags.input_length

        real_data = tf.placeholder(tf.float32, [None, run_flags.input_width * run_flags.scale,
                                             run_flags.input_width * run_flags.scale, 3]) #128*128*3 as default
        lr_img = tf.placeholder(tf.float32, [None, run_flags.input_width,
                                             run_flags.input_length, 3])#64*64*3 as default
        myModel = Model(locals())

        if run_flags.train_model == 'wgangp':
            # fake_data = Model.generative(myModel, lr_img)
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
            # ########################################
            # with sess:
            #     # sess.run(tf.global_variables_initializer())
            #     # # saver_edge = tf.train.Saver()
            #     # saver_edge = tf.train.import_meta_graph(run_flags.edge_detect_model)
            #     # saver_edge.restore(sess, '/home/yyl/pjs/pycharm-remote-data/HED-BSDS/holy-edge/hed/models/hed-model-5000')
            #     # # saver_edge.restore(sess_edge,run_flags.edge_detect_model)
            #     # # graph_edge = tf.get_default_graph()
            #
            #     #define model
            edge_detector = EdgeDetector('configs/hed.yaml')
            edge_detector.setup(sess)
            real_edge = edge_detector.run(sess,real_data)
            fake_edge = edge_detector.run(sess,fake_data)



            optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=run_flags.lr_gen). \
                minimize(cost_gen, var_list=var_gen)
            optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=run_flags.lr_dis). \
                  minimize(cost_dis, var_list=var_dis)
            #optimizer_dis = tf.train.AdamOptimizer(learning_rate=run_flags.lr_dis, beta1=0.5, beta2=0.9). \
             #   minimize(cost_dis, var_list=var_dis)

        if run_flags.train_model == 'gan':
            fake_data = Model.generative(myModel, lr_img)
            real_out_dis = Model.discriminative_gan(myModel, real_data)
            fake_out_dis = Model.discriminative_gan(myModel, fake_data, reuse=True)
            cost_gen, cost_dis, _,  var_gen, var_dis = Model.costs_and_vars(myModel, real_data, fake_data, real_out_dis, fake_out_dis)
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=run_flags.lr_gen). \
                minimize(cost_gen, var_list=var_gen)
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=run_flags.lr_dis). \
                minimize(cost_dis, var_list=var_dis)

        init = tf.global_variables_initializer()

        with sess:
            sess.run(init)

            saver = tf.train.Saver()

            if not exists('models'):
                makedirs('models')

            passed_iters = 0

            for epoch in range(1, run_flags.epochs + 1):
                print('Epoch:', str(epoch))
                start_time = time.time()

                for batch in BatchGenerator(run_flags.batch_size, self.datasize):
                    batch_hr = self.dataset[batch] / 255.0
                    batch_lr = array([imresize(img, size=(run_flags.input_width, run_flags.input_length, 3)) \
                                      for img in batch_hr])

                    if batch.shape[0] != run_flags.batch_size:
                        break
                    if passed_iters%3==0:
                        _, gc, dc = sess.run([optimizer_gen, cost_gen, cost_dis],
                                         feed_dict={real_data : batch_hr, lr_img: batch_lr})

                    op_gen, dis_gc, dis_dc = sess.run([optimizer_dis,cost_gen, cost_dis],
                             feed_dict={real_data : batch_hr, lr_img: batch_lr})

                    passed_iters += 1

                    if passed_iters % run_flags.sample_iter == 0:
                        print('Passed iterations=%d, Generative cost=%.9f, Discriminative cost=%.9f' % \
                              (passed_iters, gc, dc))
                        plot.plot('train_dis_cost_'+run_flags.train_model, abs(dc))
                        plot.plot('train_gen_cost_'+run_flags.train_model, abs(gc))

                    if (passed_iters < 5) or (passed_iters % 100 == 99):
                        plot.flush()

                    plot.tick()

                if run_flags.checkpoint_iter and epoch % run_flags.checkpoint_iter == 0:
                    saver.save(sess, '/'.join(['models', run_flags.model, run_flags.model]))

                    print('Model \'%s\' saved in: \'%s/\'' \
                          % (run_flags.model, '/'.join(['models', run_flags.model])))

            print('Optimization finished.')

            saver.save(sess, '/'.join(['models', run_flags.model, run_flags.model]))

            print('Model \'%s\' saved in: \'%s/\'' \
                  % (run_flags.model, '/'.join(['models', run_flags.model])))