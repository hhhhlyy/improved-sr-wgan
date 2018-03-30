import tensorflow as tf

import numpy as np
import random
from time import strftime

from myTrain import Trainer
from myTest  import Tester

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('run', 'train',
                            "choose to run or train.")

tf.app.flags.DEFINE_string('train_model', 'wgangp',
                            "model name.")

tf.app.flags.DEFINE_string('model', 'default',
                            "model name.")

tf.app.flags.DEFINE_integer('batch_size',32,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_integer('epochs', 15,
                            "Number of training epochs")

tf.app.flags.DEFINE_float('lr_gen', 1e-4,
                            "learning rate for generative network")

tf.app.flags.DEFINE_float('lr_dis', 1e-4,
                            "learning rate for discriminative network")

tf.app.flags.DEFINE_float('loss_weight', 0.8,
                            "final loss function = loss_weight*L1_loss + (1-loss_weight)*gen_loss")

tf.app.flags.DEFINE_integer('sample_iter', 10,
                            "display cost per sample-iter iterations")

tf.app.flags.DEFINE_integer('checkpoint_iter', None,
                            "save checkpoints per checkpoint-iter iterations")

tf.app.flags.DEFINE_integer('scale', 2,
                            "image edge size of high-resolution / that of low-resolution")

tf.app.flags.DEFINE_integer('input_width', 64,
                            "input width")

tf.app.flags.DEFINE_integer('input_length', 64,
                            "input length")

tf.app.flags.DEFINE_string('data','/home/yyl/pjs/pycharm-remote/MyGAN/data/train.npy',
                           "data path")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_string('out_path', '/'.join(['test_out_imgs', strftime('%Y%m%d-%H%M%S')]),
                           "output imgs path.")

tf.app.flags.DEFINE_string('edge_detect_model', '/home/yyl/pjs/pycharm-remote-data/HED-BSDS/holy-edge/hed/models/hed-model-5000.meta',
                           "output imgs path.")

def main(argv=None):
    #initialize tensorflow session and prepare for tensorboard
    sess = tf.Session()

    flags = FLAGS
    with sess.graph.as_default():
        tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    sw = tf.summary.FileWriter(flags.checkpoint_dir, sess.graph)

    if FLAGS.run == 'train':
        trainer = Trainer(locals())
        trainer.train()
    else :
        tester = Tester(locals())
        tester.test()
    print("nice!")

if __name__ == '__main__':
    tf.app.run()
