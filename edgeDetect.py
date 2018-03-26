import  tensorflow as tf
import yaml

from vgg16 import Vgg16
# from io import EdgeIO
import myIo

class EdgeDetector():
    def __init__(self, config_file):

        self.io = myIo.EdgeIO()
        self.init = True
        try:
            pfile = open(config_file)
            self.cfgs = yaml.load(pfile)
            pfile.close()
        except Exception as err:
            self.init = False
            self.io.print_error('Error reading config file {}, {}'.format(config_file), err)

    def setup(self, sess):
        try:
            self.edge_model = Vgg16(self.cfgs)
            saver_edge = tf.train.import_meta_graph('/home/yyl/pjs/pycharm-remote-data/HED-BSDS/holy-edge/hed/models/hed-model-5000.meta')
            saver_edge.restore(sess, '/home/yyl/pjs/pycharm-remote-data/HED-BSDS/holy-edge/hed/models/hed-model-5000')
        except Exception as err:
            self.io.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False

    def run(self, sess, img):
        if not self.init:
            return
        self.edge_model.setup_testing(sess)
        newsess = tf.Session()
        newsess.run(tf.initialize_all_variables())
        img_array = newsess.run(img)
        edgemap = sess.run(self.edge_model.predictions, feed_dict={self.edge_model.images: img_array})

        return edgemap
