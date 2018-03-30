import  tensorflow as tf
import yaml

from vgg16 import Vgg16
# from io import EdgeIO
import myIo

class EdgeDetector():
    def __init__(self, config_file,img_place_holder):

        self.io = myIo.EdgeIO()
        self.init = True
        self.img_place_holder = img_place_holder
        try:
            pfile = open(config_file)
            self.cfgs = yaml.load(pfile)
            pfile.close()
        except Exception as err:
            self.init = False
            self.io.print_error('Error reading config file {}, {}'.format(config_file), err)

    def setup(self, sess, reuse=False):
        try:
            with tf.name_scope('vgg16') as scope:
                # if reuse:
                #     scope.reuse_variables()
                self.edge_model = Vgg16(self.cfgs,self.img_place_holder,reuse=reuse)

            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader('/home/yyl/pjs/pycharm-remote-data/HED-BSDS/holy-edge/hed/vgg16/hed-model-5000')
            var_to_shape_map = reader.get_variable_to_shape_map()

            # saver_edge = tf.train.import_meta_graph('/home/yyl/pjs/pycharm-remote-data/HED-BSDS/holy-edge/hed/vgg16/hed-model-5000.meta')

            t_vars = tf.global_variables()
            var_edge = [var for var in t_vars if 'vgg16' in var.name]
            saver_edge = tf.train.Saver(var_edge)
            saver_edge.restore(sess, '/home/yyl/pjs/pycharm-remote-data/HED-BSDS/holy-edge/hed/vgg16/hed-model-5000')
        except Exception as err:
            self.io.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False

    def run(self, sess, img, reuse=False):
        if not self.init:
            return
        self.edge_model.setup_testing(sess)
        # newsess = tf.Session()
        # newsess.run(tf.initialize_all_variables())
        # img_array = newsess.run(img)
        # edgemap = sess.run(self.edge_model.predictions, feed_dict={self.edge_model.images: img_array})
        return self.edge_model.predictions
        #return edgemap





