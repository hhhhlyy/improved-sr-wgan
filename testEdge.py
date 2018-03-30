from edgeDetect import EdgeDetector
from newEdgeDetect import NewEdgeDetector
import tensorflow as tf

def test():
    with tf.Session() as sess:
        real_data = tf.placeholder(tf.float32, [None, 64 * 2,128, 3])
        fake_data = tf.placeholder(tf.float32, [None, 64 * 2,128, 3])

        edge_detector1 = EdgeDetector('configs/hed.yaml',real_data)
        edge_detector1.setup(sess)
        real_edge = edge_detector1.run(sess,real_data)
        edge_detector2 = EdgeDetector('configs/hed.yaml',fake_data)
        edge_detector2.setup(sess)
        fake_edge = edge_detector2.run(sess,fake_data,reuse=True)
