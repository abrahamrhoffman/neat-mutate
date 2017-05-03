import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

class MultiGPU(object):

    def __init__(self, gpus):
        self.gpus = gpus
        self.tower = []

    def run(self, model, evaluate = ''):
        for gpu in xrange(self.gpus):
            with tf.device('/gpu:%d' % gpu):
                self.tower.append(model)

        with tf.device('/cpu:0'):
            if evaluate == '':
                result = self.tower
            else:
                result = eval(evaluate[0])
        sess = tf.Session()
        return sess.run(result)

