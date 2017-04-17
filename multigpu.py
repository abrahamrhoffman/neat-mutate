import tensorflow as tf

class MultiGPU(object):

    def __init__(self, gpus):
        self.gpus = gpus
        self.tower = []

    def initialize_gpus(self):
        for gpu in xrange(self.gpus):

    def model(self):
        for gpu in xrange(self.gpus):
            with tf.device('/gpu:%d' % gpu):
                a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
                b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
                c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
                self.tower.append(tf.matmul(a,b,c))
        with tf.device('/cpu:0'):
          sum = tf.add_n(c)
        sess = tf.Session()
        print sess.run(sum)

def main():
    multigpu = MultiGPU()
    multigpu.model()

if __name__ == "__main__":
    main()
