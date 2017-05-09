from __future__ import print_function
import tensorflow as tf

class Phenome(object):
    '''
    A Neural Network Constructor: Accepts a Genome as input
    '''

    def __init__(self, GENOME):
        self.genome = GENOME

    def create(self):
        nodes,connections = self.genome

        x_ = tf.placeholder(tf.float32, name="x-input")
        y_ = tf.placeholder(tf.float32, name="y-input")

        input_count = 0
        for i in nodes.loc[:,['type']].values.tolist():
            if ('sensor') in i:
                input_count += 1

        output_count = 0
        for i in nodes.loc[:,['type']].values.tolist():
            if ('output') in i:
                output_count += 1

        weights_dim = connections['weight'].shape[0]
        weights = [i for i in connections['weight']]

        weight = tf.constant(weights,shape=[weights_dim,connections['weight'].shape[1]],dtype=tf.float32)

        #weight = tf.Variable(tf.random_uniform([(input_count),(output_count)], -1, 1), name="Weight")
        bias = tf.Variable(tf.zeros([(output_count)]), name="Bias")
        activation = tf.sigmoid(tf.matmul(x_, weight) + bias)

        return x_,y_,weight,bias,activation
