from __future__ import print_function
import tensorflow as tf
import numpy as np

class Phenome(object):
    '''
    A Neural Network Constructor: Accepts a Genome as input
    '''

    def __init__(self, GENOME):
        self.genome = GENOME

    def create(self):
        x_ = tf.placeholder(tf.float32, name="x-input")
        y_ = tf.placeholder(tf.float32, name="y-input")

        sensor = self.genome.loc[self.genome['type'] == ('sensor'),]                                             # All Sensor Nodes
        sensor_weights = self.genome.loc[self.genome['type'] == ('sensor'),]['weight'].values.astype(np.float32) # All Sensor Node Weights
        sensor_weights = sensor_weights.reshape([(sensor_weights.shape[0]),1])                                   # Reshape Tensor for MatMul
        op = tf.sigmoid(tf.matmul(x_, sensor_weights))                                                           # Initial Neural Network (Phenome) for Genome

        PHENOME = x_,y_,op

        return PHENOME
