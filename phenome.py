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
        op = tf.sigmoid(tf.reduce_mean(tf.multiply(x_, sensor_weights), 1))                                       # Initial Neural Network (Phenome) for Genome

        PHENOME = x_,y_,op

        return PHENOME
