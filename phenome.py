from __future__ import print_function
import pandas as pd
import numpy as np
import random
import torch

class Phenome(object):
    '''
    A Neural Network Constructor: Accepts a Genome as input
    '''

    def __init__(self, GENOME):
        self.genome = GENOME

    def create(self):
        sensors = self.genome.loc[self.genome['type'] == ('sensor')]           # Select the Sensors
        sensor_weights = sensors['weight'].values                              # Select the weight values
        sensor_weights = sensor_weights.reshape([(sensor_weights.shape[0]),1]) # Reshape the Tensor for MatMul
        weights = torch.from_numpy(sensor_weights)                             # Convert to Torch Tensor
        PHENOME = weights
        return PHENOME
