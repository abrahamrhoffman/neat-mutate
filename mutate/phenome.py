from __future__ import print_function
from collections import OrderedDict
import pandas as pd
import numpy as np
import random
import torch

class Phenome(object):
    '''
    A Neural Network Constructor: Accepts a Genome as input
    '''

    def __init__(self, GENOME):
        # Only use enabled genes
        self.genome = GENOME.loc[GENOME['enabled'] == True]

    def create(self):
        '''
        1. Mutate's (neural network) layer sorting algorithm
        2. Return a list of all layer weights in Torch Tensor format
        '''

        # Only use enabled genes
        #self.genome = self.genome.loc[self.genome['enabled'] == True]

        # Build the Neural Network layers
        sensorNodes = sorted(self.genome.loc[self.genome['type'] == ("sensor")]['node'].values.tolist())
        outputNodes = sorted(self.genome.loc[self.genome['type'] == ("output")]['node'].values.tolist())

        # Add the DataFrame Column
        self.genome.loc[:,'layer'] = ("")

        # Assign sensor nodes to layer 0
        self.genome.loc[self.genome.type == "sensor", "layer"] = ("0")

        # Assign sensor node adjacent nodes as next layer: 1
        for ix,row in self.genome.loc[self.genome['type'] == ("hidden")].iterrows():
            if row['in'] in sensorNodes:
                self.genome.loc[ix, "layer"] = ("1")

        #While nodes are unassigned to layers, loop and add to layers
        while (self.genome.loc[self.genome['type'] == ("hidden")].layer == "").any():
            lastLayerNum = sorted(self.genome['layer'].unique())[-1]
            lastLayerNodes = self.genome.loc[self.genome.layer == str(lastLayerNum)]['node'].unique()
            for ix,row in self.genome.iterrows():
                if row['layer'] == "":
                    if str(row['in']) in str(lastLayerNodes):
                        self.genome.loc[ix, "layer"] = str(int(lastLayerNum)+1)

        # Finally, assign the output layer nodes the final layer number
        lastLayerNum = sorted(self.genome['layer'].unique())[-1]
        for ix,row in self.genome.loc[self.genome['type'] == ("output")].iterrows():
            if row['in'] in outputNodes:
                self.genome.loc[ix, "layer"] = str(int(lastLayerNum)+1)

        # Build a weight list based on the layers
        layers = sorted(self.genome['layer'].unique())
        weightList = []
        for layer in layers:
            weightList.append([self.genome.loc[self.genome['layer'] == (layer)]['weight'].values])
        # Reshape Numpy Arrays as non-zero-dim
        weightList = [ele[0].reshape([(ele[0].shape[0]),1]) for ele in weightList]
        # Convert to Torch Tensors
        weightList = [torch.from_numpy(ele) for ele in weightList]

        ## Return all layer weights ##
        return weightList
