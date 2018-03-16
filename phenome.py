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
        self.genome = GENOME

    def create(self):
        '''
        1. Mutate's (neural network) layer sorting algorithm
        2. Return a list of all layer weights in Torch Tensor format
        '''

        # Prepare partial dataframes for weight and order selection
        onlyEnabledDF = self.genome.loc[self.genome['enabled'] == True]
        removeSensorDF = onlyEnabledDF.loc[onlyEnabledDF['type'] != ("sensor")]
        removeOutputDF = onlyEnabledDF.loc[onlyEnabledDF['type'] != ("output")]
        weightsDF = removeOutputDF.loc[removeOutputDF['type'] != ("sensor")]

        # Store all target nodes (non-sensor) in list
        nodeWeightList = removeSensorDF['node'].values.tolist()

        # Build the list of nodes and their incoming weights
        aWeightList = []
        for ele in nodeWeightList:
            A = weightsDF.loc[weightsDF['node'] == ele]
            B = weightsDF.loc[weightsDF['out'] == ele]
            C = pd.concat([A,B]).drop_duplicates().reset_index(drop=True)
            aWeightList.append([ele,C['weight'].values.tolist()])

        # Build the Neural Network layers
        sensorNodes = sorted(onlyEnabledDF.loc[onlyEnabledDF['type'] == ("sensor")]['node'].values.tolist())
        outputNodes = sorted(onlyEnabledDF.loc[onlyEnabledDF['type'] == ("output")]['node'].values.tolist())

        # Add the DataFrame Column
        self.genome['layer'] = ""

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

        return self.genome
        # Select the sensor genes
        #sensors = enabledDF.loc[enabledDF['type'] == ('sensor')]
        # Select the sensor weight values
        #sensor_weights = sensors['weight'].values
        # Reshape the Tensor for MatMul
        #sensor_weights = sensor_weights.reshape([(sensor_weights.shape[0]),1])
        # Convert to Torch Tensor
        #sensor_weights = torch.from_numpy(sensor_weights)

        ## Return all layer weights ##
        #PHENOME = []
        #PHENOME.append(sensor_weights)
        #return PHENOME
