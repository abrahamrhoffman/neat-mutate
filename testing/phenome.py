from __future__ import print_function
import pandas as pd
import numpy as np
import collections

class Phenome(object):
    '''
    A Neural Network Constructor
    - Accepts a Genome as input
    - Parses values and weights into an execution order
    - Returns an operation (Phenome / Phenotype) to be executed in the Fitness class
    '''

    def __init__(self, aGenome):
        # Store the aGenome tuple in DataFrame objects
        self.nodes,self.connections = aGenome
        # Only use enabled connection genes
        self.connections = self.connections.loc[self.connections['enabled'] == True]

    def create(self):
        ## Build layer execution order
        # An ordered dictionary to hold layer node mappings
        layerDict = collections.OrderedDict()
        # Create a list of all the layers in the nodes DataFrame
        layerList = sorted(self.nodes.layer.unique().tolist())
        # Use the list to create
        for ix,ele in enumerate(layerList):
            # Select the nodes in layer ele as a list
            layerN = self.nodes.loc[self.nodes.layer == ele,("node")].values.tolist()
            layerDict[ele] = layerN
        # Sort layers and nodes in 'reverse' layer order: output, layerN[-1], layerN[-2], ..., sensor
        layerDict = collections.OrderedDict(reversed(list(layerDict.items())))

        # Store weights for each node in each layer in an ordered dictionary
        aPhenome = collections.OrderedDict()
        for k,v in layerDict.items():
            aPhenome[k] = []
            # For each node in v in each layer k, how many weights are inbound?
            for aNode in v:
                aPhenome[k].extend(self.connections.loc[self.connections['out'] == aNode]['weight'].values)

        # Sort the dictionary in sensor -> output format and return it
        aPhenome = collections.OrderedDict(reversed(list(aPhenome.items())))

        return aPhenome

