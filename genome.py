from __future__ import print_function
import pandas as pd
import numpy as np

class Genome(object):
    '''
    Neural Network Definition
    '''

    def __init__(self, data):
        self.data = data

    def create(self):
        '''
        Generate a Genome Definition
        '''
        X,Y = self.data

        X_count = (X.shape[-1] + 1)
        Y_count = (X_count + Y.shape[-1])

        node_genes_labels = ['node','type']                                      # Define Node Labels
        node_genes = [[i, 'sensor'] for i in xrange(1, X_count)]                 # Generate Sensor Nodes
        [node_genes.extend([[i, 'output']]) for i in xrange(X_count, Y_count)]   # Generate Output Nodes
        nodes = pd.DataFrame.from_records(node_genes, columns=node_genes_labels) # Convert Nodes to DataFrame
        connection_genes_labels = ['in','out','weight','enabled','innovation']   # Define Connection Labels
        connection_genes = [[i] for i in xrange(1, Y_count)]                     # Generate Input Connections
        [i.extend([j[0]]) for i in connection_genes for j in node_genes if ('output') in j] # Generate Output Connections
        [i.extend([np.random.uniform(-1.0,1.0)]) for i in connection_genes]                 # Generate Connection Weights
        [i.extend([True]) for i in connection_genes]                                        # Enable Initial Connection Genes
        innovation_count = len([j.extend([i+1]) for i,j in enumerate(connection_genes)])    # Generate Innovation Numbers
        connections = pd.DataFrame.from_records(connection_genes, columns=connection_genes_labels) # Convert Connections to Dataframes
        GENOME = pd.concat([nodes,connections], axis=1)                                            # GENOME = Nodes + Connections

        return GENOME

    def mutate(self, GENOME):
        '''
        Two types of structural mutations: Add Connection or Node
        '''
        ### Add Connection ###
        

        ### Add Node ###

        return GENOME
