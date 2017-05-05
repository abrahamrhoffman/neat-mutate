from __future__ import print_function
import pandas as pd

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

        ### Node Genes ###

        node_genes_labels = ['node','type']                      # Define Node Labels
        node_genes = [[i, 'sensor'] for i in xrange(1, X_count)] # Generate Sensor Nodes
        for i in xrange(X_count, Y_count):                       # Generate Output Nodes
            node_genes.extend([[i, 'output']])
        nodes = pd.DataFrame.from_records(node_genes, columns=node_genes_labels) # Convert Nodes to DataFrame

        ### Connection Genes ###

        connection_genes_labels = ['in','out','weight','enabled','innovation']   # Define Connection Labels
        connection_genes = []
        for i in xrange(1, Y_count):        # Generate Input Connections
            connection_genes.extend([[i]])
        for i in connection_genes:          # Generate Output Connections
            for j in node_genes:
                if ('output') in j:
                   i.extend([j[0]])
        for i in connection_genes:          # Generate Connection Weights (start at 0, fully initialized in phenome)
            i.extend([0])
        for i in connection_genes:          # Enable all initial connection Genes
            i.extend([True])
        innovation_count = 1                # Generate connection gene innovation numbers
        for i in connection_genes:
            i.extend([innovation_count])
            innovation_count += 1

        connections = pd.DataFrame.from_records(connection_genes, columns=connection_genes_labels)

        return nodes,connections
