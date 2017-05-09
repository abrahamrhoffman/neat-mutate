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
        # Our first order of business is to generate a genome per Ken Stanley's et. al.
        # For simplicity, we define the node genes in a python list.

        # Deprecated the Node Count, since Pandas indexes already. Just gonna use that.
        node_genes_labels = ['type']                          # Define Node Labels
        node_genes = [['sensor'] for i in xrange(1, X_count)] # Generate Sensor Nodes
        for i in xrange(X_count, Y_count):                    # Generate Output Nodes
            node_genes.extend([['output']])

        # Now that the Node genes are built, let's send them to Pandas.
        # This will allow us to ship data frames around in our distributed model
        nodes = pd.DataFrame.from_records(node_genes, columns=node_genes_labels) # Convert Node Genes to DataFrame

        ### Connection Genes ###
        # Define our connection genes in a python list

        # Deprecated the Innovation Number, since this will match up to the node count
        # and is already indexed (starting at 0) by Pandas
        connection_genes_labels = ['in','out','weight','enabled']   # Define Connection Labels
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

        # Now that the Connection genes are built, let's send them to Pandas.
        # This will allow us to ship data frames around in our distributed model
        connections = pd.DataFrame.from_records(connection_genes, columns=connection_genes_labels) # Convert Connection Genes to DataFrame

        return nodes,connections

    def mutate(self, GENOME):
        '''
        Two types of structural mutations: Add Connection or Node
        '''
        nodes,connections = GENOME

        ### Add Connection ###
        #new_connection = {'type'}

        ### Add Node ###
        new_node = {'type': ['hidden']}
        mutate_node = pd.DataFrame(new_node, columns=['type'])
        nodes = nodes.append(mutate_node, ignore_index=True)

        # Adding a New Node, requires that the weight of the connections
        # be updated to 1.

        GENOME = nodes,connections
        return GENOME
