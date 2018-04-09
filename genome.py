from __future__ import print_function
import pandas as pd
import numpy as np
import random

class Genome(object):
    '''
    Genome (Genotype) creation and modification
    * All counting is zero indexed
    '''

    def __init__(self, data):
        self.data = data

    def create(self):
        '''
        Create a Genome from two sets of genes using the data (X,Y)
        1. Node genes : Defines the number of nodes in the Genome
        2. Connection genes : Defines the connections, weights, and important metadata for the Genome
        3. Return two gene objects as a single Genome (Genotype)
        '''
        X,Y = self.data

        # Each input tensor's column represents a Neural Network Node
        # Ex.: X = np.array([[0,0]]) equals two input nodes
        #      X = np.array([[0,0,0],[0,1,0]]) equals three input nodes
        #      X = np.array([[0,0,0,0],[0,1,0,1],[n,m,o,p],...]) equals four input nodes

        # Count the input (sensor) columns to measure NN input (sensor) nodes
        X_count = X.shape[-1]
        # Count the input (sensor) columns to measure NN output nodes
        Y_count = Y.shape[-1]

        ### Create a Node Gene DataFrame that represents the data (X,Y) ###
        # Create a list of nodes (numerical increments) and their type (input,sensor)
        node_genes_labels = ['node','type']
        # Create another list with the column count of X as the number of nodes (sensor nodes)
        node_genes = [[i, 'sensor'] for i in range(X_count)]
        # Extend the list to include the output nodes (column count from the Y tensor)
        node_genes.extend([i, 'output'] for i in range(Y_count))
        # Create the DataFrame
        nodes = pd.DataFrame.from_records(node_genes, columns=node_genes_labels)
        # Fix the node count to match the actual number of nodes
        nodes['node'] = nodes.index

        ### Create a Connection Gene DataFrame ###
        connection_gene_labels = ['in', 'out', 'weight', 'enabled', 'innovation']
        connections = pd.DataFrame(columns=connection_gene_labels)
        # Collect nodes from nodes df
        node_count = nodes['node'].values.tolist()
        del node_count[-1]
        # Assign metadata to connections
        for i in node_count:
            connections.loc[i] = [i,i,np.random.uniform(-1.0,1.0),True,i]

        ## Ensure initial Sensor 'in' and 'out' metadata is correct
        # Create a list of all the output nodes
        output_nodes = sorted(nodes.loc[nodes['type'] == ("output")]['node'].values.tolist())
        # Calculate an integer for all sensor nodes (to assign an output node)
        sensorOutputLen = len(nodes.loc[nodes.type == ("sensor"),:])
        # Assign the sensor connections a random output node
        connections.loc[nodes.type == ("sensor"),('out')] = np.random.randint(output_nodes[0],output_nodes[-1]+1,size=sensorOutputLen)

        # Create the GENOME Object
        GENOME = nodes,connections
        # Cleanup DataFrames
        del nodes,connections

        return GENOME

    def add_node(self,GENOME):
        '''
        Add a new node to the Genome
        1. Split a connection and and a new node
        2. The in weight for the new node is the original connection's weight
        3. The out weight for the new node is 1.00
        4. Innovation number is incremented by +1
        '''

        nodes,connections = GENOME

        # Create a new hidden node
        nodes.loc[-1] = [len(nodes),("hidden")]
        nodes.reset_index(drop=True,inplace=True)

        ## Split a random enabled connection
        enabled = connections.loc[connections.enabled == True,:]
        sample = enabled.sample().values.tolist()[0]

        # Create 1st Connection
        new_node_num = nodes.iloc[-1].node
        new_conn_A = sample.copy()
        new_conn_A[-1] = len(connections)
        new_conn_A[1] = new_node_num

        # Create 2nd Connection
        new_conn_B = sample.copy()
        new_conn_B[-1] = len(connections)+1
        new_conn_B[2] = 1.00
        new_conn_B[0] = new_conn_A[1]

        # Disable the originally selected connection 'sample'
        connections.loc[connections.innovation == (sample[-1]),("enabled")] = False

        # Append the two new connections to the DataFrame
        connections.loc[-1] = new_conn_A
        connections.reset_index(drop=True,inplace=True)
        connections.loc[-1] = new_conn_B
        connections.reset_index(drop=True,inplace=True)

        GENOME = nodes,connections
        del nodes,connections

        return GENOME

    def add_connection(self,GENOME):
        '''
        Add a new connection to the Genome
        '''

        nodes,connections = GENOME

        # Grab all the enabled and unique in/out nodes
        enabled = connections.loc[connections.enabled == True,:]
        conns = enabled[['in','out']].values
        connList = np.unique(conns).tolist()
        # Randomly choose an in and out target
        inNode = np.random.randint(connList[0],connList[-1]+1)
        outNode = np.random.randint(connList[0],connList[-1]+1)
        # Set Innovation Number, Weight
        innov = connections['innovation'].unique()[-1] + 1
        weight = np.random.uniform(-1.0,1.0)
        # Create the new connection
        connections.loc[-1] = [inNode,outNode,weight,True,innov]
        connections.reset_index(drop=True,inplace=True)

        GENOME = nodes,connections
        del nodes,connections

        return GENOME

    def mutate(self,GENOME):
        '''
        Randomly mutation a Genome (Genotype)
        '''

        aMutation = random.choice(['node', 'connection'])
        if aMutation == ("node"):
            GENOME = self.add_node(GENOME)
        if aMutation == ("connection") :
            GENOME = self.add_connection(GENOME)

        return GENOME
