#!/usr/bin/python
from __future__ import print_function
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np

def create_data():
    '''
    XOR Data
    '''
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    return X,Y

def initial_genome(data):
    '''
    An Initial Neural Network Definition, based on data compatibility
    '''
    node_genes = [[1,'sensor'],[2,'sensor'],[3,'output']]
    connection_genes = [['in',1,'out',3,'weight',0,True,1],
                        ['in',2,'out',3,'weight',0,True,2],
                        ['in',3,'out',3,'weight',0,True,3]]

    return node_genes,connection_genes

def phenome(agenome):
    '''
    A Neural Network Constructor: Accepts a Genome as input
    '''
    node_genes,connection_genes = agenome

    nodes = [tf.placeholder(tf.float32, shape=1, name=(node[1] + '-' + str(node[0]))) for node in node_genes]

    print(nodes)

def fitness():
    pass

def evaluate(data,phenome):
    pass

def main():
    data = create_data()
    agenome = initial_genome(data)
    phenome(agenome)

if __name__ == "__main__":
    main()

