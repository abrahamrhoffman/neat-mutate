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

def main():
    data = create_data()
    agenome = initial_genome(data)
    phenome(agenome)

if __name__ == "__main__":
    main()

