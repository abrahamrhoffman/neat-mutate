#!/usr/bin/python
from __future__ import print_function
import os;os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from subprocess import Popen
import numpy as np
from neat import NEAT

def create_data():
    '''
    XOR Data
    '''
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    return X,Y

def main():
    data = create_data()
    NEAT(data)
    Popen('rm -f *.pyc',shell=True) # Cleanup after execution

if __name__ == "__main__":
    main()
