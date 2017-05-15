#!/usr/bin/python
from __future__ import print_function
from subprocess import Popen
import numpy as np
from mutate import NEAT

def main():
    ### Create XOR Data ###
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y = np.array([[0],[1],[1],[0]], dtype=np.float32)
    data = X,Y

    ### Use Mutate's NEAT Algorithm ###
    mutate = NEAT(data)
    mutate.run()
    #results = mutate.run()
    #results.save('xor.model')

    ### (Optional) Clean Generated PYC Files ###
    Popen('rm -f *.pyc',shell=True)

if __name__ == "__main__":
    main()
