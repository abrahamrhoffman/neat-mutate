# This example is antiquated. New XOR example will be provided in the v0.2.1 release.
# For any questions, please contact Abe Hoffman at: abraham r hoffman [at] gmail dot com

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
    experiment = NEAT(data)
    experiment.run()
    #results = experiment.run()
    #results.save('xor.model')

    ### (Optional) Clean Generated Files ###
    Popen('rm -f *.pyc',shell=True)     # Remove pyc files
    Popen('rm -f *.hdf5',shell=True)    # Remove hdf5 files

if __name__ == "__main__":
    main()
