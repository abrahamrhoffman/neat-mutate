from __future__ import print_function
import pandas as pd
import h5py

class Speciate(object):
    '''
    Speciating Genomes
    '''

    def __init__(self, GENOME):
        self.genome = GENOME
