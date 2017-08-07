from __future__ import print_function
import pandas as pd
import h5py

class Speciate(object):
    '''
    Speciating Genomes
    '''

    def __init__(self, GENOME):
        self.genome = GENOME

    def duplicate(self):
        '''
        Speciate a set of GENOMES.
        '''
        h5py.File('population.hdf5')                # Generate or load an HDF5 filestore for the population
        population = pd.HDFStore('population.hdf5') # Load the Filestore

        population['member0'] = self.genome         # Add our initial member to the population
        population['member1'] = self.genome         # Add an identical member to the population

        print(population['member0'])
        print(population['member1'])

        population.close()                          # Close the file
