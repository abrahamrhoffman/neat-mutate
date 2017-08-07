from __future__ import print_function
import pandas as pd
import h5py

class Speciate(object):
    '''
    Speciating Genomes
    '''

    def __init__(self, GENOMES):
        self.genomes = GENOMES

    def speciate(self):
        '''
        Speciate a set of GENOMES.
        '''
        genome_count = len(self.genomes)
        print(genome_count)
        print(self.genomes)

    def population(self, GENOME, PHENOME):
        h5py.File('population.hdf5')                # Generate or load an HDF5 filestore for the population
        population = pd.HDFStore('population.hdf5') # Load the Filestore

        population['member0'] = GENOME              # Add our initial member to the population
        population['member1'] = GENOME              # Add an identical member to the population

        print(population['member0'])
        print(population['member1'])

        population.close()                          # Close the file
