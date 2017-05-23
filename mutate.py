from __future__ import print_function
import sys,os;os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pandas as pd
import h5py
from genome import Genome
from phenome import Phenome
from fitness import Fitness
from report import Report

class NEAT(object):
    '''
    Neuroevolution of Augmenting Topologies
    '''

    def __init__(self, data):
        self.data = data

    def run(self):
        g = Genome(self.data)             # Instantiate the Genome Class
        GENOME = g.create()               # Create an indirectly encoded (NEAT) Genome
        p = Phenome(GENOME)               # Instantiate the Phenome Class with Genome
        PHENOME = p.create()              # Create a Phenome (Neural Network)
        f = Fitness(self.data,PHENOME)    # Instantiate Phenome Fitness (Tensorflow)
        FITNESS = f.evaluate()            # Evaluate Phenome Fitness
        r = Report(self.data,FITNESS)     # Instantiate the Report Class
        REPORT = r.start()                # Print the Report to StdOut
        self.population(GENOME, PHENOME)  # Send GENOME and PHENOME to the population pool to kickstart evolution

    def population(self, GENOME, PHENOME):
        h5py.File('population.hdf5')                # Generate or load an HDF5 filestore for the population
        population = pd.HDFStore('population.hdf5') # Load the Filestore
        population['member0'] = GENOME              # Add our initial member to the population
        population['member1'] = GENOME              # Add an identical member to the population
