from __future__ import print_function
import sys,os;os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
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
        g = Genome(self.data)           # Instantiate the Genome Class
        GENOME = g.create()             # Create an indirectly encoded (NEAT) Genome
        p = Phenome(GENOME)             # Instantiate the Phenome Class with Genome
        PHENOME = p.create()            # Create a Phenome (Neural Network)
        f = Fitness(self.data,PHENOME)  # Instantiate Phenome Fitness (Tensorflow)
        FITNESS = f.evaluate()          # Evaluate Phenome Fitness
        r = Report(FITNESS)             # Instantiate the Report Class
        REPORT = r.start()              # Print the Report to StdOut
