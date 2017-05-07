from __future__ import print_function
from genome import Genome
from phenome import Phenome
from fitness import Fitness

class NEAT(object):
    '''
    Neuroevolution of Augmenting Topologies
    '''

    def __init__(self, data):
        self.data = data

    def run(self):
        g = Genome(data)            # Instantiate the Genome Class with our XOR Data
        GENOME = g.create()         # Create an indirectly encoded (NEAT) Genome
        p = Phenome(GENOME)         # Instantiate the Phenome Class with our newly minted Genome
        PHENOME = p.create()        # Create a Phenome (Neural Network)
        f = Fitness(data,PHENOME)   # Evaluate Phenome Fitness
        FITNESS = f.evaluate()
        print(FITNESS)      
