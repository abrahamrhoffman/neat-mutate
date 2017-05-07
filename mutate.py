from __future__ import print_function
import sys
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
        g = Genome(self.data)           # Instantiate the Genome Class with our XOR Data
        GENOME = g.create()             # Create an indirectly encoded (NEAT) Genome
        p = Phenome(GENOME)             # Instantiate the Phenome Class with our newly minted Genome
        PHENOME = p.create()            # Create a Phenome (Neural Network)
        f = Fitness(self.data,PHENOME)  # Evaluate Phenome Fitness
        FITNESS = f.evaluate()

        X,Y = self.data
        result,error,solved = FITNESS

        print('Expected:')
        print('{}'.format(Y))
        print('Result:')
        print('{}'.format(result))
        print('Error:')
        print('{}'.format(error))
