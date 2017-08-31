from __future__ import print_function
import sys,os;os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pandas as pd
from genome import Genome
from phenome import Phenome
from fitness import Fitness
from progress import Progress
from speciation import Speciate

class NEAT(object):
    '''
    Neuroevolution of Augmenting Topologies
    '''

    def __init__(self, data):
        self.data = data

    def run(self):
        #### Phase I : Genome, Phenome, Fitness, duplicate initial Genome, then Mutate ####
        p = Progress()                                # Instantiate the Progress Class
        PROGRESS = p.start()                          # Print the Progress to StdOut
#        g = Genome(self.data)                       # Instantiate the Genome Class
#        GENOME = g.create()                         # Create an indirectly encoded (NEAT) Genome
#        p = Phenome(GENOME)                         # Instantiate the Phenome Class with Genome
#        PHENOME = p.create()                        # Create a Phenome (Neural Network)
#        f = Fitness(self.data,PHENOME)              # Instantiate Phenome Fitness (Tensorflow)
#        FITNESS = f.evaluate()                      # Evaluate Phenome Fitness
#        population = g.duplicate(GENOME)            # Duplicate the initial Genome. This is the Genome created
#        print(population['member0'])
#        member = g.add_node(population['member0'])
#        print(member)
#
#        del p;del f;del r
#
#        p = Phenome(member)
#        PHENOME = p.create()
#        print(PHENOME)
#
#        f = Fitness(self.data,PHENOME)
#        print(f)
        #FITNESS = f.evaluate()
        #r = Report(self.data,FITNESS)
        #REPORT = r.start()
