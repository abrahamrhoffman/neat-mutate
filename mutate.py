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
        #self.population(GENOME, PHENOME)  # Send GENOME and PHENOME to the population pool to kickstart evolution

    def population(self, GENOME, PHENOME):
        h5py.File('population.hdf5')                # Generate or load an HDF5 filestore for the population
        population = pd.HDFStore('population.hdf5') # Load the Filestore

        population['member0'] = GENOME              # Add our initial member to the population
        population['member1'] = GENOME              # Add an identical member to the population

        print(population['member0'])
        print(population['member1'])

        population.close()                          # Close the file

    def add_node(self,df):
        # Select a synapse to split (and disable the connection), then update innovation numbers
        ## Select a connection from the sensor nodes randomly
        potential_mutations = (df['enabled'] == True) & (df['type'] != ('output'))
        nodes_to_split = potential_mutations[potential_mutations == True].index.tolist()
        split = random.choice(nodes_to_split)
        ## Duplicate the node
        df.loc[len(df)] = df.iloc[(split)]
        dup_node_ix = df.iloc[-1].name
        dup_node_out = df.iloc[-1]['out']
        ## Discover how many nodes exist, then add a new one in sequential order
        new_node_num = list(set(df['node'].tolist()))[-1] + 1
        new_node_out = int(df.iloc[dup_node_ix]['in'])
        innov = list(set(df['node'].tolist()))[-1]
        ## Create the new node with weight of (1)
        df.loc[len(df)] = [new_node_num,'hidden',new_node_out,dup_node_out,1,True,(innov + 2)]
        new_node_ix = (df.iloc[-1].name)
        ## Update the duplicated node so it's output is the new node
        df.iloc[dup_node_ix] = df.iloc[dup_node_ix].set_value('out', (new_node_ix))
        ## Disable the original node
        df.iloc[split] = df.iloc[split].set_value('enabled', False)
        ## Update the duplicated node's innovation number
        df.iloc[dup_node_ix] = df.iloc[dup_node_ix].set_value('innovation', (innov + 1))
        return df

def add_connection(self,df):
        # Add a new (non-duplicate) connection to the df
        ## Select an output or hidden node's index as the outbound connection
        potential_mutations = (df['enabled'] == True) & (df['type'] != ('sensor'))
        nodes_to_connect = potential_mutations[potential_mutations == True].index.tolist()
        node_connect_out = random.choice(nodes_to_connect)
        ## Select a sensor or hidden node as the inbound connection
        potential_mutations = (df['enabled'] == True) & (df['type'] != ('output'))
        nodes_to_connect = potential_mutations[potential_mutations == True].index.tolist()
        node_connect_in = random.choice(nodes_to_connect)
        print(node_connect_in)
        print(node_connect_out)
        ## Instantiate the connection gene in the df
        innov = (list(set(df['innovation'].tolist()))[-1])
        ## Create the New Connection
        conn = [df.iloc[node_connect_in]['node'],df.iloc[node_connect_in]['type'],df.iloc[node_connect_in]['node'],df.iloc[node_connect_out]['node'],(np.random.uniform(-1.0,1.0)),True,(innov + 1)]
        ## Check for Duplicates
        not_dup = True
        conn_check_A = list(conn[i] for i in [0,2,3])
        conn_check_B = list(conn[i] for i in [0,3])
        for ix,ser in df[['node','in','out']].iterrows():
            if ser.tolist() == conn_check_A:
                not_dup = False
                print('Duplicate')
        for ix,ser in df[['node', 'out']].iterrows():
            if ser.tolist() == conn_check_B:
                not_dup = False
                print('Duplicate')
        ## If no duplicates, then add the new connection
        if not_dup:
            df.loc[len(df)] = conn
        return df
