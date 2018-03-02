import pygraphviz as pgv
import pandas as pd

class Visualize(object):
    '''
    Mutate Visualization: Using PyGraphViz and Dot Syntax
    '''

    def makeGraph(self, aDataFrameToGraph):
        onlyEnabledDF = aDataFrameToGraph.loc[aDataFrameToGraph['enabled'] == True]

        outputDF = onlyEnabledDF.loc[onlyEnabledDF['type'] == "output"]
        outputCluster = [row['node'] for ix,row in outputDF[['node', 'out']].iterrows()]

        notOutputDF = onlyEnabledDF.loc[onlyEnabledDF['type'] != "output"]
        hiddenCluster = [row['node'] for ix,row in notOutputDF[['node', 'out']].iterrows()]

        sensorDF = notOutputDF.loc[notOutputDF['type'] == "sensor"]
        sensorCluster = [row['node'] for ix,row in sensorDF[['node', 'out']].iterrows()]

        graphDF = notOutputDF[['node', 'out']]

        graphString = ("digraph {")
        for ix,row in graphDF.iterrows():
            graphString += (str(row['node']) + "->" + str(row['out']) + ";")
        graphString += ("}")

        G = pgv.AGraph(graphString, strict=False, directed=True, rankdir='LR')
        G.node_attr['shape']='circle'
        G.add_subgraph(sensorCluster, name='cluster_sensors', label="Sensor Nodes", rank="same")
        G.add_subgraph(hiddenCluster, name='cluster_hidden', label="Hidden Nodes")#, rank="same")
        G.add_subgraph(outputCluster, name='cluster_output', label="Output Nodes", rank="same")

        return Image(G.draw(format='png', prog='dot'))
