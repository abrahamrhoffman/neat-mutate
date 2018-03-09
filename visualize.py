import pygraphviz as pgv
import pandas as pd

class Visualize(object):
    '''
    Mutate Visualization: Using PyGraphViz and Dot Syntax
    '''

    def makeGraph(self, aDataFrameToGraph):
        onlyEnabledDF = aDataFrameToGraph.loc[aDataFrameToGraph['enabled'] == True]
        removeOutputDF = onlyEnabledDF.loc[onlyEnabledDF['type'] != ("output")]

        # Node and Edge Logic
        connectionList = []
        for ix,row in removeOutputDF[['node','in','out']].iterrows():
            # If it is a sensor
            if row['node'] == row['in']:
                connectionList.append(row[['in', 'out']].values.tolist())
            # If it isn't a sensor, ensure both edges are created
            else:
                connectionList.append(row[['in', 'node']].values.tolist())
                connectionList.append(row[['node', 'out']].values.tolist())

        # Segregate the Sensor Nodes
        sensorDF = onlyEnabledDF.loc[onlyEnabledDF['type'] == "sensor"]
        # Create a list of sensor nodes to cluster
        sensorNodes = sensorDF['node'].values.tolist()

        # Segregate the Hidden Nodes
        hiddenDF = onlyEnabledDF.loc[onlyEnabledDF['type'] == "hidden"]
        # Create a list of hidden nodes to cluster
        hiddenNodes = hiddenDF['node'].values.tolist()

        # Segregate the Output Nodes
        outputDF = onlyEnabledDF.loc[onlyEnabledDF['type'] == "output"]
        # Create a list of output nodes to cluster
        outputNodes = outputDF['node'].values.tolist()

        # Create the dot syntax map
        graphString = ("digraph {")
        for ix,ele in enumerate(connectionList):
            graphString += (str(ele[0]) + "->" + str(ele[1]) + ";")
        graphString += ("}")

        # Create the Graph
        G = pgv.AGraph(graphString, strict=False, directed=True, rankdir='LR')
        G.node_attr['shape']='circle'
        G.add_subgraph(sensorNodes, name='cluster_sensors', label="Sensor Nodes", rank="same")
        G.add_subgraph(hiddenNodes, name='cluster_hidden', label="Hidden Nodes")
        G.add_subgraph(outputNodes, name='cluster_output', label="Output Nodes", rank="same")
        imageResult = G.draw(format='png', prog='dot')

        return imageResult
