import pygraphviz as pgv
import pandas as pd

class Visualize(object):
    '''
    Mutate Visualization: Using PyGraphViz and Dot Syntax
    '''

    def create_graph(self, aDataFrameToGraph):
        onlyEnabledDF = aDataFrameToGraph.loc[aDataFrameToGraph['enabled'] == True]
        removeOutputDF = onlyEnabledDF.loc[onlyEnabledDF['type'] != ("output")]

        # Node and Edge Logic
        connectionList = []
        for ix,row in removeOutputDF[['type','node','in','out']].iterrows():
            # If a sensor node
            if row['type'] == ("sensor"):
                connectionList.append(row[['node', 'out']].values.tolist())
            # If a hidden node
            if row['type'] == ("hidden"):
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

        # Remove non-unique nodes and edges
        connectionList = [tuple(ele) for ix,ele in enumerate(connectionList)]
        connectionList = set(connectionList)
        connectionList = [list(ele) for ix,ele in enumerate(connectionList)]

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
