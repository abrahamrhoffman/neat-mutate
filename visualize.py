import pygraphviz as pgv
import pandas as pd

class Visualize(object):
    '''
    Mutate Visualization: Using PyGraphViz and Dot Syntax
    '''

    def makeGraph(aDataFrameToGraph):
        onlyEnabledDF = aDataFrameToGraph.loc[aDataFrameToGraph['enabled'] == True]

        # Segregate the Sensor Nodes
        sensorDF = onlyEnabledDF.loc[onlyEnabledDF['type'] == "sensor"]
        # Create a list of sensor node connections
        sensorConnections = sensorDF[['node', 'out']].values.tolist()
        # Create a list of sensor nodes to cluster
        sensorNodes = sensorDF['node'].values.tolist()

        # Segregate the Hidden Nodes
        hiddenDF = onlyEnabledDF.loc[onlyEnabledDF['type'] == "hidden"]
        # Create a list of hidden node connections
        hiddenConnections = hiddenDF[['node', 'out']].values.tolist()
        # Create a list of hidden nodes to cluster
        hiddenNodes = hiddenDF['node'].values.tolist()

        # Segregate the Output Nodes
        outputDF = onlyEnabledDF.loc[onlyEnabledDF['type'] == "output"]
        # Create a list of output node connections
        outputConnections = outputDF[['node', 'out']].values.tolist()
        # Create a list of output nodes to cluster
        outputNodes = outputDF['node'].values.tolist()

        # Create the List of ClusterLists
        # Do not include the Output cluster as it would create invalid recursive edges
        aLargeList = []
        aLargeList.extend(sensorConnections)
        aLargeList.extend(hiddenConnections)

        # Create the dot syntax map
        graphString = ("digraph {")
        for ix,ele in enumerate(aLargeList):
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
