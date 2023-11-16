## neat-mutate
Distributed Neuroevolution : v0.2.0 (<i>alpha</i>)

```
# Suggested usage in Jupyter Notebook
import mutate

from IPython.display import Image

# Define data
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)
data = X,Y

g = mutate.Genome(data)
GENOME = g.create()
v = mutate.Visualize(GENOME)
Image(v.create()) # Creates a simple visualization of Neural Net layers
#Image(v.create(simple=False)) # Creates a more accurate visualization of Neural Net layers
```

## Description
This library is a brand-spanking-new implementation of <a href = "http://www.cs.ucf.edu/~kstanley/">Kenneth O. Stanley's</a> neuroevolution algorithm NEAT. <a href = "http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf">NEAT</a> (Neuroevolution of Augmenting Topologies) is a novel method for modeling Neural Networks. Instead of using back-propagation, you simply grow Neural Networks over time. Initially, this may seem like a disorganized way to solve a problem. However, as <a href="https://www.cs.ucf.edu/~kstanley/neat.html">Stanley et. al. demonstrate</a> the fitness function cuts through the dreaded dimensionality curse and solves the problem space quickly.<br>

The primary benefit of using this NeuroEvolution library is that it distributes either your dynamic graphs (models) to N number of nodes. This is a critical feature as it allows dense problem spaces to be traversed quickly. To find out more, please read here: https://goo.gl/vtU63o

## Installation
```
# Follow PyTorch installation requirements here: http://pytorch.org/
$ pip install --user torchvision [not needed if already installed]
$ pip install --user pandas numpy fastparquet ray pyarrow minio
$ git clone https://github.com/abrahamrhoffman/mutate.git
```
<i>Work is being done to get all dependencies pushed to PyPi.</i>

## Alpha Features
- Accept dimensionally arbitrary data as input
- Auto-defines a full Neural Network of nodes and connections (Genome)
- Auto-creates a Neural Network for your data (Phenome)
- Evaluates Neural Network fitness
- Grows arbitrary Neural Networks: removes non-performant nns and generates more nns similiar to the most performant

<i>This library is under active development.</i> See <a href="https://github.com/abrahamrhoffman/mutate/blob/master/documentation/changelog.log">change log</a> for details.
