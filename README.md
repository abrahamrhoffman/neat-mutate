# mutate
Distributed Neuroevolution <i>v0.1 (alpha)</i>

This library is a brand-spanking-new implementation of <a href = "http://www.cs.ucf.edu/~kstanley/">Kenneth O. Stanley's</a> neuroevolution algorithm NEAT. <a href = "http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf">NEAT</a> (Neuroevolution of Augmenting Topologies) is a novel method for modelling Neural Networks. Instead of using back-propagation, you simply grow Neural Networks over time. Initially, this may seem like a disorganized way to solve a problem. However, as <a href="https://www.cs.ucf.edu/~kstanley/neat.html">Stanley et. al. demonstrate</a> the fitness function cuts through the dreaded dimensionality curse and solves the problem space quickly.<br>
<br>
The primary benefit of using this neuroevolution library is that it distributes either your model or data (depending on your choice of parallelism).<br>

<h3>MUTATE Installation</h3>

```
pip install --user tensorflow pandas numpy tables reprint h5py
git clone https://github.com/abrahamrhoffman/MUTATE.git
cd MUTATE
```

<h3>Development XOR Example</h3>

```
python xor.py
...
Expected:
[[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Result:
[[ 0.5       ]
 [ 0.61023378]
 [ 0.40548778]
 [ 0.51640528]]
Error:
0.505479216576
```

Development Features:
- Accept dimensionally arbitrary data as input
- Auto-defines a full Neural Network of nodes and connections (Genome)
- Auto-creates a Neural Network for your data (Phenome)
- Evaluates Neural Network fitness

<h3>Model or Data Parallelism? Your choice.</h3>
...

<h3>Single Node(CPU, GPU or Multi-GPU) [or] Multi-Node(CPU, GPU or Multi-GPU)</h3>
...<br>
<br>

<i>This library is under active development.</i>

<h4>Change Log</h4>

- 05-15-2017: Pandas refactor, ops streamlined and pushed to Tensorflow
- 05-07-2017: Single-pass XOR initial commit 
- 05-01-2017: Genome, Phenome, Fitness initial commit
- 04-21-2017: Multi-GPU and distributed Genome design
