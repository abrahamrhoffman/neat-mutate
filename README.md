# mutate
Distributed Neuroevolution

This library is a brand-spanking-new implementation of <a href = "http://www.cs.ucf.edu/~kstanley/">Kenneth O. Stanley's</a> neuroevolution algorithm NEAT. NEAT (Neuroevolution of Augmenting Topologies) is a novel method for modelling Neural Networks. The primary benefit of using this library is that it distributes either the model or data (depending on your choice of parallelism).

<i>This library is under active development.</i>

<h3>Development XOR Example</h3>

```
git clone https://github.com/abrahamrhoffman/MUTATE.git
cd MUTATE
python xor.py
...
[[ 0.5       ]
 [ 0.34557512]
 [ 0.4065465 ]
 [ 0.26564977]]
```

Development Features:
- Accept dimensionally arbitrary data as input
- Auto-defines a full Neural Network of nodes and connections (Genome)
- Auto-creates a Neural Network for your data (Phenome)
- Evaluates Neural Network fitness

<h3>Model or Data Parallelism? Your choice.</h3>
...

<h3>Single Node(CPU, GPU or Multi-GPU) [or] Multi-Node(CPU, GPU or Multi-GPU)</h3>
...
