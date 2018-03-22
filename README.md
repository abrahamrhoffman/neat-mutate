## mutate
Distributed Neuroevolution : v0.2.0 `[alpha]`

```
### Algorithm features for v2.0.1 ###
- NEAT improvements     : Speciation, Species (adjusted) Fitness, Global Fitness updates
                          http://nn.cs.utexas.edu/?nodine:ugthesis10
- Custom Functions      : Species ablation and honing invented by Abe Hoffman
                          [Pending Medium article for definition and explanation]
- Neuroevolution Tuning : Connection costing for modular and regular neural networks
                          http://www.evolvingai.org/huizinga-mouret-clune-2014-evolving-neural-networks-are
```

This library is a brand-spanking-new implementation of <a href = "http://www.cs.ucf.edu/~kstanley/">Kenneth O. Stanley's</a> neuroevolution algorithm NEAT. <a href = "http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf">NEAT</a> (Neuroevolution of Augmenting Topologies) is a novel method for modeling Neural Networks. Instead of using back-propagation, you simply grow Neural Networks over time. Initially, this may seem like a disorganized way to solve a problem. However, as <a href="https://www.cs.ucf.edu/~kstanley/neat.html">Stanley et. al. demonstrate</a> the fitness function cuts through the dreaded dimensionality curse and solves the problem space quickly.<br>
<br>

## XOR Example
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

## Motivation

The primary benefit of using this neuroevolution library is that it distributes either your model or data (depending on your choice of parallelism). This is a critical feature as it allows dense problem spaces to be traversed quickly.

## Installation

```
# Follow PyTorch installation requirements here: http://pytorch.org/
$ pip install --user torchvision [not needed if already installed]
$ pip install --user pandas numpy fastparquet ray pyarrow minio
$ git clone https://github.com/abrahamrhoffman/mutate.git
$ cd mutate
$ python xor.py
```

## Alpha Features
- Accept dimensionally arbitrary data as input
- Auto-defines a full Neural Network of nodes and connections (Genome)
- Auto-creates a Neural Network for your data (Phenome)
- Evaluates Neural Network fitness
- Grows arbitrary Neural Networks: removes non-performant nns and generates more nns similiar to the most performant

<i>This library is under active development.</i>

## Change Log
- 03-22-2018: (v0.2.1) Algorithm updates in-flight: NEAT improvements, ablation and honing, and connection costing
- 02-01-2018: (v0.2.0) Update complete: Torch, Ray, Object Storage and Parquet.
- 01-28-2018: (v0.2.0) Complete overhaul underway. Please expect instability in the master branch.
- 08-02-2017: (v0.1.5) Re-factoring population class to accept genomes as 'jobs' to mutate or kill based on fitness
- 05-30-2017: (v0.1.4) Full Mutation Commit: Add Node & Add Connection 
- 05-15-2017: (v0.1.1) Pandas refactor, ops streamlined and pushed to Tensorflow
- 05-07-2017: (v0.1.0) Single-pass XOR initial commit 
- 05-01-2017: (v0.0.2) Genome, Phenome, Fitness initial commit
- 04-21-2017: (v0.0.1) Multi-GPU and distributed Genome design
