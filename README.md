## mutate
Distributed Neuroevolution <i>v0.1 (alpha)</i>

*v0.2 Introduces several major changes:
- PyTorch completely replaces Tensorflow
- PyTorch to TensorRT for inference. This squashes fp16 to int8 for speedy inference.
- Parquet instead of hdf5
- Ray for distributed job scheduling / logging
- Ray for parallelized phenome execution
- [During refactor, please expect increased instability in the master branch]

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
pip install --user tensorflow pandas numpy tables reprint h5py
git clone https://github.com/abrahamrhoffman/MUTATE.git
cd MUTATE
```

## Alpha Features
- Accept dimensionally arbitrary data as input
- Auto-defines a full Neural Network of nodes and connections (Genome)
- Auto-creates a Neural Network for your data (Phenome)
- Evaluates Neural Network fitness

<i>This library is under active development.</i>

## Change Log
- 08-02-2017: Re-factoring population class to accept genomes as 'jobs' to mutate or kill based on fitness
- 05-30-2017: Full Mutation Commit: Add Node & Add Connection 
- 05-15-2017: Pandas refactor, ops streamlined and pushed to Tensorflow
- 05-07-2017: Single-pass XOR initial commit 
- 05-01-2017: Genome, Phenome, Fitness initial commit
- 04-21-2017: Multi-GPU and distributed Genome design
