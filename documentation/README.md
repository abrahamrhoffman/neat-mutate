## Current Work

As always, please contact me directly with any questions or submit an issue here on GitHub. abraham r hoffman [at] gmail dot com

04/20/2018 : [Developer Notes]
- To-date, the master branch has been used to store changes to mutate's core functionality. In an effort to turn mutate into a more developer friendly library, I am pulling the development effort out of the master branch and putting into the development branch.
- Moving forward, all development effort will originate in the development branch and then be merged into the master branch once approved.
- Since work is still being done on the speciation, crossover and adjusted fitness functions, today the master branch will only represent a single forward pass on an dynamic graph.
- Developers will nicely note that the Visualize class now accepts `simple=True/False` for a quick understanding of how the dynamic graphs are represented both in the Phenome expression of the Neural Network and in the Fitness function (executing the graph and measuring species/global fitness).

04/20/2018 : [Algorithm Notes]
- Currently, I am prioritizing speciation, crossover and species (adjusted) fitness functions. These functions are targeted to live in a new `Population` class.
- My mathematical models for species ablation and honing will be implemented immediately following the completion and stabilization of the above core NEAT functions.
- It is on the road map to integrate connection costing and modularization techniques. Please see here for details: http://www.evolvingai.org/huizinga-mouret-clune-2014-evolving-neural-networks-are

04/20/2018 : [Library Notes]
- While I have completed work for using Ray, Minio/AWS S3 (Object Storage), and Df -> Parquet, I am removing this functionality from the master branch at this time. Initially, I spent too much time researching the distributed portion of mutate rather than improving the core NEAT functionality which drives the entire library.
- I plan on integrating local parquet storage after core NEAT function burn-in.
