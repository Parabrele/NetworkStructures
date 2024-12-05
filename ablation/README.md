# Ablation
-----
This folder contains two main functions :
- node ablation
- edge ablation

These functions are used to run a forward pass on a computational graph representing a model.

In a node ablation setting, the circuit discovery was done by running attribution on the nodes and keeping the most important ones. The resulting computational graph is complete between the kept nodes and can be run in evaluation mode in a single forward pass.

In an edge ablation setting, the circuit discovery was done by running attribution on the edges. We thus get a graph of dependencies between the nodes. Each node at the output of some module requires a different forward pass on the whole module. This approach is more costly.
