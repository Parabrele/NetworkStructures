# Computational Graphs

This project contains implementation for building and evaluating representations of neural networks as computational graphs.

You can find your favorite dataset and graph building algorithm in the `data` and `connectivity` folders.

To run forward passes on graphs, see the `ablation` folder.

To evaluate graphs, see the `evaluation` folder.

## Data

Contains functions to create, load and prepare various datasets.

## Connectivity

Contains functions to build computational graphs based on several paradigms, which we will call by their corresponding name in neurosciences :
- **Structural connectivity** : Builds the computational graph of the neural network purely based on an analysis of its weights.
    - In neurosciences, this would be the anatomical connections between neurons, e.g. axons between individual neurons or regions of interest.
    - Nothing implemented yet. Future work.
- **Functional connectivity** : Builds the computational graph of the neural network based on the activation patterns of its neurons.
    - Uses the covariance of the activations of the neurons to infer connections between them.
    - This is pretty much the same thing as what is done in neurosciences.
    - In practice we can either use activations patterns (a large vector containing all activations) or "*attribution*" patterns (similarly, a large vector containing all attributions).
- **Effective connectivity** : Builds the computational graph of the neural network based on the causal relationships between its neurons.
    - Uses more expensive attribution methods to infer the causal relationships between neurons.
    - In neurosciences, an estimation of this would be given by time series data

## Ablation

Contains functions to run forward passes on graphs using either node or edge ablation.
- node ablation : In a node ablation setting, the circuit discovery was done by running attribution on the nodes and keeping the most important ones. The resulting computational graph is complete between the kept nodes and can be run in evaluation mode in a single forward pass by ablating all nodes outside of the circuit.
- edge ablation : In an edge ablation setting, the circuit discovery was done by running attribution on the edges. We thus get a graph of dependencies between the nodes. Each node at the output of some module requires it's own forward pass of the whole module. This approach is more costly.

## Evaluation

- `SBM.py` contains functions to fit SBMs to graphs for later study of the block structure.
- `faithfulness.py` contains functions to evaluate the faithfulness and completeness of a computational graph to it's original model. Faithfulness measures how well the graph represents the model - or how sufficient the graph is to explain the model - while completeness measures how necessary to the model is the graph.
