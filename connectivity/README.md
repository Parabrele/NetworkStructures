# Connectivity

This folder contains functions to get the computational graph of a neural network with three main distinctions, as in neurosciences:
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