# Computational Graphs

## Project Overview  
This project aims to represent neural networks as computational graphs capturing meaningful informations on the inner algorithms of these networks. This can be used for mechanistic interpretability and analyze the internal structures that emerge during training and by scaling up neural networks. Key innovations include the introduction of **Regions of Interest (ROIs)** for controlled granularity analysis of neural networks - lying between individual neurons and full modules, and a framework for the construction and evaluation of computational graphs.

For more details, see the [project.md](project.md) file.

## Repository structure

This repository contains implementation for building and evaluating representations of neural networks as computational graphs.

You can find your favorite datasets and associated buffers in the `data` folder.

You can find methods for computational graph construction in the `connectivity` folder.

To run forward passes with interventions, see the `ablation` folder.

To evaluate the graph, see the `evaluation` folder.

## Data

Contains functions to load various datasets, general or task specific.

## Connectivity

Contains functions to build computational graphs based on several paradigms, which we will call by their corresponding name in neurosciences as a hope of bridging the gap between the two fields.
- **Structural connectivity** : computational graph built purely based on an analysis of the weights of the neural network.
    - TODO : Nothing here yet.
    - In neurosciences, this would be the anatomical connections between neurons, e.g. axons between individual neurons or regions of interest.
- **Functional connectivity** : Builds the computational graph of the neural network based on the activations (or attribution) of its neurons.
    - TODO : Only ROI related functions are currently here. Nothing yet for computational graphs.
    - Uses the covariance of the activations of the neurons to infer connections between them.
    - This is pretty much the exact same thing as what is done in neurosciences.
- **Effective connectivity** : Builds the computational graph of the neural network based on the causal relationships between its neurons.
    - Uses more expensive attribution methods to infer the causal relationships between neurons.
    - In neurosciences, an estimation of this would be given by time series data.

## Ablation

Contains edge and node ablation.
- edge ablation : run a given computational graph supposed to represent a neural network.
- node ablation : Ablates some nodes during a normal forward pass of a neural network. The effective computational graph is the complete graph between the remaining nodes.

## Evaluation

Contains functions for
- evaluation of the faithfulness of the computational graph to the original model on some input data.
- detection of communities and block structures in the graph using classical methods such as Louvain or spectral clustering as well as less known methods especially in machine learning such as Stochastic Blockmodels (SBM).

## Utils

Contains utility functions and classes used throughout the project.