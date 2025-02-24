# Computational Graphs

## Project Overview  
This project aims to represent neural networks as computational graphs capturing meaningful informations on the inner algorithms of these networks. This can be used for mechanistic interpretability and analises of the internal structures that emerge during training and by scaling up neural networks. Key innovations include the introduction of **Regions of Interest (ROIs)** for controlled granularity analysis of neural networks - lying between individual neurons and full modules, and a framework for the construction and evaluation of computational graphs.

For more details, see the [project.md](project.md) file.

## Installation

Install all dependencies in the [requirements.txt](requirements.txt) file. Some require to use conda forge, while others can be installed with pip.

## Repository structure

This repository contains implementation for building and evaluating representations of neural networks as computational graphs.

- See the [minimal_example.ipynb](minimal_example.ipynb) notebook for a quick start.
- Tasks and Datasets:
    - You can find your favorite task datasets and associated buffers in [data/buffer.py](data/buffer.py).

- Computational Graphs:
    - Computational graph is constructed with functions in [src/attribution.py](src/attribution.py).
    - To run forward passes with interventions, see [src/ablation.py](src/ablation.py).
    - To evaluate graphs, see [src/evaluation.py](src/evaluation.py).

- ROIs:
    - To discover ROIs, see [src/roi.py](src/roi.py).
    - To evaluate ROIs, see [src/evaluation.py](src/evaluation.py).

<!--->

## TODOs
just a draft containing random thoughts and todos. Most of them are already outdated.

## URGENT :
- Avg Cossim inside ROIs :
    - if ~ 1 : partitioning features is just ~ to training a smaller SAE
    - otherwise : "atomic functions" don't have to be 1D, and maybe that's what we find. Is there a better way to do it ?
- in load_saes, load them on model's layer's device
- deal with multi GPU
- (not su urgent : just use resid SAEs) Look into pre/post ln : e.g. gemma seems to use ln(module(ln(resid))). More generally consider that SAEs might be attached to different points.
    - When loading Gemma SAEs, do E <- E * O^-1 and D <- D * O
        - What about when O is not invertible ?
            - Alors un sous espace de l'espace avant O est inutile et peut être supprimé. Comment le faire proprement ? C'est encore un truc que les entraineurs de SAE doivent faire et pas moi...

## ROIs :
- find and save the ROIs for any dictionary
    - extract activation & attribution vector from modules
    - get cov
    - save cov
    - fit sbm
    - save sbm
    - use sbm to define ROI class
    - save ROIs

## getting the graphs :
- support ROIs as nodes

## evaluating the graphs :
- metric & faithfulness for
    - do it for one seq f(s)
    - for all s, do f(s), and average (see if on average I can find a graph responsible for one output individually)
    - f(all s) (see if I can find a graph responsible for all outputs)

## node vs edge :
- replicate marks debias bias in bios
- maybe do a mixture of tasks : circuit for both gender and job classification, ablate gender node
- show model unable to use gender : the node responsible for gender is gone
- do edge ablation instead
- show model still able to use gender : the node is still there but not connected to one specific task.

## Head partitioning :
- a posteriori partitioning
- attribute head to feature on many examples and aggregate.
- show stats on the partitioning : % of features per head, ambiguity per feature (if it has multiple heads roughly equally), etc.
- for the plot roi vs head comme dans la these de je sais plus qui, pour que l'ordre et l'alignement soit visuellement gentil, trier les features par tête puis par ROI à droite, et seulement par tête à gauche.

## paper :
- restructure the paper :
- method :
    - partitioning : mention that we will be interested in three partitions refered to as discrete/features, head and ROIs and develop on head and ROIs.
        - order ? put after circuit evaluation ?
    - circuit extraction
    - circuit evaluation (node vs edge ablation here ?)
- experiments :
    - ROIs
    - Heads
    - Overlap head - ROI for attn layers
    - circuit evaluation :
        - node vs edge marks vs me which is best
            - metrics vs graph size
            - metric vs degree/density -> degree might be clearer, density might show that feature circuits really are more sparse but that's it.
            - intervention : controling model behavior
        - partition which is best
            - same
            - do it in appendix ? No, this is also super important, but it might seem a bit redundant and take a lot of space
    - bias in bios : node vs edge

## ROI attribution :

divide by N = # of feature in ROI ? Otherwise, inner product will be naturally higher for larger ROIs.
Sparsity of SAEs : this might not be required as only a small number of features will be active at a time. If this number, is typically smaller than the size of ROIs, the problem will not appear, and dividing by N will actualy penalize larger ROIs.

## Misc :
- Check that masked graphs edge ablation with threshold >> 1 has no connection (not even res - res)
- Do a complete run step by step in a notebook to check that everything behaves as expected on the right objects.
- Better visualisations
    - for node/edge ablation : with clean, patch/baseline and patch forwards
    - for attribution : forward pass with only relevant edges impacted
- if no embed SAE : do get_circuit with a graph that excludes embed, then for inference manually set the embed SAE to Identity and all edges 1... Not satisfying : later layer likely do not use the original embeding informations. Alternatively, start with a resid SAE from layer 0 (skip layer 1).

<!--->

<!--->
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
<!--->
