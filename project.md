# Emerging Structures in Computational Graphs of Neural Networks

This project explores **computational graphs** of neural networks to better understand the emergence of structures and functionalities in their internal mechanisms. Below, I outline the design and implementation of the project, highlighting how I developed methods, achieved milestones, and how the original objective helped me stay on track and decide which directions to take.

## Contribution statement

This project was the sole focus of my master's thesis, where I independently designed, executed, and analyzed all aspects of the work. From the initial problem formulation to the final evaluation, I was responsible for every step, including the development of novel methods for extracting and analyzing computational graphs of neural networks. My supervisor provided guidance on the scientific direction of the project and feedback on the writing process, but all technical work, design decisions, and implementations were my responsibility.

## Problem Definition 

The project started with the overarching goal of studying *the emergence of structures in neural networks* (NNs) *through training and scaling.* This ambitious goal was not only intellectually stimulating but also practically relevant for understanding the inner workings of neural networks and improving their interpretability.

## Project Execution and Workflow

The original plan was to quickly address the issue of representing NNs as computational graphs to dedicate most of my time to the analysis of these graphs and the underlying graph-theoretical aspects of this problem, which were of deeper fundamental interest to me. I however quickly realized that the extraction of these graphs was a non-trivial problem that would require significant investment to solve.

As a result, I did not solve the original problem, with all of my master thesis' time spent on the extraction of these graphs. I thus defined the following sub-objective as the first chapter in the journey towards the original goal: representing neural networks as computational graphs, with meaningful properties suchas
- functional segregation - i.e. functionally separable units belong to different communities,
- causal relationships between nodes - i.e. subgraphs or communities correspond to *circuits* in the mech interp vocabulary.

To achieve this, I decomposed the project into manageable steps:
1. **Construct graphs**:
   I explored various approaches for constructing computational graphs, including structural, functional, and effective connectivity. Through fast iterations of experimentation, I refined my focus and determined that effective connectivity offered the most relevant insights for my analysis.
2. **Define computational nodes**:
   - After settling on effective connectivity, I needed to define the units of analysis, or nodes I would consider for constructing the computational graphs. I first settled on high grained *features* for simplicity and early results.
   - Once this foundational approach was in place, I introduced **Regions of Interest (ROIs)**, striking a balance between interpretability and computational complexity, offering a scalable alternative to finer-grained representations like individual features.
3. **Evaluate quality**: 
   I validated the relevance of ROIs and computational graphs by demonstrating:
   - Faithfulness to the original model's behavior against the size and sparsity of the graphs.
      - **Graph sparsity vs. faithfulness**: Sparse graphs risk losing functional fidelity. Being able to faithfully represent a complex model by a sparse graph ensures that this representation captures relevant information.
   - Clear separability of tasks.
      - **Task disentanglement**: If the *true* computational graph is a subgraph of the one we get, with all additional edges being irrelevant noise, all interesting structures will be lost. To verify the quality of the graphs, I partitioned them by tasks in multi-task settings. These partitions aligned with functional subgraphs, validating the graphs' structural relevance.
   - Predictable changes in model behavior when intervening on specific subgraphs.
4. **Explore scalability**:  
   I scaled the methods to include more complex datasets and models while ensuring faithfulness and sparsity of the resulting graphs.

## Outcomes and Significance  
This work resulted in:
- **Novel Representations**: Computational graphs that are interpretable, scalable, and capture functional and causal relationships within neural networks.
- **A New Framework for Units of Analysis**: The introduction of **ROIs** offers an interpretable, disantangled and coarsely grained units of analysis, keeping all the advantages of both ends of the spectrum (individual neurons and full modules).
- **Significant advances in Circuit Discovery** with the ability to isolate and study task-specific circuits within a model as subgraphs of a general computational graph.
- **Research Directions**:
    - Evolution of structures as the size of the model grows
    - and their emergence during training, offering promising paths for developmental interpretability.
    - Evaluating the potential of ROIs as a robust alternative to traditional units of analysis, such as neurons or attention heads, in AI safety related research.
