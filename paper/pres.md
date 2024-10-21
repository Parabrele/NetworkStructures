<!--
Draft.
-->

### Granularity

- High : low-level details, detailed information
    - neurons :
        - ✗ Polysemantic : multiple meanings, interpretations. Involved in multiple tasks. Each task can involve multiple neurons.
        - ✗ Expensive : for large models, large number of neurons. Due to previous point, dense dependencies. This makes it expensive to compute the computational graph
        - ✓ Insights from the computational graph are more meaningful due to the high detail of the nodes. It is however almost impossible to manually interpret any of it.
        - ✗ Insights from the computational graph are less meaningful due to the polysemantic nature of neurons and their dense dependencies.
        - ✓ No overhead cost : use canonical representations of the model.
    - features :
        - ✓ Monosemantic : single meaning, interpretation.
        - ✗ Expensive : for large models, large number of features. However, sparse dependencies due to the disentangled representations. The computational graph is less expensive to construct, but scalability is still an issue.
        - ✓ Insights from the computational graph are more meaningful due to the high detail of the nodes. It is however almost impossible to manually interpret any of it.
        - ✓ Insights from the computational graph are more meaningful due to the monosemantic nature of features and their sparse dependencies.
        - ✗ Overhead cost : requires more compute before the analysis to get the feature dictionaries
- Low : high-level details, abstract /coarse information
    - Attention heads :
        - ✗ Polysemantic : multiple meanings, interpretations.
        - ✓ Cheap : far fewer attention heads than neurons. Even with dense dependencies, the computational graph is much more reasonable to construct.
        - ✗ Insights from the computational graph are less meaningful due to the low detail of the nodes. It is however much more reasonable to manually interpret the nodes and structures.
        - ✗ Insights from the computational graph are less meaningful due to the polysemantic nature of attention heads and their dense dependencies.
        - ✓ No overhead cost : use canonical representations of the model.
        - ✗ No equivalent for MLP layers or other architectures.
    - Regions of Interest (RoI) :
        - ✓ Monosemantic : single functionality.
        - ✓ Cheap : far fewer RoIs than neurons or features. With sparse dependencies, the computational graph is very reasonable to construct.
        - ✗ Insights from the computational graph are less meaningful due to the low amount of information of the nodes. It is however much more reasonable to manually interpret the nodes and structures.
        - ✓ Insights from the computational graph are meaningful due to the monosemantic nature of RoIs and their sparse dependencies.
        - ✗ Overhead cost : requires more compute before the analysis to get the feature dictionaries
        - ✓ Equivalent for MLP layers and any other architectures.

### What to include in the presentation

History on explaining neural network internal mechanisms.

- Vision :
    - attribution
        - explain output decision by attributing it to input pixels
        - Provide visual explanations for the model's decision
        - Fails in extremely simple cases (Trojan)
    - intermediate components
        - Understand isolated components of the model (in CNN, low -> high level features, edge/curve detectors, snout detectors, etc.)
        - Visual interpretation of model components (neurons or layers) through feature visualization
        - Find seemingly related features across layers that appear to compose into higher level features

-> visualizations can work for visual features but not for other types of data like text.
-> Explain both what parts of the input lead to some conclusion and what intermediate nodes seem to be doing. Circuit discovery : explain *how* the model arrives at a decision based on the input by also focusing on interactions between intermediate nodes and their interpretations.

- Circuit Discovery :
    - Use attribution on intermediate nodes to understand which are relevant in solving some task, then interpret them.
    - Can use several types of nodes : (fine-grained) neurons, features, (coarse) attention heads, whole MLP layers, RoIs, etc.
    - task specific -> general ?
    