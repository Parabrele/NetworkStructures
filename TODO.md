just a draft containing random thoughts and todos. Most of them are already outdated.

## URGENT :
- in load_saes, load them on model's layer's device
- Look into pre/post ln : e.g. gemma seems to use ln(module(ln(resid))). More generally consider that SAEs might be attached to different points.
    - When loading Gemma SAEs, do E <- E * O^-1 and D <- D * O
        - What about when O is not invertible ?
            - Alors un sous espace de l'espace avant O est inutile et peut être supprimé. Comment le faire proprement ? C'est encore un truc que les entraineurs de SAE doivent faire et pas moi...
    - Put LN pre and post in Submod

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

