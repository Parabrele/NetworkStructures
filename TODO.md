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
- for downstream = y, also consider arbitrary connections.
- for input gradient computation, to only consider direct paths, use a copy of the input vectors
- step = 1 : AtP (donc ] 0 -> x ] et pas [0 -> x [), step = 10 : IG
- node and edge attribution for any architecture
- support ROIs as nodes

## running the graphs :
- Fix edge ablation error propagation nodes always present
- node and edge ablation for any module dictionaries

## evaluating the graphs :
- metric & faithfulness for
    - do it for one seq f(s)
    - for all s, do f(s), and average (see if on average I can find a graph responsible for one output individually)
    - f(all s) (see if I can find a graph responsible for all outputs)

## node vs edge :
- replicate marks debias bias in bios
- maybe do a mixture of tasks : circuit for both gender and job determination, ablate gender node
- show model unable to use gender : the entire node responsible for gender is gone
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