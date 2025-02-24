import torch

import networkx as nx
# import networkit as nk

from utils.activation import SparseAct
from utils.sparse_coo_helper import sparse_coo_maximum
from collections import deque, defaultdict

def topological_sort(graph):
    """
    Topological sort of a graph represented as {downstream : [upstream]}
    """
    
    nx_graph = nx.DiGraph()
    for down in graph:
        for up in graph[down]:
            nx_graph.add_edge(up, down)

    return list(nx.topological_sort(nx_graph))

@torch.no_grad()
def get_mask(graph, threshold, threshold_on_nodes=False):
    """
    graph :
    edges : dict of dict of sparse_coo tensors, [name_downstream][name_upstream] -> edge weights

    returns a dict similar to edges but binary sparse coo tensors : only weights above threshold are kept
    """
    if threshold_on_nodes:
        node_mask = {}
        for module in graph:
            if module == 'y':
                continue
            if threshold == -1:
                node_mask[module] = (graph[module] * 0) > 1 # all False
            else:
                node_mask[module] = graph[module].abs() > threshold

        return node_mask
    else:
        edge_mask = {}
        for downstream in graph:
            edge_mask[downstream] = {}
            for upstream in graph[downstream]:
                weights = graph[downstream][upstream].coalesce()
                if threshold == -1:
                    edge_mask[downstream][upstream] = torch.sparse_coo_tensor(
                        [[]] if downstream == 'y' else [[], []],
                        [],
                        weights.size()
                    )
                else:
                    mask = weights.values() > threshold
                    edge_mask[downstream][upstream] = torch.sparse_coo_tensor(
                        weights.indices()[:, mask],
                        torch.ones(mask.sum(), device=weights.device, dtype=torch.bool),
                        weights.size(),
                        dtype=torch.bool
                    )
        return edge_mask

def prune(
    circuit
):
    """
    circuit : nx.DiGraph or dict of dict of sparse_coo tensors
    returns a new nx.DiGraph or dict of dict of sparse_coo tensors
    """
    if circuit is None:
        return None
    if isinstance(circuit, nx.DiGraph):
        return prune_nx(circuit)
    else:
        return prune_sparse_coos(circuit)

def coalesce_edges(edges):
    """
    coalesce all edges sparse coo weights
    """
    for down in edges:
        for up in edges[down]:
            edges[down][up] = edges[down][up].coalesce()
    return edges

@torch.no_grad()
def prune_sparse_coos(
    circuit
):
    """
    circuit : edges downstream -> upstream -> sparse_coo tensor
    returns a new edges

    Assumes edges is a dict of bool sparse coo tensors. If not, it will just forget the values.
    """
    circuit = coalesce_edges(circuit)
    # Copy the circuit to avoid modifying the original
    circuit = {down: {up: circuit[down][up] for up in circuit[down]} for down in circuit}

    architectural_graph = {}
    for down in circuit:
        if down not in architectural_graph:
            architectural_graph[down] = []
        for up in circuit[down]:
            if up not in architectural_graph:
                architectural_graph[up] = []
            architectural_graph[down].append(up)

    topological_order = topological_sort(architectural_graph)

    input_node = topological_order[0]
    if not ('resid' in input_node or 'embed' in input_node):
        raise ValueError("First node should be 'resid' or 'embed'")
    if 'y' != topological_order[-1]:
        raise ValueError("Last node should be 'y'")

    up2down = {}
    for down in circuit:
        if down not in up2down:
            up2down[down] = []
        for up in circuit[down]:
            if up not in up2down:
                up2down[up] = []
            up2down[up].append(down)

    # build a dict module_name -> sparse vector of bools, where True means that the feature is reachable from one embed node
    size = {}
    reachable = {}

    for downstream in circuit:
        for upstream in circuit[downstream]:
            if downstream == 'y':
                size[upstream] = circuit[downstream][upstream].size(0)
                size[downstream] = 1
            else:
                size[upstream] = circuit[downstream][upstream].size(1)
                size[downstream] = circuit[downstream][upstream].size(0)                    
    
    # build a sparse_coo_tensor of ones with size (size['embed']) :
    reachable[input_node] = torch.sparse_coo_tensor(
        torch.arange(size[input_node]).unsqueeze(0),
        torch.ones(size[input_node], device=circuit[downstream][upstream].device, dtype=torch.bool),
        (size[input_node],),
        device=circuit[downstream][upstream].device,
        dtype=torch.bool
    ).coalesce()

    for upstream in topological_order:
        for downstream in up2down[upstream]:
            if upstream not in reachable:
                raise ValueError(f"Upstream {upstream} reachability not available. Check the topological ordering.")
            if downstream not in reachable:
                reachable[downstream] = torch.sparse_coo_tensor(
                    [[]], [],
                    (size[downstream],),
                    device=circuit[downstream][upstream].device
                )

            idx1 = circuit[downstream][upstream].indices() # (2, n1)
            idx2 = reachable[upstream].indices() # (1, n2)

            # keep only rows of circuit[downstream][upstream] at idx in idx2
            new_edges = torch.sparse_coo_tensor(
                [[]] if downstream == 'y' else [[], []],
                [],
                circuit[downstream][upstream].size(),
                device=circuit[downstream][upstream].device,
                dtype=torch.bool
            ).coalesce()

            # A for loop on all ones of the input node is stupidly slow, so we bypass this case.
            if upstream != input_node:
                for u in idx2[0]:
                    mask = (idx1[0] == u if downstream == 'y' else idx1[1] == u)
                    new_edges = torch.sparse_coo_tensor(
                        torch.cat([new_edges.indices(), idx1[:, mask]], dim=1),
                        torch.cat([new_edges.values(), circuit[downstream][upstream].values()[mask]]),
                        new_edges.size(),
                        device=circuit[downstream][upstream].device,
                        dtype=torch.bool
                    ).coalesce()
            else:
                mask = idx1[0] < size[upstream] if downstream == 'y' else idx1[1] < size[upstream]
                new_edges = torch.sparse_coo_tensor(
                    idx1[:, mask],
                    circuit[downstream][upstream].values()[mask],
                    new_edges.size(),
                    device=circuit[downstream][upstream].device,
                    dtype=torch.bool
                ).coalesce()


            circuit[downstream][upstream] = new_edges

            # now look at what downstream features are reachable, as just the indices in the new_edges and add them to reachable[downstream]
            idx = new_edges.indices()[0].unique()
            reachable[downstream] += torch.sparse_coo_tensor(
                idx.unsqueeze(0),
                torch.ones(idx.size(0), ),
                (size[downstream],),
                device=circuit[downstream][upstream].device
            )
            reachable[downstream] = reachable[downstream].coalesce()
    
    return circuit

@torch.no_grad()
def prune_nx(
    G
):
    """
    circuit : nx.DiGraph
    returns a new nx.DiGraph
    """

    G = G.copy()

    # save the 'embed' nodes and their edges to restore them later
    save = []
    to_relabel = {}
    for node in G.nodes:
        if 'embed' in node:
            save += G.edges(node)
            to_relabel[node] = 'embed'

    # merge nodes from embedding into a single 'embed' node, like 'y' is single.
    G = nx.relabel_nodes(G, to_relabel)

    # do reachability from v -> 'y' for all v, remove all nodes that are not reachable
    reachable = nx.ancestors(G, 'y')
    reachable.add('y')
    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # do reachability from 'embed' -> v for all v, remove all nodes that are not reachable
    reachable = nx.descendants(G, 'embed')
    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # untangle the 'embed' node into its original nodes and return the new graph
    G.add_edges_from(save)

    return G

def get_n_nodes(G):
    if isinstance(G, nx.DiGraph):
        return G.number_of_nodes()
    elif isinstance(G, dict):
        n_nodes = 0
        # if G is given as a dict of SparseAct : these are only the nodes
        # if G is given as a dict of dict of sparse_coo tensors : these are the edges
        is_edges = False
        for up in G:
            if isinstance(G[up], dict):
                is_edges = True
                break
        if not is_edges:
            for up in G:
                n_nodes += G[up].to_tensor().sum().item()
            return n_nodes
        else:
            module_nodes = {}
            for down in G:
                for up in G[down]:
                    if down == 'y':
                        down_nodes = torch.tensor([0]) if G[down][up].indices().size(1) == 0 else torch.tensor([1])
                        up_nodes = G[down][up].indices()[0].unique()
                    else:
                        down_nodes = G[down][up].indices()[0].unique()
                        up_nodes = G[down][up].indices()[1].unique()
                    if module_nodes.get(up) is None:
                        module_nodes[up] = up_nodes
                    else:
                        module_nodes[up] = torch.cat([module_nodes[up], up_nodes])
                    if module_nodes.get(down) is None:
                        module_nodes[down] = down_nodes
                    else:
                        module_nodes[down] = torch.cat([module_nodes[down], down_nodes])
            for node in module_nodes:
                n_nodes += module_nodes[node].unique().size(0)
            return n_nodes
    else :
        raise ValueError("Unknown graph type")

def get_n_edges(G):
    if G is None:
        return 0
    if isinstance(G, nx.DiGraph):
        return G.number_of_edges()
    elif isinstance(G, dict):
        # if G is a dict of dict of sparse coo tensors, they contain the edges :
        n_edges = 0
        for down in G:
            for up in G[down]:
                n_edges += G[down][up].values().size(0)
        return n_edges
    elif isinstance(G, tuple):
        # if G is a tuple of nodes and edges, we consider that we are in the node ablation setting and the edge dict is only here to give the dependencies between layers.
        n_edges = 0
        nodes, graph = G
        for down in graph:
            for up in graph[down]:
                if down == 'y':
                    n_edges += nodes[up].to_tensor().sum().item()
                    continue
                elif up == 'y':
                    n_edges += nodes[down].to_tensor().sum().item()
                    continue
                n_up = nodes[up].to_tensor().sum().item()
                n_down = nodes[down].to_tensor().sum().item()
                n_edges += n_up * n_down
        return n_edges
    else :
        raise ValueError("Unknown graph type")

def get_density(edges):
    # edges is a dict of dict of sparse_coo tensors
    if edges is None:
        return 0
    if isinstance(edges, nx.DiGraph):
        return nx.density(edges)
    n_edges = get_n_edges(edges)
    max_edges = 0
    nodes = None
    if isinstance(edges, tuple):
        nodes, edges = edges
    for down in edges:
        for up in edges[down]:  
            if down == 'y':
                n_up = torch.tensor(nodes[up].to_tensor().size()).prod() if nodes is not None else edges[down][up].size(0)
                n_down = 1
            else:
                n_up = torch.tensor(nodes[up].to_tensor().size()).prod() if nodes is not None else edges[down][up].size(1)
                n_down = torch.tensor(nodes[down].to_tensor().size()).prod() if nodes is not None else edges[down][up].size(0)
            max_edges += n_up * n_down
    max_edges = max(max_edges, 1)
    return n_edges / max_edges
