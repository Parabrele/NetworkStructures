import os
from datetime import datetime

import torch as t

def get_min_value(dtype):
    if dtype.is_floating_point:
        return t.finfo(dtype).min
    elif dtype.is_signed:
        return t.iinfo(dtype).min
    else:
        return 0

@t.jit.script
def compile_for_loop_sparse_coo_max(inverse, new_values, x_values):
    for i, j in enumerate(inverse):
        new_values[j] = t.max(new_values[j], x_values[i])
        
def sparse_coo_max(x, dim):
    """
    x : a sparse COO tensor
    dim : a dimension to reduce
    return a sparse COO tensor with the maximum value along the specified dimension
    """
    # coalesce x
    x = x.coalesce()
    x_shape = x.shape
    new_shape = x_shape[:dim] + x_shape[dim+1:]

    idxs = x.indices() # [len(x_shape), n]
    new_idxs = t.cat((idxs[:dim], idxs[dim+1:]), dim=0) # [len(x_shape)-1, n]
    # TODO : here and in maximum, iterate only over values that are not unique. Define new_values as cat, and then take new_values = new_values[...], and itterate only where ... has count > 1
    new_idxs, inverse = new_idxs.unique(dim=-1, return_inverse=True)
    new_values = t.full((new_idxs.shape[1],), get_min_value(x.dtype), dtype=x.dtype, device=x.device)

    # for each new index, get the max value along original indexes that were merged to it. Use return_inverse
    compile_for_loop_sparse_coo_max(inverse, new_values, x.values())

    return t.sparse_coo_tensor(new_idxs, new_values, new_shape).coalesce()

def sparse_coo_amax(x, dim):
    """
    x : a sparse COO tensor
    dim : a dimension to reduce/a tuple of dimensions to reduce
    return a sparse COO tensor with the maximum value along the specified dimension(s)
    """
    if isinstance(dim, int):
        return sparse_coo_max(x, dim)
    else:
        dim = sorted(list(dim), reverse=True)
        for d in dim:
            x = sparse_coo_max(x, d)
        return x

@t.jit.script
def compile_for_loop_sparse_coo_maximum(new_indices, new_values, indices, values):
    for i, idx in enumerate(indices.t()):
        mask = (new_indices.t() == idx).all(dim=1)
        new_values[mask] = t.max(new_values[mask], values[i])

def sparse_coo_maximum(x, y):
    """
    x, y : sparse COO tensors with positive values !
    return a sparse COO tensor with the maximum value elementwise between x and y
    """
    x = x.coalesce()
    y = y.coalesce()
    assert x.shape == y.shape
    assert x.device == y.device
    assert x.dtype == y.dtype

    new_indices = t.cat((x.indices(), y.indices()), dim=1)
    new_indices = new_indices.unique(dim=-1)

    new_values = t.full((new_indices.shape[1],), 0, dtype=x.dtype, device=x.device)

    compile_for_loop_sparse_coo_maximum(new_indices, new_values, x.indices(), x.values())
    compile_for_loop_sparse_coo_maximum(new_indices, new_values, y.indices(), y.values())

    return t.sparse_coo_tensor(new_indices, new_values, x.shape).coalesce()

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def sparse_permute(x, perm):
    """
    x : a sparse COO tensor
    perm : a permutation of the dimensions
    return x permuted according to perm
    """
    x = x.coalesce()
    new_indices = x.indices()[list(perm)]
    return t.sparse_coo_tensor(new_indices, x.values(), tuple(x.shape[perm[i]] for i in range(len(perm))))

def rearrange_weights(shapes, edges):
    # rearrange weight matrices # dict : downstream -> upstream -> weight_matrix
    for downstream in edges:
        # get shape for child
        if downstream != 'y':
            bd, sd, fd = shapes[downstream]
        for upstream in edges[downstream]:
            bu, su, fu = shapes[upstream]
            weight_matrix = edges[downstream][upstream]
            if downstream == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bu, su, fu+1))
            else:
                assert bd == bu
                weight_matrix = sparse_reshape(weight_matrix, (bd, sd, fd+1, bu, su, fu+1))
            edges[downstream][upstream] = weight_matrix

def update_frequency(freq, data, is_node=False, threshold=0.0):
    # side effect on freq to count the number of times a feature is used.
    # If freq is None, we are not trying to count that information.
    if freq is not None:
        if is_node:
            # When circuit discovering nodes, data is a dict of SparseAct, not thresholded
            for node in data:
                if node != 'y':
                    if freq.get(node) is None:
                        freq[node] = 0
                    b = data[node].act.shape[0]
                    for i in range(b):
                        freq[node] += (t.cat((data[node].act[i], data[node].resc[i])) > threshold)
        else:
            # When circuit discovering edges, data is a dict of dict of sparse_coo tensors, already thresholded.
            for downstream in data:
                for upstream in data[downstream]:
                    w = data[downstream][upstream] # sparse_coo_tensor, shape [b, f+1] or [b, f+1, b, f+1]
                    b = w.shape[0]
                    for i in range(b):
                        if downstream == 'y':
                            # w has shape [b, f+1]
                            idx = w[i].coalesce().indices()[0].unique()
                            shape = w[i].shape[0]
                        else:
                            # w has shape [b, f+1, b, f+1] : indexing in sparse_coo tensors is weird, so take batch
                            # for downstream, then permute dimensions and take batch for upstream. Now, we have a
                            # weight matrix of shape [f_down, f_up]. Take the up indices and unique them.
                            wii = sparse_permute(w[i], [1, 0, 2])[i]
                            idx = wii.coalesce().indices()[1].unique()
                            shape = wii.shape[1]
                        
                        if freq.get(upstream) is None:
                            freq[upstream] = t.zeros(shape, dtype=t.int64, device=w.device)
                        freq[upstream][idx] += 1

def aggregate_nodes(nodes, aggregation='sum', freq=None, threshold=0.0):
    if aggregation == 'sum':
        n_s_fct = lambda n: n.sum(dim=1)
        n_b_fct = lambda n: n.mean(dim=0)
    elif aggregation == 'max':
        n_s_fct = lambda n: n.amax(dim=1)
        n_b_fct = lambda n: n.amax(dim=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    def _aggregate(n_fct):
        for node in nodes:
            if node != 'y':
                nodes[node] = n_fct(nodes[node])

    # aggregate across sequence position
    _aggregate(n_s_fct)
    update_frequency(freq, nodes, is_node=True, threshold=threshold)
    # aggregate across batch dimension
    _aggregate(n_b_fct)

def aggregate_weights(
    shapes, edges, aggregation='sum', dump_all=False, save_path=None, freq=None
):
    if dump_all:
        raise NotImplementedError("dump_all is not implemented in the new version of aggregate_weights")
    
    if aggregation == 'sum':
        w_y_s_fct = lambda w, b: w.sum(dim=1)
        w_s_fct = lambda w, b: w.sum(dim=(1, 4))

        w_y_b_fct = lambda w, b: w.sum(dim=0) / b
        w_b_fct = lambda w, b: w.sum(dim=(0, 2)) / b # TODO : shouldn't this be / (b**2) ?
    elif aggregation == 'max':
        w_y_s_fct = lambda w, b: sparse_coo_amax(w, dim=1)
        w_s_fct = lambda w, b: sparse_coo_amax(w, dim=(1, 4))

        w_y_b_fct = lambda w, b: sparse_coo_amax(w, dim=0)
        w_b_fct = lambda w, b: sparse_coo_amax(w, dim=(0, 2))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    def _aggregate(w_y_fct, w_fct):
        for downstream in edges:
            for upstream in edges[downstream]:
                shape = shapes[upstream]
                b = shape[0]
                weight_matrix = edges[downstream][upstream]
                if downstream == 'y':
                    weight_matrix = w_y_fct(weight_matrix, b)
                else:
                    weight_matrix = w_fct(weight_matrix, b)
                edges[downstream][upstream] = weight_matrix

    # aggregate across sequence position
    _aggregate(w_y_s_fct, w_s_fct)
    update_frequency(freq, edges)
    # aggregate across batch dimension
    _aggregate(w_y_b_fct, w_b_fct)
