# Run the faithfulness of circuits experiments.

###
# In the correct directory, execute e.g. :
# python -m experiments.main &
###

##########
# Get arguments from the command line
##########

# DONE : faithfulness seems to be broken, negative even when KL is clearly low.
#        Sometimes the empty circuit has very low KL, then the faithfulness is very negative and the mean of all gets negative.
# TODO : crashes with empty circuit
# TODO : test freq

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--dataset", type=str, default="ioi")

parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--eval_batch_size", type=int, default=100)

parser.add_argument("--node_circuit", action="store_true", default=False)
parser.add_argument("--threshold", type=float, default=1e-4)
parser.add_argument("--start_at_layer", type=int, default=2)

parser.add_argument("--save_path", type=str, default="/scratch/pyllm/dhimoila/outputs/main_output/")

args = parser.parse_args()

##########
# Import the necessary modules
##########

import os

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

multiprocessing.set_start_method('spawn', force=True)

import torch
from transformers import logging
logging.set_verbosity_error()
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from connectivity.effective import get_circuit_feature

from evaluation.faithfulness import faithfulness as faithfulness_fn

from data.buffer import single_input_buffer, wikipedia_buffer, gp_buffer, gt_buffer, ioi_buffer, bool_buffer, mixture_buffer, unpack_batch


from utils.ablation_fns import zero_ablation, mean_ablation, id_ablation
from utils.savior import save_circuit
from utils.plotting import plot_faithfulness
from utils.metric_fns import metric_fn_logit, metric_fn_KL, metric_fn_statistical_distance, metric_fn_acc, metric_fn_MRR
from utils.experiments_setup import load_model_and_modules, load_saes, get_architectural_graph

import math

##########
# Initialize the parameters
##########

save_path = args.save_path

edge_circuit = not args.node_circuit

use_attn_mlp = False
use_resid = True
start_at_layer = args.start_at_layer
threshold = args.threshold
steps = 10
get_freq = True

experiment_name = ('resid' if use_resid else '') \
    + ('_' if use_resid and use_attn_mlp else '') \
    + ('attn_mlp' if use_attn_mlp else '') \
    + '/' + ('edge' if edge_circuit else 'node') + '_ablation/'

save_path += args.dataset + '/'
save_path += experiment_name

batch_size = args.batch_size
eval_batch_size = args.eval_batch_size

nb_eval_thresholds = 20

if args.dataset == "ioi":
    DATASET = ioi_buffer
    n_elts = 200
    n_tests = 200
elif args.dataset == "mixture":
    DATASET = mixture_buffer
    n_elts = 600
    n_tests = 600
else:
    raise ValueError(f"Unknown dataset : {args.dataset}")

metric_fn = metric_fn_logit
metric_fn_dict = {
    'logit': metric_fn_logit,
    'KL': metric_fn_KL,
    'Statistical Distance': metric_fn_statistical_distance,
    # 'acc': metric_fn_acc,
    # 'MRR': metric_fn_MRR,
}

default_ablation = 'id'
if default_ablation == 'mean':
    ablation_fn = mean_ablation
elif default_ablation == 'zero':
    ablation_fn = zero_ablation
elif default_ablation == 'id':
    ablation_fn = id_ablation
else:
    raise ValueError(f"Unknown default ablation function : {default_ablation}")

def circuit_to_device(circuit, device):
    """
    Helper function to move all tensors in the circuit to the device.
    Depending on whether we are considering nodes or edges, circuit is a dict of tensors or a dict of dict of tensors.
    """
    if not edge_circuit:
        circuit = {k : v.to(device) if k != 'y' else None for k, v in circuit.items()}
    else:
        circuit = {k : {kk : vv.to(device) for kk, vv in v.items()} for k, v in circuit.items()}
    return circuit

def add_circuit(tot_circuit, circuit, factor=1):
    """
    Helper function to add a circuit to the total.
    Depending on whether we are considering nodes or edges, circuit is a dict of tensors or a dict of dict of tensors.
    """
    if tot_circuit is None:
        if not edge_circuit:
            tot_circuit = {k : factor * v if k != 'y' else None for k, v in circuit.items()}
        else:
            tot_circuit = {k : {kk : factor * vv for kk, vv in v.items() } for k, v in circuit.items()}
    else:
        if not edge_circuit:
            for k, effect in circuit.items():
                if k == 'y': continue
                tot_circuit[k] += factor * effect
        else:
            for k in circuit.keys():
                for kk, effect in circuit[k].items():
                    tot_circuit[k][kk] += factor * effect
    return tot_circuit

def normalize_circuit(circuit, factor):
    if not edge_circuit:
        circuit = {k : v / factor if k != 'y' else None for k, v in circuit.items()}
    else:
        circuit = {k : {kk : vv / factor for kk, vv in v.items()} for k, v in circuit.items()}
    return circuit

def get_circuit(circuit_fn, device_id=0, perm=None):
    DEVICE = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else torch.device('cpu')    

    # Set up the model and data
    model, name2mod = load_model_and_modules(device=DEVICE, resid=use_resid, attn=use_attn_mlp, mlp=use_attn_mlp, start_at_layer=start_at_layer)
    architectural_graph = get_architectural_graph(model, name2mod)
    dictionaries = load_saes(model, name2mod, device=DEVICE)

    tot_circuit = None
    tot_inputs = 0

    buffer = DATASET(model, batch_size, DEVICE, ctx_len=None, perm=perm)

    # Compute the circuit

    freq = {} if get_freq else None
    
    for batch in tqdm(buffer):
        tokens, trg_idx, trg, corr, corr_trg = unpack_batch(batch)

        b = tokens.shape[0]
        tot_inputs += b

        try:
            circuit = circuit_fn(
                clean=tokens,
                patch=corr,
                model=model,
                architectural_graph=architectural_graph,
                name2mod=name2mod,
                dictionaries=dictionaries,
                metric_fn=metric_fn_logit,
                metric_kwargs={"trg_idx": trg_idx, "trg_pos": trg, "trg_neg": corr_trg},
                ablation_fn=ablation_fn,
                threshold=threshold,
                steps=steps,
                edge_circuit=edge_circuit,
                freq=freq,
            )
        except ValueError as e:
            print(e)
            print(b)
            print(tot_inputs)
            print("Tokens :")
            print(tokens.shape)
            print(tokens)
            print("Trg_idx :")
            print(trg_idx.shape)
            print(trg_idx)
            print("Trg :")
            print(trg.shape)
            print(trg)
            print("Corr :")
            print(corr.shape)
            print(corr)
            print("Corr_trg :")
            print(corr_trg.shape)
            print(corr_trg)

            tot_inputs -= b
            import time
            # sleep one minute
            time.sleep(60)
            continue

        tot_circuit = add_circuit(tot_circuit, circuit, b)

    # normalize and return
    tot_circuit = normalize_circuit(tot_circuit, tot_inputs)

    # Save the circuit
    save_circuit(
        save_path+f"circuit/{DEVICE.type}_{DEVICE.index}/",
        tot_circuit,
        0,   
    )

    # save the frequency
    if get_freq:
        if not os.path.exists(save_path+f"freq/{DEVICE.type}_{DEVICE.index}/"):
            os.makedirs(save_path+f"freq/{DEVICE.type}_{DEVICE.index}/")
        for node in freq:
            freq[node] = freq[node].cpu()
        torch.save(freq, save_path+f"freq/{DEVICE.type}_{DEVICE.index}/0.pt")

    return tot_circuit

def get_faithfulness(circuit, perm=None):
    """
    Compute the faithfulness of the circuit.
    """
    model, name2mod = load_model_and_modules(device=DEVICE, resid=use_resid, attn=use_attn_mlp, mlp=use_attn_mlp, start_at_layer=start_at_layer)
    architectural_graph = get_architectural_graph(model, name2mod)
    dictionaries = load_saes(model, name2mod, device=DEVICE)

    buffer = DATASET(model, eval_batch_size, DEVICE, perm=perm)

    aggregated_outs = None
    n_batches = 0
    tot_inputs = 0

    for batch in tqdm(buffer):
        tokens, trg_idx, trg, corr, corr_trg = unpack_batch(batch)

        tot_inputs += tokens.shape[0]
        if tot_inputs > n_tests:
            break
        n_batches += 1
        
        thresholds = torch.logspace(math.log10(threshold)-2, 0, nb_eval_thresholds, 10).tolist()

        faithfulness = faithfulness_fn(
            model,
            name2mod,
            dictionaries,
            clean=tokens,
            circuit=circuit,
            architectural_graph=architectural_graph,
            thresholds=thresholds,
            metric_fn=metric_fn_dict,
            metric_fn_kwargs={"trg_idx": trg_idx, "trg_pos": trg, "trg_neg": corr_trg},
            patch=corr,
            ablation_fn=ablation_fn,
            default_ablation=default_ablation,
            node_ablation=(not edge_circuit),
        )

        if aggregated_outs is None:
            aggregated_outs = faithfulness
            continue

        for t, out in faithfulness.items():
            if t == 'complete' or t == 'empty':
                continue
            else:
                for fn_name in out['faithfulness']:                        
                    aggregated_outs[t]['faithfulness'][fn_name] += out['faithfulness'][fn_name]
                for fn_name in out['completeness']:                        
                    aggregated_outs[t]['completeness'][fn_name] += out['completeness'][fn_name]

        del faithfulness

    for t, out in aggregated_outs.items():
        if t == 'complete' or t == 'empty':
            continue
        for fn_name in out['faithfulness']:
            aggregated_outs[t]['faithfulness'][fn_name] /= n_batches
        for fn_name in out['completeness']:
            aggregated_outs[t]['completeness'][fn_name] /= n_batches

    return aggregated_outs

if __name__ == "__main__":
    # for n_elts in [8, 16, 32, 64, 128, 256]:
        ##########
        # Get Marks circuit
        ##########

        print("Getting circuit...")

        seed = 42

        if True:#not os.path.exists(save_path+"circuit/merged/0.pt"):
            # set the seed
            torch.manual_seed(seed)
            perm = torch.randperm(n_elts)
            available_gpus = torch.cuda.device_count()
            elts_per_gpu = n_elts // available_gpus
            if elts_per_gpu == 0:
                elts_per_gpu = n_elts
                available_gpus = 1
            
            if True:
                # Sometimes this crashes right at the end, I don't know why, so I just manually get the last saved circuit and add them.
                futures = []
                with ProcessPoolExecutor(max_workers=available_gpus) as executor:
                    for i in range(available_gpus):
                        futures.append(executor.submit(get_circuit, get_circuit_feature, i, perm[i*elts_per_gpu:(i+1)*elts_per_gpu]))

                tot_circuit = None
                for future in futures:
                    circuit = future.result()
                    circuit = circuit_to_device(circuit, DEVICE)
                    tot_circuit = add_circuit(tot_circuit, circuit)
            else:
                tot_circuit = None
                for i in range(available_gpus):
                    circuit = torch.load(save_path+f"circuit/cuda_{i}/0.pt")["circuit"]
                    circuit = circuit_to_device(circuit, DEVICE)
                    tot_circuit = add_circuit(tot_circuit, circuit)

            circuit = normalize_circuit(tot_circuit, available_gpus)

            # Save the circuit
            save_circuit(
                save_path+f"circuit/{n_elts}/merged/",
                circuit,
                0,
            )

            # save the frequency
            if get_freq:
                tot_freq = {}
                for i in range(available_gpus):
                    freq = torch.load(save_path+f"freq/cuda_{i}/0.pt")
                    for node in freq:
                        if tot_freq.get(node) is None:
                            tot_freq[node] = freq[node]
                        else:
                            tot_freq[node] += freq[node]

                for node in tot_freq:
                    tot_freq[node] = tot_freq[node].cpu()
                if not os.path.exists(save_path+"freq/merged/"):
                    os.makedirs(save_path+"freq/merged/")
                torch.save(tot_freq, save_path+"freq/merged/0.pt")

                
                all_flattened = torch.zeros(0,)
                all_attr = torch.zeros(0,)
                print(all_flattened)
                for k, v in tot_freq.items():
                    print(v.nonzero()[:, 0].shape)
                    all_flattened = torch.cat([all_flattened, v.cpu()])
                    attr = circuit[k].to_tensor()
                    all_attr = torch.cat([all_attr, attr.cpu()])
                
                # mean attr per frequency :
                freq, count = all_flattened.unique(return_counts=True)
                attr_per_freq = torch.zeros(freq.shape[0])
                for i, f in enumerate(freq):
                    attr_per_freq[i] = all_attr[all_flattened == f].mean()
                
                # Create a histogram and print it :
                print(freq)
                print(freq.shape)
                print(count)
                print(count.shape)
                print(attr_per_freq)
                print(attr_per_freq.shape)

                # All nodes appearing less than 25 times are removed :
                for k in tot_freq.keys():
                    circuit[k].act[tot_freq[k][:-1] < 5] = 0

        else:
            # Load the circuit :
            circuit_dict = torch.load(save_path+"circuit/merged/0.pt")
            circuit = circuit_dict["circuit"]
        
        print("Done.")

        ##########
        # Evaluate
        ##########

        print("Evaluation")
        torch.manual_seed(seed)
        perm = torch.randperm(n_elts)
        aggregated_outs = get_faithfulness(circuit, perm)
        plot_faithfulness(aggregated_outs, save_path=save_path+f"test_freq/{n_elts}/")
        print("Done.")
