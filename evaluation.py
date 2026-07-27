import argparse
import os
import numpy as np
import torch
import pickle
import yaml

from torch_geometric.loader import DataLoader
from baseline.lb_baseline_fitting import feat_extraction as lb_feat_extraction
from baseline.act_baseline_fitting import feat_extraction as act_feat_extraction
from struct_stats import graph_stats_pyg
from training import get_transforms, get_data, get_model, evaluate
from baseline.baseline_utils import annotate, evaluate_predictor, load_data, split


def threshold_graph(graph, t):
    edge_attr = graph.edge_attr[:,:-2]
    condition = (edge_attr < t)
    edge_attr[condition] = 0.0
    to_keep = torch.where(edge_attr.sum(-1) != 0.0)[0]
    graph.edge_attr = graph.edge_attr[to_keep, :]
    graph.edge_index = graph.edge_index[:, to_keep]
    return graph

def get_sparsity(data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    prompt_len = data.prompt_len
    ref_num_edges = (num_nodes * (num_nodes - 1)) / 2 - (prompt_len * (prompt_len - 1)) / 2
    return 1 - num_edges / ref_num_edges

def get_median_degree(data):
    return graph_stats_pyg(data, prompt_graph=False)['median_degree']


def read_yaml_config(path: str) -> dict:
    """
    Load a YAML configuration file into a Python dict.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)  # safe_load prevents execution of arbitrary code
    return config

def run_eval(data, checkpoint_name, lb_checkpoint_name, act_checkpoint_name, checkpoint_folder, threshold, verbose=True):

    if verbose:
        print(f'\n\n[i] checkpoint: {checkpoint_name}')
        print(f'[i] lb checkpoint: {lb_checkpoint_name}')
        print(f'[i] act checkpoint: {act_checkpoint_name}')
        print(f'[i] data: {data}')

    if len(checkpoint_name) > 0:

        # Checkpoint loading
        checkpoint = torch.load(os.path.join(checkpoint_folder, checkpoint_name), weights_only=False)
        chk_args = checkpoint['args']
        model_state_dict = checkpoint['model_state_dict']

        # Overwrite data for zero-shot transfer experiments
        chk_args.data = data
        chk_args.attention_threshold = threshold
        chk_args.by_size = False

        # Set everything up, i.e., get data and model
        T, transform_name, T_, online_transform_name = get_transforms(chk_args)
        _, val_dataset, test_dataset, dims_for_linear_layers = get_data(chk_args, T, transform_name, T_, online_transform_name)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model = get_model(chk_args, test_dataset, dims_for_linear_layers)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Load pretrained weights
        model.load_state_dict(model_state_dict)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Evaluate
        val_loss, val_auroc, val_aupr, val_aupr_hallu = evaluate(model, val_loader, criterion, device)
        test_loss, test_auroc, test_aupr, test_aupr_hallu = evaluate(model, test_loader, criterion, device)
        if verbose:
            print( "\n--------------------- CHARM -------------------------------------------------")
            print(f"Val Loss:   {val_loss:.4f}  | Val AUROC:   {val_auroc:.4f}  | Val AUPR:   {val_aupr:.4f}   | Val AUPR (hallu):   {val_aupr_hallu:.4f}    ")
            print(f"Test Loss:  {test_loss:.4f} | Test AUROC:  {test_auroc:.4f} | Test AUPR:   {test_aupr:.4f} | Test AUPR (hallu):   {test_aupr_hallu:.4f}    ")

        return test_auroc, test_aupr, test_aupr_hallu

    if len(lb_checkpoint_name) > 0:
        with open(os.path.join(checkpoint_folder, lb_checkpoint_name), 'rb') as handle:
            lb_chk_args, lb_model = pickle.load(handle)
        data_seed = int(data.split('_')[-1])
        cut = len(f'_{data_seed}')
        data_pattern = data[:-cut][len("./data/"):]
        data_dict_list, splits, labels = load_data(data_pattern, 'lb', seed=data_seed)
        data_dict_list = annotate(data_dict_list, labels)
        _, lb_val_data, lb_test_data = split(data_dict_list, splits)
  
        pooling = lb_chk_args['pooling'] if not lb_chk_args['tokenwise'] else None
        X_val, Y_val = lb_feat_extraction(lb_val_data, pooling)
        X_test, Y_test = lb_feat_extraction(lb_test_data, pooling)
        val_auroc, val_aupr, val_aupr_hallu = evaluate_predictor(lb_model, X_val, Y_val)
        test_auroc, test_aupr, test_aupr_hallu = evaluate_predictor(lb_model, X_test, Y_test)
        if verbose:
            print( "\n--------------------- Lookback Lens -------------------------------------------------")
            print(f"Val AUROC:   {val_auroc:.4f}  | Val AUPR:   {val_aupr:.4f}   | Val AUPR (hallu):   {val_aupr_hallu:.4f}")
            print(f"Test AUROC:  {test_auroc:.4f} | Test AUPR:   {test_aupr:.4f} | Test AUPR (hallu):   {test_aupr_hallu:.4f}")

        return test_auroc, test_aupr, test_aupr_hallu

    if len(act_checkpoint_name) > 0:
        with open(os.path.join(checkpoint_folder, act_checkpoint_name), 'rb') as handle:
            act_chk_args, act_model = pickle.load(handle)
        data_seed = int(data.split('_')[-1])
        cut = len(f'_{data_seed}')
        data_pattern = data[:-cut][len("./data/"):]
        data_dict_list, splits, labels = load_data(data_pattern, 'act', seed=data_seed)
        data_dict_list = annotate(data_dict_list, labels)
        _, act_val_data, act_test_data = split(data_dict_list, splits)
  
        pooling = act_chk_args['pooling'] if not act_chk_args['tokenwise'] else None
        X_val, Y_val = act_feat_extraction(act_val_data, pooling)
        X_test, Y_test = act_feat_extraction(act_test_data, pooling)
        val_auroc, val_aupr, val_aupr_hallu = evaluate_predictor(act_model, X_val, Y_val)
        test_auroc, test_aupr, test_aupr_hallu = evaluate_predictor(act_model, X_test, Y_test)
        if verbose:
            print( "\n--------------------- Act Probe -------------------------------------------------")
            print(f"Val AUROC:   {val_auroc:.4f}  | Val AUPR:   {val_aupr:.4f}   | Val AUPR (hallu):   {val_aupr_hallu:.4f}")
            print(f"Test AUROC:  {test_auroc:.4f} | Test AUPR:   {test_aupr:.4f} | Test AUPR (hallu):   {test_aupr_hallu:.4f}")

        return test_auroc, test_aupr, test_aupr_hallu


if __name__=='__main__':

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='./data/movies-10k_10k-mistral-7b-i-001__24_28_32___16_16bits_42')
    parser.add_argument("--checkpoint_folder", type=str, default='./checkpoints')
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--lb_checkpoint_name", type=str, default="")
    parser.add_argument("--act_checkpoint_name", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--train_offset", type=int, default=10000)

    # Parse and pring args
    args = parser.parse_args()

    assert len(args.config) > 0 or (len(args.config) == 0 and (len(args.checkpoint_name) > 0 or len(args.lb_checkpoint_name) > 0))
    print('[i] Args:')
    print(args)

    if len(args.config) > 0:
        config = read_yaml_config(args.config)
        for exp in config:
            name = exp['name']
            charms = exp['charms']
            lb = exp['lb']
            act = exp['act']
            data = exp['data']
            threshold = exp['threshold']
            charm_results = list()
            for checkpoint in charms:
                charm_results.append(run_eval(data, checkpoint, '', '', args.checkpoint_folder, threshold, verbose=False))
            lb_results = run_eval(data, '', lb, '', args.checkpoint_folder, threshold, verbose=False)
            act_results = run_eval(data, '', '', act, args.checkpoint_folder, threshold, verbose=False)
            print(f'\n\n\n\n=========== {name}')
            tot_mean_results = [np.mean([tot_res[0] for tot_res in charm_results])]  # AUROC
            tot_mean_results += [np.mean([tot_res[1] for tot_res in charm_results])]  # AUPR
            tot_mean_results += [np.mean([tot_res[2] for tot_res in charm_results])]  # AUPR hallu
            tot_std_results = [np.std([tot_res[0] for tot_res in charm_results])]  # AUROC
            tot_std_results += [np.std([tot_res[1] for tot_res in charm_results])]  # AUPR
            tot_std_results += [np.std([tot_res[2] for tot_res in charm_results])]  # AUPR hallu
            # NOTE: In the paper we report the AUPR w.r.t. the hallucination class, so here we are going to print AUPR hallu as AUPR.
            print(f'Lookback Lens:         Test AUROC {lb_results[0]:.4f}          | Test AUPR {lb_results[2]:.4f}')
            print(f'Act Probe:             Test AUROC {act_results[0]:.4f}          | Test AUPR {act_results[2]:.4f}')
            print(f"CHARM:                 Test AUROC {tot_mean_results[0]:.4f} ± {tot_std_results[0]:.4f} | Test AUPR {tot_mean_results[2]:.4f} ± {tot_std_results[2]:.4f}        ")
            for t, tot_res in enumerate(charm_results):
                print(f'\t({t}):    Test AUROC {tot_res[0]:.4f}           | Test AUPR {tot_res[2]:.4f}')
    else:
        if len(args.checkpoint_name) > 0:
            run_eval(args.data, args.checkpoint_name, '', '', args.checkpoint_folder, args.threshold, verbose=True)
        if len(args.lb_checkpoint_name) > 0:
            run_eval(args.data, '', args.lb_checkpoint_name, '', args.checkpoint_folder, args.threshold, verbose=True)
        if len(args.act_checkpoint_name) > 0:
            run_eval(args.data, '', '', args.act_checkpoint_name, args.checkpoint_folder, args.threshold, verbose=True)