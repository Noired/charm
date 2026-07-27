import argparse
import torch
import numpy as np
import wandb

from tqdm import tqdm
from .baseline_utils import evaluate_predictor, fit_baseline, print_results, C_VALUES, POOL_VALUES
from .baselines import AttentionBaseline
from dataset.dataset import AttentionDataset
from dataset.transforms import *
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything


@torch.no_grad()
def feat_extraction(loader, readout, baseline='node_averaging'):
    """Extract feats from baseline model"""
    X, Y = list(), list()

    # Model
    in_dim_x = loader.dataset[0].x.shape[-1]
    in_dim_e = loader.dataset[0].edge_attr.shape[-1]
    num_classes = 1
    model = AttentionBaseline(
        in_dim_x,
        in_dim_e,
        num_classes,
        baseline=baseline,
        readout=('none' if readout is None else readout),
        prediction_head=False)
    print(f'[i] Baseline: {model}')

    # Extract feats
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    for data in tqdm(loader):  
        data = data.to(device)
        target = data.y.squeeze().cpu().numpy()
        if target.ndim == 0:
            target = np.expand_dims(target, 0)
        X.append(model(data.x, data.edge_index, data.edge_attr, data.batch, data.response_index, data.ptr, data.prompt_len).cpu().numpy())
        Y.append(target)
    X = np.vstack(X)
    Y = np.concatenate(Y)

    return X, Y

def main(args):

    # Seeding
    seed_everything(args.seed)

    # (Pre)Transforms
    transform_list = list()
    if args.llm_layers != 'all':
        if '+' in args.llm_layers:  # pattern: `first+` or `layer` or 'start-end' or `l1,l2,l3`
            layer_transform = DiscardFirstLayers(args.llm_layers)
        elif '-' in args.llm_layers:
            layer_transform = KeepOnlyLayerInterval(args.llm_layers)
        elif ',' in args.llm_layers:
            layer_transform = KeepOnlySomeLayers(args.llm_layers)
        else:
            layer_transform = KeepOnlyLayer(args.llm_layers)
        transform_list.append(layer_transform)
    if not args.tokenwise:
        transform_list.append(LabelPooling())
    if args.attention_threshold > 0.001:
        transform_list.append(ThresholdAttention(args.attention_threshold))
    T = Compose(transform_list)
    transform_name = '___'.join([repr(t) for t in transform_list])
    args.transform_name = transform_name
    print(f'[i] PreTransforms: {transform_name}')

    # Transforms
    transform_list_on_the_fly = [Cast()]
    if args.baseline == 'lookback':
        transform_list_on_the_fly.append(MarkPrompt())
    T_ = Compose(transform_list_on_the_fly)
    online_transform_name = '___'.join([repr(t) for t in transform_list_on_the_fly])
    args.online_transform_name = online_transform_name
    print(f'[i] Transforms: {online_transform_name}')

    # Data
    data_path = f'./data/{args.data}'
    train_dataset = AttentionDataset(root=data_path, split='train', pre_transform=T, transform=T_, transform_name=transform_name, force_reload=False)
    val_dataset = AttentionDataset(root=data_path, split='val', pre_transform=T, transform=T_, transform_name=transform_name, force_reload=False)
    test_dataset = AttentionDataset(root=data_path, split='test', pre_transform=T, transform=T_, transform_name=transform_name, force_reload=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    if args.verbose:
        print(f'[i] Data (train): {train_dataset}')
        print(f'[i] Data (val): {val_dataset}')
        print(f'[i] Data (test): {test_dataset}')

    # Sweep or single run?
    if args.C == 'none':
        C_values = C_VALUES
    else:
        C_values = [float(args.C)]
    if args.pooling == 'none':
        pool_values = POOL_VALUES
    else:
        pool_values = [args.pooling]

    if args.log:
        sweep_config = {
            'name': args.log_name,
            'method': 'grid',
            'parameters': {
                'C': {
                    'values': C_values}}}
        if not args.tokenwise:
            sweep_config['parameters']['pooling'] = {'values': pool_values}
        for k, v in vars(args).items():
            if k not in sweep_config['parameters']:
                sweep_config['parameters'][k] = {'value': v}
        sweep_id = wandb.sweep(sweep_config, project='charm')
        fit_fn = lambda: fit_baseline(train_loader, val_loader, test_loader, evaluate_predictor, feat_extraction, args.tokenwise, C=None, pooling=None, balance=args.balance, log=args.log, other_args={'baseline': args.baseline})
        wandb.agent(sweep_id, function=fit_fn)
    else:
        for C in C_values:
            if args.tokenwise:
                header = f"Baseline: {args.baseline}, C: {C}"
                results = fit_baseline(train_loader, val_loader, test_loader, evaluate_predictor, feat_extraction, args.tokenwise, C=C, pooling=None, balance=args.balance, log=args.log, other_args={'baseline': args.baseline})
                if args.verbose:
                    print_results(header, results)
            else:
                for readout in pool_values:
                    header = f"Baseline: {args.baseline}, C: {C}, readout: {readout}"
                    results = fit_baseline(train_loader, val_loader, test_loader, evaluate_predictor, feat_extraction, args.tokenwise, C=C, pooling=readout, balance=args.balance, log=args.log, other_args={'baseline': args.baseline})
                    if args.verbose:
                        print_results(header, results)
        train_auroc, train_aupr, train_aupr_hallu, val_auroc, val_aupr, val_aupr_hallu, test_auroc, test_aupr, test_aupr_hallu = results
        res = {
            'train_auroc': train_auroc,
            'train_aupr': train_aupr,
            'train_aupr_hallu': train_aupr_hallu,
            'val_auroc': val_auroc,
            'val_aupr': val_aupr,
            'val_aupr_hallu': val_aupr_hallu,
            'test_auroc': test_auroc,
            'test_aupr': test_aupr,
            'test_aupr_hallu': test_aupr_hallu}
        return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str)
    parser.add_argument("--tokenwise", action='store_true')
    parser.add_argument("--attention_threshold", type=float, default=0.05)
    parser.add_argument("--llm_layers", type=str, default='all')
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--C", type=str, default='none')
    parser.add_argument("--pooling", type=str, default='none')
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_name", type=str,)

    args = parser.parse_args()
    print('[i] Args:')
    print(args)

    _ = main(args)