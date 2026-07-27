import argparse
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
import wandb

from tqdm import tqdm
from .baseline_utils import evaluate_predictor, print_results
from .baselines import AttentionBaseline
from dataset.dataset import AttentionDataset
from dataset.transforms import *
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from sklearn.decomposition import PCA

TOPK_VALUES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]

@torch.no_grad()
def feat_extraction(loader, topk):
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
        baseline='lapeig',
        readout='none',
        prediction_head=False)
    print(f'[i] Baseline: {model}')

    # Extract feats
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    discarded = 0
    for data in tqdm(loader):  
        data = data.to(device)
        target = data.y.squeeze().cpu().numpy()
        if target.ndim == 0:
            target = np.expand_dims(target, 0)
        x = model(data.x, data.edge_index, data.edge_attr, data.batch, data.response_index, data.ptr, data.prompt_len).cpu().numpy()
        if x.shape[0] < topk:
            discarded += 1
            continue
        topk_sorted = np.sort(x, axis=0)[-topk:, :]
        flattened = topk_sorted.flatten(order="F")
        X.append(flattened)
        Y.append(target)
    X = np.vstack(X)
    Y = np.concatenate(Y)

    return X, Y, discarded

def fit_baseline(train_iter, val_iter, test_iter, eval_fn, feat_ex_fn, topk=None, log=False):

    if log:
        run = wandb.init(project="charm")
        topk = wandb.config.topk
    else:
        assert topk is not None
        run = None

    X_train, Y_train, discarded_train = feat_ex_fn(train_iter, topk)
    X_val, Y_val, discarded_val = feat_ex_fn(val_iter, topk)
    X_test, Y_test, discarded_test = feat_ex_fn(test_iter, topk)
    print(f'Discarded (train): {discarded_train}')
    print(f'Discarded (val): {discarded_val}')
    print(f'Discarded (test): {discarded_test}')

    pca = PCA(n_components=512)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    predictor = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=2000)
    predictor.fit(X_train, Y_train)

    train_auroc, train_aupr, train_aupr_hallu = eval_fn(predictor, X_train, Y_train)
    val_auroc, val_aupr, val_aupr_hallu = eval_fn(predictor, X_val, Y_val)
    test_auroc, test_aupr, test_aupr_hallu = eval_fn(predictor, X_test, Y_test)

    if log:
        wandb.log({
            "train/best_auroc": train_auroc,
            "val/best_auroc": val_auroc,
            "test/best_auroc": test_auroc,
            "train/best_aupr": train_aupr,
            "val/best_aupr": val_aupr,
            "test/best_aupr": test_aupr,
            "train/best_aupr_hallu": train_aupr_hallu,
            "val/best_aupr_hallu": val_aupr_hallu,
            "test/best_aupr_hallu": test_aupr_hallu,
            "train/discarded": discarded_train,
            "val/discarded": discarded_val,
            "test/discarded": discarded_test}
            )
        wandb.finish()

    return train_auroc, train_aupr, train_aupr_hallu, val_auroc, val_aupr, val_aupr_hallu, test_auroc, test_aupr, test_aupr_hallu

def main(args):

    # Seeding
    seed_everything(args.seed)

    # (Pre)Transforms
    transform_list = list()
    transform_list.append(LabelPooling())
    if args.attention_threshold > 0.001:
        transform_list.append(ThresholdAttention(args.attention_threshold))
    T = Compose(transform_list)
    transform_name = '___'.join([repr(t) for t in transform_list])
    args.transform_name = transform_name
    print(f'[i] PreTransforms: {transform_name}')

    # Transforms
    transform_list_on_the_fly = [Cast()]
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
    if args.topk == 'none':
        topk_values = TOPK_VALUES
    else:
        topk_values = [int(args.topk)]

    if args.log:
        sweep_config = {
            'name': args.log_name,
            'method': 'grid',
            'parameters': {
                'topk': {
                    'values': topk_values}}}
        for k, v in vars(args).items():
            if k not in sweep_config['parameters']:
                sweep_config['parameters'][k] = {'value': v}
        sweep_id = wandb.sweep(sweep_config, project='charm')
        fit_fn = lambda: fit_baseline(train_loader, val_loader, test_loader, evaluate_predictor, feat_extraction, topk=None, log=args.log)
        wandb.agent(sweep_id, function=fit_fn)
    else:
        exp_count = 0
        for topk in topk_values:
            header = f"Baseline: LapEig, topk: {topk}"
            results = fit_baseline(train_loader, val_loader, test_loader, evaluate_predictor, feat_extraction, topk=topk, log=args.log)
            if args.verbose:
                print_results(header, results)
            exp_count += 1
        if exp_count == 1:  # We have used the script for a target run, let us return results
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
    parser.add_argument("--topk", type=str, default='none')
    parser.add_argument("--data", type=str)
    parser.add_argument("--attention_threshold", type=float, default=0.001)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_name", type=str)

    args = parser.parse_args()
    print('[i] Args:')
    print(args)

    _ = main(args)
