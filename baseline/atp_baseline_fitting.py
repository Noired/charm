import argparse
import torch
import numpy as np
import wandb

from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric import seed_everything
from .act_baseline_fitting import annotate, load_data, split
from .baseline_utils import print_results

POOL_VALUES = ['mean', 'sum', 'max', 'min']

def pool(x, pooling, response_only, data):
    if response_only:
        x = x[data['prompt_len']:]
    if pooling == 'mean':
        x = torch.mean(x, 0)
    elif pooling == 'sum':
        x = torch.sum(x, 0)
    elif pooling == 'max': 
        x = torch.max(x, 0)[0]
    elif pooling == 'min': 
        x = torch.min(x, 0)[0]
    else:
        raise NotImplementedError(pooling)
    return x


def feat_extraction(dict_list, pooling, response_only=True):
    X, Y = list(), list()
    for data in dict_list:
        x = data['atps']
        y = data['annotation']
        if pooling is not None:
            if y.ndim > 0:
                y = y.min()
            y = y.unsqueeze(0)
            x = pool(x, pooling, response_only, data)
        else:
            x = x[data['prompt_len']:]
            assert x.shape[0] == y.shape[0]
        X.append(x.numpy())
        Y.append(y.numpy())
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y


def evaluate(X, Y):
    """Evaluate the baseline predictor."""
    auroc = roc_auc_score(Y, X)
    aupr = average_precision_score(Y, X)
    aupr_hallu = average_precision_score(1 - Y, - X)
    return auroc, aupr, aupr_hallu


def fit_baseline(train_iter, val_iter, test_iter, eval_fn, feat_ex_fn, tokenwise, pooling=None, log=False, other_args=None):

    if log:
        wandb.init()
        if not tokenwise:
            pooling = wandb.config.pooling
            assert pooling is not None

    assert not tokenwise or pooling is None
    X_train, Y_train = feat_ex_fn(train_iter, pooling, **other_args)
    X_val, Y_val = feat_ex_fn(val_iter, pooling, **other_args)
    X_test, Y_test = feat_ex_fn(test_iter, pooling, **other_args)

    train_auroc, train_aupr, train_aupr_hallu = eval_fn(X_train, Y_train)
    val_auroc, val_aupr, val_aupr_hallu = eval_fn(X_val, Y_val)
    test_auroc, test_aupr, test_aupr_hallu = eval_fn(X_test, Y_test)

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
            "test/best_aupr_hallu": test_aupr_hallu})
        wandb.finish()

    return train_auroc, train_aupr, train_aupr_hallu, val_auroc, val_aupr, val_aupr_hallu, test_auroc, test_aupr, test_aupr_hallu

def main(args):

    # Seeding
    seed_everything(args.seed)

    # Data
    data_seed = int(args.data.split('_')[-1])
    cut = len(f'_{data_seed}')
    data_pattern = args.data[:-cut]
    data_dict_list, splits, labels = load_data(data_pattern, 'atp', seed=data_seed)
    data_dict_list = annotate(data_dict_list, labels)
    train_dict_list, val_dict_list, test_dict_list = split(data_dict_list, splits)

    if args.verbose:
        print(f'[i] Data (train): {len(train_dict_list)} samples')
        print(f'[i] Data (val): {len(val_dict_list)} samples')
        print(f'[i] Data (test): {len(test_dict_list)} samples')

    # Sweep or single run?
    if args.pooling == 'none':
        pool_values = POOL_VALUES
    else:
        pool_values = [args.pooling]

    assert not args.tokenwise or args.response_only
    if args.log:
        sweep_config = {
            'name': args.log_name,
            'method': 'grid',
            'parameters': {}}
        if not args.tokenwise:
            sweep_config['parameters']['pooling'] = {'values': pool_values}
        for k, v in vars(args).items():
            if k not in sweep_config['parameters']:
                sweep_config['parameters'][k] = {'value': v}
        sweep_id = wandb.sweep(sweep_config, project='charm')
        fit_fn = lambda: fit_baseline(train_dict_list, val_dict_list, test_dict_list, evaluate, feat_extraction, args.tokenwise, pooling=None, log=args.log, other_args={'response_only': args.response_only})
        wandb.agent(sweep_id, function=fit_fn)
    else:
        exp_count = 0
        if args.tokenwise:
            header = f"Response only: {args.response_only}"
            results = fit_baseline(train_dict_list, val_dict_list, test_dict_list, evaluate, feat_extraction, args.tokenwise, pooling=None, log=args.log, other_args={'response_only': args.response_only})
            if args.verbose:
                print_results(header, results)
            exp_count += 1
        else:
            for pooling in pool_values:
                header = f"Response only: {args.response_only}, pooling: {pooling}"
                results = fit_baseline(train_dict_list, val_dict_list, test_dict_list, evaluate, feat_extraction, args.tokenwise, pooling=pooling, log=args.log, other_args={'response_only': args.response_only})
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
    parser.add_argument("--data", type=str)
    parser.add_argument("--tokenwise", action='store_true')
    parser.add_argument("--pooling", type=str, default='none')
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--response_only", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_name", type=str)

    args = parser.parse_args()
    print('[i] Args:')
    print(args)

    _ = main(args)
