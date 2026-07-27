import argparse
import numpy as np
import wandb

from torch_geometric import seed_everything
from .baseline_utils import annotate, evaluate_predictor, fit_baseline, load_data, pool, print_results, split, C_VALUES, POOL_VALUES

def feat_extraction(dict_list, pooling):
    X, Y = list(), list()
    for data in dict_list:
        x = data['lookback_ratio']
        x = x.permute(2, 0, 1)
        x = x.flatten(start_dim=1)
        y = data['annotation']
        if pooling is not None:
            y = y.min().unsqueeze(0)
            x = pool(x, pooling)
        else:
            assert x.shape[0] == y.shape[0]
        X.append(x.numpy())
        Y.append(y.numpy())
    X = np.vstack(X)
    Y = np.concatenate(Y)
    return X, Y

def main(args):

    # Seeding
    seed_everything(args.seed)

    # Data
    data_seed = int(args.data.split('_')[-1])
    cut = len(f'_{data_seed}')
    data_pattern = args.data[:-cut]
    data_dict_list, splits, labels = load_data(data_pattern, 'lb', seed=data_seed)
    data_dict_list = annotate(data_dict_list, labels)
    train_dict_list, val_dict_list, test_dict_list = split(data_dict_list, splits)

    if args.verbose:
        print(f'[i] Data (train): {len(train_dict_list)} samples')
        print(f'[i] Data (val): {len(val_dict_list)} samples')
        print(f'[i] Data (test): {len(test_dict_list)} samples')

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
        fit_fn = lambda: fit_baseline(train_dict_list, val_dict_list, test_dict_list, evaluate_predictor, feat_extraction, args.tokenwise, C=None, pooling=None, balance=args.balance, log=args.log, other_args={}, dataname=args.data, baseline_name='lb', checkpoint_folder=(args.checkpoint_folder if args.checkpoint else None))
        wandb.agent(sweep_id, function=fit_fn)
    else:
        exp_count = 0
        for C in C_values:
            if args.tokenwise:
                header = f"C: {C}"
                results = fit_baseline(train_dict_list, val_dict_list, test_dict_list, evaluate_predictor, feat_extraction, args.tokenwise, C=C, pooling=None, balance=args.balance, log=args.log, other_args={}, dataname=args.data, baseline_name='lb', checkpoint_folder=(args.checkpoint_folder if args.checkpoint else None))
                if args.verbose:
                    print_results(header, results)
                exp_count += 1
            else:
                for pooling in pool_values:
                    header = f"C: {C}, pooling: {pooling}"
                    results = fit_baseline(train_dict_list, val_dict_list, test_dict_list, evaluate_predictor, feat_extraction, args.tokenwise, C=C, pooling=pooling, balance=args.balance, log=args.log, other_args={}, dataname=args.data, baseline_name='lb', checkpoint_folder=(args.checkpoint_folder if args.checkpoint else None))
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
    parser.add_argument("--C", type=str, default='none')
    parser.add_argument("--pooling", type=str, default='none')
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_name", type=str)
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument("--checkpoint_folder", type=str, default='./checkpoints')
    
    args = parser.parse_args()
    print('[i] Args:')
    print(args)

    _ = main(args)