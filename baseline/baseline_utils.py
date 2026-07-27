from datetime import datetime
import os
import pickle
import wandb
import torch
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

C_VALUES = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 100000.0]
POOL_VALUES = ['mean', '-3', '-2', '-1', '0', '1', '2']

def load_data(data_pattern, artifact_key, seed=42):
    data_path = f'raw_data/{data_pattern}/{artifact_key}_{data_pattern}_{seed}.pt'
    if not os.path.exists(data_path):
        data_path = f'./data/{data_pattern}_{seed}/raw/all_{artifact_key}s.pt'
    labels_path = f'./data/{data_pattern}_{seed}/raw/labels.pt'
    split_path = f'./data/{data_pattern}_{seed}/splits.pt'
    data_dict_list = torch.load(data_path, weights_only=False)
    splits = torch.load(split_path, weights_only=False)
    labels = torch.load(labels_path, weights_only=False)
    return data_dict_list, splits, labels


def split(data_dict_list, split):
    train_dict_list = [data for data in data_dict_list if data['data_index'] in split['train']]
    val_dict_list = [data for data in data_dict_list if data['data_index'] in split['val']]
    test_dict_list = [data for data in data_dict_list if data['data_index'] in split['test']]
    assert len(train_dict_list) + len(val_dict_list) + len(test_dict_list) == len(data_dict_list)
    return train_dict_list, val_dict_list, test_dict_list


def extract_response(x, prompt_len):
    assert x.shape[0] > prompt_len
    return x[prompt_len:,:].float()


def pool(x, pooling):
    if pooling == 'mean':
        x = torch.mean(x, 0, keepdim=True)
    elif pooling in ['-3', '-2', '-1', '0', '1', '2']:
        i = int(pooling)
        x = x[i,:].unsqueeze(0)
    else:
        raise NotImplementedError(pooling)
    return x


def annotate(dict_list, labels):
    new_data = list()
    for data in dict_list:
        if data['data_index'] not in labels:
            continue
        data['annotation'] = labels[data['data_index']]
        new_data.append(data)
    return new_data


def fit_baseline(train_iter, val_iter, test_iter, eval_fn, feat_ex_fn, tokenwise, pooling=None, C=None, balance=False, log=False, other_args=None, dataname='', baseline_name='', checkpoint_folder=None):

    if log:
        run = wandb.init(project="charm")
        C = wandb.config.C
        if not tokenwise:
            pooling = wandb.config.pooling
    else:
        assert C is not None
        run = None

    assert not tokenwise or pooling is None
    X_train, Y_train = feat_ex_fn(train_iter, pooling, **other_args)
    X_val, Y_val = feat_ex_fn(val_iter, pooling, **other_args)
    X_test, Y_test = feat_ex_fn(test_iter, pooling, **other_args)

    if C >= 100000.0:
        C = +np.inf
    if balance:
        predictor = LogisticRegression(penalty='l2', C=C, class_weight='balanced', max_iter=1000)
    else:
        predictor = LogisticRegression(penalty='l2', C=C, max_iter=1000)
    predictor.fit(X_train, Y_train)
    if checkpoint_folder is not None:
        tokenwise_str = 'tokenwise' if tokenwise else ""
        nick = f"{run.name}_{run.id}" if run is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"{baseline_name}_{nick}_{tokenwise_str}_{dataname.replace('/', '_').replace('.', '_')}"
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_folder, f'{experiment_name}.pkl')
        args = {
            'C': C,
            'pooling': pooling,
            'tokenwise': tokenwise}
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump((args, predictor), handle)

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
            "test/best_aupr_hallu": test_aupr_hallu
            })
        wandb.finish()

    return train_auroc, train_aupr, train_aupr_hallu, val_auroc, val_aupr, val_aupr_hallu, test_auroc, test_aupr, test_aupr_hallu


def print_results(header, results):
    train_auroc, train_aupr, train_aupr_hallu, val_auroc, val_aupr, val_aupr_hallu, test_auroc, test_aupr, test_aupr_hallu = results
    print(f"\n--------- {header} ----------")
    print(f"Train AUROC        {train_auroc:.4f}  | Val AUROC        {val_auroc:.4f}  | Test AUROC        {test_auroc:.4f}")
    print(f"Train AUPR         {train_aupr:.4f}  | Val AUPR         {val_aupr:.4f}  | Test AUPR         {test_aupr:.4f}")
    print(f"Train AUPR (hallu) {train_aupr_hallu:.4f}  | Val AUPR (hallu) {val_aupr_hallu:.4f}  | Test AUPR (hallu) {test_aupr_hallu:.4f}")


def evaluate_predictor(predictor, X, Y):
    preds = predictor.decision_function(X)
    auroc = roc_auc_score(Y, preds)
    aupr = average_precision_score(Y, preds)
    aupr_hallu = average_precision_score(1 - Y, - preds)
    return auroc, aupr, aupr_hallu
