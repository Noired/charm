import argparse
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from dataset.transforms import LabelPooling


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/nq-7b-001_16bits/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    args = parser.parse_args()
    assert args.train_ratio + args.val_ratio + args.test_ratio == 1.0
    assert args.train_ratio > 0.0 and args.val_ratio > 0.0 and args.test_ratio > 0.0

    data_list = torch.load(f'{args.data_root}/raw/data_list.pt', weights_only=False)
    transform = LabelPooling()
    labels = [int(transform(data).y.item()) for data in data_list]
    index = [data.data_idx for data in data_list]
    idx_to_order = {idx: i for (i, idx) in enumerate(index)}
    
    # Carve out test set
    train, test = train_test_split(index, test_size=args.test_ratio, random_state=args.seed, stratify=labels)

    # Carve out val set from the remaining part
    val_size = (args.val_ratio/(1-args.test_ratio))
    remaining_labels = [labels[idx_to_order[idx]] for idx in train]
    train, val = train_test_split(train, test_size=val_size, random_state=args.seed, stratify=remaining_labels)

    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert len(set(train) & set(val)) == 0 and len(set(train) & set(test)) == 0 and len(set(test) & set(val)) == 0
    assert len( (set(train) | set(val) | set(test)) & set(index) ) == len(index)
    assert len(train) + len(val) + len(test) == len(index)
    splits = {
        'train': train,
        'val': val,
        'test': test}
    torch.save(splits, args.data_root+'/splits.pt')

    