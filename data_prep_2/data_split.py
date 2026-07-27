import argparse
import torch

from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/movies-5k-mistral-7b-i-001_16bits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_offset", type=int, default=5000)

    args = parser.parse_args()
    assert args.train_ratio + args.val_ratio == 1.0
    assert args.train_ratio > 0.0 and args.val_ratio > 0.0

    data_list = torch.load(f'{args.data_root}/raw/data_list.pt', weights_only=False)
    labels = [data.y.item() for data in data_list if data.data_idx < args.test_offset]
    index = [data.data_idx for data in data_list if data.data_idx < args.test_offset]
    idx_to_order = {idx: i for (i, idx) in enumerate(index)}
    
    # Carve out val set
    train, val = train_test_split(index, test_size=args.val_ratio, random_state=args.seed, stratify=labels)

    # Construct test idx
    test = [data.data_idx for data in data_list if data.data_idx >= args.test_offset]

    assert len(set(train) & set(val)) == 0 and len(set(train) & set(test)) == 0 and len(set(test) & set(val)) == 0
    index = [data.data_idx for data in data_list]
    assert len( (set(train) | set(val) | set(test)) & set(index) ) == len(index)
    splits = {
        'train': train,
        'val': val,
        'test': test}
    torch.save(splits, args.data_root+'/splits.pt')

    