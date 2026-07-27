import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_path", type=str, default="./raw_data/lookback-ratio-nq-7b.pt")
    parser.add_argument("--tf_path", type=str, default="./raw_data/tf-nq-7b.jsonl")

    args = parser.parse_args()

    print("[i] Loading pre-saved LLM completions...")
    loaded = torch.load(args.dump_path, weights_only=False)
    tf = list()
    for data in loaded:
        tf.append({'data_index': data['data_index'], 'model_completion_ids': data['model_completion_ids'].tolist(), 'model_completion': data['model_completion']})

    print("[i] Saving teacher forcing data")
    offset = - 1000 if 'cnndm-7b' in args.dump_path else 0  # NOTE: Fixes indexing error in the Lookback Lens dump
    with open(args.tf_path, 'w') as fout:
        for data in tf:
            msg = '{"data_index": '
            msg += str(data['data_index']+offset)
            msg += ', "model_completion_ids": '
            msg += str(data["model_completion_ids"])
            msg += '}\n'
            fout.write(msg)