import yaml
import argparse
import numpy as np
from argparse import Namespace
from training import main

TARGET_METRICS = {
    'test_auroc': "Test AUROC",
    'test_aupr_hallu': "Test AUPR"}

assert len(TARGET_METRICS) > 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dump", action='store_true')
    current_args = parser.parse_args()
    config_path = current_args.config
    seeds = current_args.seeds

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    if current_args.dump:
        print("[i] Dumping data to disk only...")
        cfg['dump_data_only'] = True
        cfg['on_disk'] = False
        cfg['seed'] = 0
        args = Namespace(**cfg)
        main(args)
    else:
        results = {metric: list() for metric in TARGET_METRICS}
        for seed in range(seeds):
            cfg['seed'] = seed
            args = Namespace(**cfg)
            res = main(args)
            for metric in TARGET_METRICS:
                results[metric].append(res[metric])

        msg = "=================================\n"
        for metric in TARGET_METRICS:
            mean = np.mean(results[metric])
            std = np.std(results[metric])
            msg += f"{TARGET_METRICS[metric]}: {100*mean:.2f} ± {100*std:.2f}\n"
        msg += f"(results for {len(results[metric])} runs over seeds: {list(range(seeds))})\n"
        msg += "================================="
        print(msg)
