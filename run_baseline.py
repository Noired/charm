import yaml
import argparse
from argparse import Namespace
from baseline.lb_baseline_fitting import main as lb_main
from baseline.atp_baseline_fitting import main as atp_main
from baseline.lc_baseline_fitting import main as lc_main
from baseline.lc_pp_baseline_fitting import main as lc_pp_main
from baseline.lapeig_baseline_fitting import main as lapeig_main
from baseline.act_baseline_fitting import main as act_main
from baseline.baseline_fitting import main as base_main

TARGET_METRICS = {
    'test_auroc': "Test AUROC",
    'test_aupr_hallu': "Test AUPR"}
MAIN_FNS = {
    'atp': atp_main,
    'lb': lb_main,
    'lb++': lb_main,
    'lc': lc_main,
    'lc++': lc_pp_main,
    'lapeig': lapeig_main,
    'act': act_main,
    'node_avg': base_main,
    'edge_avg': base_main}

def get_name(base, cfg):
    lookup = {
        'atp': "Probas",
        'lb': "Lookback Lens",
        'lb++': "Lookback Lens++",
        'lc': "LLM-Chk-",
        'lc++': "LLM-Chk++-",
        'lapeig': "LapEig",
        'act': "Act-",
        'node_avg': "Neigh-Avg(N)",
        'edge_avg': "Neigh-Avg(E)"}
    suffix = ''
    if base == 'act':
        suffix += cfg['llm_layers']
    elif base == 'atp':
        suffix += ' (response only)' if cfg['response_only'] else ""
    elif base == 'lc' or base == 'lc++':
        suffix += str(int(cfg['llm_layers'])+1)
    return lookup[base]+suffix

assert len(TARGET_METRICS) > 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--base", type=str)
    current_args = parser.parse_args()
    config_path = current_args.config
    main = MAIN_FNS[current_args.base]

    with open(config_path) as f:
        cfgs = yaml.safe_load(f)[current_args.base]

    for cfg in cfgs:
        args = Namespace(**cfg)
        res = main(args)

        msg = f"================ {get_name(current_args.base, cfg)} =================\n"
        for metric in TARGET_METRICS:
            msg += f"{TARGET_METRICS[metric]}: {100*res[metric]:.2f}\n"
        msg += "\n"
        print(msg)
