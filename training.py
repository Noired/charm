import argparse
import os
import subprocess
import torch
import numpy as np
import wandb
import time

from datetime import datetime
from tqdm import tqdm
from model.mp import CHARM
from dataset.dataset import AttentionDataset, OnDiskAttentionDataset
from dataset.transforms import *
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch import optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric import seed_everything
from transformers import get_scheduler
from dataset.prebatch import Prebatcher
from baseline.act_baseline_fitting import extract_layers


def get_optimizer(model, learning_rate, weight_decay, weight_decay_target, weight_decay_for_target):
    if weight_decay > 0.0 or weight_decay_for_target > 0.0:
        if weight_decay_target == "all":
            print(f"[i] Optimizer: AdamW, using weight decay {weight_decay} for all parameters")
            return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif weight_decay_target == "attr_encoders.act":
            l2_params = []
            no_l2_params = []
            for name, param in model.named_parameters():
                if (
                    'attr_encoders.act' in name and
                    'weight' in name and 
                    param.requires_grad
                ):
                    l2_params.append(param)
                else:
                    no_l2_params.append(param)
            print(f"[i] Optimizer: Adam, using weight decay {weight_decay_for_target} for target {weight_decay_target} and {weight_decay} for all other parameters (it does not take any effect if this parameter group is not used).")
            optimizer = torch.optim.Adam([
                {'params': l2_params},
                {'params': no_l2_params, 'weight_decay': weight_decay}
            ], lr=learning_rate, weight_decay=weight_decay_for_target)

            return optimizer
        else:
            raise ValueError(f"Weight decay target {weight_decay_target} not currently supported.")
    else:
        print(f"[i] Optimizer: Adam, using no weight decay")
        return optim.Adam(model.parameters(), lr=learning_rate)


def get_git_sha():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return sha
    except Exception as e:
        print(f"Error getting git SHA: {e}")
        return None

def checkpoint(model, best_epoch, track_dict, checkpoint_pack):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'tracking': track_dict,
        'git_sha': checkpoint_pack['git_sha'],
        'args': checkpoint_pack['args']}
    torch.save(save_dict, checkpoint_pack['checkpoint_path'])

def get_transforms(args):

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
    if args.attention_threshold > 0.0:
        transform_list.append(ThresholdAttention(args.attention_threshold))
    if not args.disable_marks:
        transform_list += [MarkPrompt(), MarkStartPrompt(), MarkStartResponse()]
    if args.mark_prompt_nodes:
        assert args.disable_marks
        transform_list.append(MarkPrompt())
    if args.mark_start_prompt:
        assert args.disable_marks
        transform_list.append(MarkStartPrompt())
    if args.mark_prompt_edges:
        assert args.disable_marks
        transform_list.append(MarkPromptEdges())
    T = Compose(transform_list)
    transform_name = '___'.join([repr(t) for t in transform_list])
    args.transform_name = transform_name
    print(f'[i] PreTransforms: {transform_name}')

    # Transforms
    transform_list_on_the_fly = list()

    T_ = Compose(transform_list_on_the_fly)
    online_transform_name = '___'.join([repr(t) for t in transform_list_on_the_fly])
    args.online_transform_name = online_transform_name
    print(f'[i] Transforms: {online_transform_name}')

    return T, transform_name, T_, online_transform_name

def get_data(args, T, transform_name, T_, online_transform_name):
    # Dataset set up
    if args.on_disk:
        # Are we hydrating anything?
        hydrate_suffix = ''
        if len(args.acts) > 0 and len(args.act_llm_layers) > 0:
            print(f'[i] Hydrating with ACTs for layers {args.act_llm_layers}')
            hydrate_suffix += f'__acts_{args.act_llm_layers}_{args.on_x}'
        train_load_dir = os.path.join(args.data, 'processed', f'train_data_{transform_name}{hydrate_suffix}')
        val_load_dir = os.path.join(args.data, 'processed', f'val_data_{transform_name}{hydrate_suffix}')
        test_load_dir = os.path.join(args.data, 'processed', f'test_data_{transform_name}{hydrate_suffix}')
        train_dataset = OnDiskAttentionDataset(train_load_dir, transform=T_)
        val_dataset = OnDiskAttentionDataset(val_load_dir, transform=T_)
        test_dataset = OnDiskAttentionDataset(test_load_dir, transform=T_)

    else:
        train_dataset = AttentionDataset(root=args.data, split='train', pre_transform=T, transform=T_, transform_name=transform_name, force_reload=False)
        val_dataset = AttentionDataset(root=args.data, split='val', pre_transform=T, transform=T_, transform_name=transform_name, force_reload=False)
        test_dataset = AttentionDataset(root=args.data, split='test', pre_transform=T, transform=T_, transform_name=transform_name, force_reload=False)
        
        # Hydration of other artefacts
        if len(args.acts) > 0 and len(args.act_llm_layers) > 0:
            print(f'[i] Hydrating with ACTs from {args.acts} for layers {args.act_llm_layers}')
            hydration_name = f'acts_{args.act_llm_layers}_{args.on_x}'
            act_list = torch.load(args.acts, weights_only=False)
            selection_fn = lambda x: extract_layers(x['act'], x['layers'], args.act_llm_layers)    
            train_dataset.hydrate(act_list, 'act', on_x=args.on_x, nullify_prompt=args.nullify_prompt, selection_fn=selection_fn, hydration_name=hydration_name)
            val_dataset.hydrate(act_list, 'act', on_x=args.on_x, nullify_prompt=args.nullify_prompt, selection_fn=selection_fn, hydration_name=hydration_name)
            test_dataset.hydrate(act_list, 'act', on_x=args.on_x, nullify_prompt=args.nullify_prompt, selection_fn=selection_fn, hydration_name=hydration_name)
            print(f'[i] ACTs shape: {train_dataset[0]["act"].shape}') if not args.on_x else None

    dims_for_linear_layers = dict()
    dims_for_linear_layers['act'] = train_dataset[0]['act'].shape[-1] if (len(args.acts) > 0 and len(args.act_llm_layers) > 0 and not args.on_x) else None
    assert args.on_x or any(dim is not None for dim in dims_for_linear_layers.values()), "If args.on_x is False, then at least one value in dims_for_linear_layers must not be None"

    return train_dataset, val_dataset, test_dataset, dims_for_linear_layers

def get_loaders(args, train_dataset, val_dataset, test_dataset):
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    if args.train_prebatch:
        train_prebatcher = Prebatcher(train_loader, args.train_prebatch_interval)
        train_loader = train_prebatcher.prebatched_loader
    else:
        train_prebatcher = None
    if args.eval_prebatch:
        val_loader = Prebatcher.prebatch(val_loader, shuffle=False, shutdown_loader=True)
        test_loader = Prebatcher.prebatch(test_loader, shuffle=False, shutdown_loader=True)
    print(f'[i] Data (train): {train_dataset}')
    print(f'[i] Data (val): {val_dataset}')
    print(f'[i] Data (test): {test_dataset}')
    return train_loader, val_loader, test_loader, train_prebatcher

def get_model(args, dataset, dims_for_linear_layers):
    # Model
    in_dim_x = dataset[0].x.shape[1]
    in_dim_e = dataset[0].edge_attr.shape[1]
    num_classes = 1
    print(f"[i] {in_dim_x} input node feats.")
    print(f"[i] {in_dim_e} input edge feats.")
    model = CHARM(
        args.num_layers,
        in_dim_x,
        in_dim_e,
        args.hidden_dim,
        num_classes,
        dropout=args.dropout_rate,
        aggr=args.aggr,
        flow=args.flow,
        readout=(args.readout if not args.tokenwise else 'none'),
        activation=args.activation,
        layer=args.layer,
        encoder=args.encoder,
        on_x=args.on_x,
        dims_for_linear_layers=dims_for_linear_layers,
        batch_norm=True if ('batch_norm' in args and args.batch_norm == 'yes') else False,
        residual=True if ('residual' in args and args.residual == 'yes') else False,
        attr_encoder_location=args.attr_encoder_location if ('attr_encoder_location' in args) else 'beginning',
        cat_attr=True if ('cat_attr' in args and args.cat_attr == 'yes') else False)
    print(f'Model: {model}')
    return model

def prepare_args(data):
    artifacts = {
        'act': data.act.to(torch.float32) if hasattr(data, 'act') else None}
    args = (data.x.to(torch.float32), data.edge_index, data.edge_attr.to(torch.float32), data.batch, data.response_index, data.ptr, data.prompt_len, artifacts)
    return args

def train(model, loader, optimizer, criterion, device, sched=None, verbose=False):
    """Train for one epoch. Additionally time everything"""
    model.train()
    total_loss = 0.0
    load_time = 0.0
    fwd_time = 0.0
    bwd_time = 0.0
    t_start = time.time()
    for data in tqdm(loader):  
        # --> fwd
        t0 = time.time()
        data = data.to(device)
        t1 = time.time()
        optimizer.zero_grad()
        t2 = time.time()
        args = prepare_args(data)
        out = model(*args)
        t3 = time.time()
        # <-- bwd
        loss = criterion(out.squeeze(), data.y)
        t4 = time.time()
        loss.backward()
        optimizer.step()
        t5 = time.time()
        total_loss += loss.item()
        if sched is not None:
            sched.step()
        load_time += t1 - t0
        fwd_time += t3 - t2
        bwd_time += t5 - t4
    t_end = time.time()

    if verbose:
        print(f'[i] Load time: {load_time:.4f}')
        print(f'[i] Fwd time:  {fwd_time:.4f}')
        print(f'[i] Bwd time:  {bwd_time:.4f}')
        print(f'[i] Other:     {t_end - t_start - load_time - fwd_time - bwd_time:.4f}')
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model calculating loss and perf metric on an eval set."""
    # Inference
    model.eval()
    total_loss = 0.0
    ys, preds = list(), list()
    for data in tqdm(loader):
        data = data.to(device)
        args = prepare_args(data)
        out = model(*args)
        loss = criterion(out.squeeze(), data.y)
        total_loss += loss.item()
        ys.append(data.y.unsqueeze(-1))
        preds.append(out)
    # Calculate performance metrics
    avg_loss = total_loss / len(loader)
    ys = torch.cat(ys, dim=0).squeeze(1).cpu().numpy()
    preds = torch.cat(preds, dim=0).squeeze(1).cpu().numpy()
    auroc = roc_auc_score(ys, preds)
    aupr = average_precision_score(ys, preds)
    aupr_hallu = average_precision_score(1 - ys, - preds)
    return avg_loss, auroc, aupr, aupr_hallu

def train_model(model, train_loader, val_loader, test_loader, num_epochs=100, lr=0.0001, scheduler='none', weight_decay=0.0, weight_decay_target='all', weight_decay_for_target=0.0, balance=False, train_prebatcher=None, patience=20, log=False, checkpoint_pack=None, verbose=False):

    t_start = time.time()

    # Model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set-up optimisation
    optimizer = get_optimizer(model, lr, weight_decay, weight_decay_target, weight_decay_for_target)
    if scheduler != 'none':
        if scheduler == 'cosine':
            sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs)
        elif scheduler == 'plateau':
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5)
        elif scheduler == 'cosine_warm':
            num_training_steps = len(train_loader) * num_epochs
            num_warmup_steps = int(0.1 * num_training_steps)
            sched = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        else:
            raise ValueError(f"Scheduler {scheduler} not currently supported.")
    else:
        sched = None

    if balance:
        pos_weight = calculate_class_weight(train_loader).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    train_losses = list()
    val_aurocs, test_aurocs = list(), list()
    val_auprs, test_auprs = list(), list()
    val_auprs_hallu, test_auprs_hallu = list(), list()
    best_epoch = 0
    best_val_perf = None
    no_improvement = 0
    interrupt = False
    for epoch in range(0, num_epochs):

        print(f"\n____ Epoch {epoch:03d} __________________________________________")
        train_loss = train(model, train_loader, optimizer, criterion, device, sched if 'warm' in scheduler else None, verbose=verbose)
        train_losses.append(train_loss)
        print(f"Train Loss (Avg per batch): {train_loss:.4f}")
        if scheduler in ['cosine']:
            sched.step()
        if epoch > 0 and train_prebatcher is not None:
            train_loader = train_prebatcher.refresh(epoch)

        val_loss, val_auroc, val_aupr, val_aupr_hallu = evaluate(model, val_loader, criterion, device)
        test_loss, test_auroc, test_aupr, test_aupr_hallu = evaluate(model, test_loader, criterion, device)
        print(f"Val Loss:   {val_loss:.4f} | Val AUROC:   {val_auroc:.4f} | Val AUPR:   {val_aupr:.4f} | Val AUPR (hallu):   {val_aupr_hallu:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test AUROC:  {test_auroc:.4f} | Test AUPR:   {test_aupr:.4f} | Test AUPR (hallu):   {test_aupr_hallu:.4f}")
        val_aurocs.append(val_auroc)
        test_aurocs.append(test_auroc)
        val_auprs.append(val_aupr)
        test_auprs.append(test_aupr)
        val_auprs_hallu.append(val_aupr_hallu)
        test_auprs_hallu.append(test_aupr_hallu)
        if scheduler in ['plateau']:
            sched.step(val_aupr_hallu)

        track_dict = {
                "epoch": epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                "train/loss": train_losses[-1],
                "val/loss": val_loss,
                "test/loss": test_loss,
                "val/auroc": val_auroc,
                "test/auroc": test_auroc,
                "val/aupr": val_aupr,
                "test/aupr": test_aupr,
                "val/aupr_hallu": val_aupr_hallu,
                "test/aupr_hallu": test_aupr_hallu,
                "val/best_auroc": val_aurocs[best_epoch],
                "test/best_auroc": test_aurocs[best_epoch],
                "val/best_aupr": val_auprs[best_epoch],
                "test/best_aupr": test_auprs[best_epoch],
                "val/best_aupr_hallu": val_auprs_hallu[best_epoch],
                "test/best_aupr_hallu": test_auprs_hallu[best_epoch],
                "best_epoch": best_epoch}

        if best_val_perf is None:
            best_val_perf = val_aupr_hallu
            if checkpoint_pack is not None:
                checkpoint(model, best_epoch, track_dict, checkpoint_pack)
        elif best_val_perf < val_aupr_hallu:
            best_val_perf = val_aupr_hallu
            best_epoch = epoch
            no_improvement = 0
            if checkpoint_pack is not None:
                checkpoint(model, best_epoch, track_dict, checkpoint_pack)
        else:
            no_improvement += 1
            interrupt = no_improvement >= patience

        if log:
            wandb.log(track_dict)

        if interrupt:
            print(f"[i] No further validation improvement after {no_improvement} epochs. Interrupting training.")
            break

    print("-------------------")
    print(f"Best epoch: {best_epoch:02d}")
    print(f"Val AUROC {val_aurocs[best_epoch]:.4f} | Test AUROC {test_aurocs[best_epoch]:.4f}")
    print(f"Val AUPR {val_auprs[best_epoch]:.4f} | Test AUPR {test_auprs[best_epoch]:.4f}")
    print(f"Val AUPR (hallu) {val_auprs_hallu[best_epoch]:.4f} | Test AUPR (hallu) {test_auprs_hallu[best_epoch]:.4f}")

    t_end = time.time()
    if verbose:
        print(f'[i] Overall elapsed time: {t_end - t_start}')

    result_dict = {
        'best_epoch': best_epoch,
        'val_auroc': val_aurocs[best_epoch],
        'test_auroc': test_aurocs[best_epoch],
        'val_aupr': val_auprs[best_epoch],
        'test_aupr': test_auprs[best_epoch],
        'val_aupr_hallu': val_auprs_hallu[best_epoch],
        'test_aupr_hallu': test_auprs_hallu[best_epoch]}

    return result_dict

def calculate_class_weight(loader):
    ys = list()
    for data in loader:
        ys.append(data.y.long())
    ys = torch.cat(ys, 0).numpy()
    freqs = np.bincount(ys)
    print(f"[i] Train label freq: {freqs}")
    weights = len(ys) / (len(freqs) * freqs)
    assert len(weights) == 2
    pos_weight = weights[1] / weights[0]
    print(f"[i] Positive weight: {pos_weight}")
    return torch.tensor([pos_weight])

def main(args):

    print('[i] Args:')
    print(args)

    # Seeding
    seed_everything(args.seed)
    # Check consistency between dump_data_only and on_disk
    if args.dump_data_only:
        assert not args.on_disk

    # Transforms
    T, transform_name, T_, online_transform_name = get_transforms(args)

    # Datasets
    train_dataset, val_dataset, test_dataset, dims_for_linear_layers = get_data(args, T, transform_name, T_, online_transform_name)

    if args.dump_data_only:
        assert not args.on_disk
        where = train_dataset.dump_data_list()
        print(f"[i] Dumped train data list at: {where}")
        where = val_dataset.dump_data_list()
        print(f"[i] Dumped val data list at: {where}")
        where = test_dataset.dump_data_list()
        print(f"[i] Dumped test data list at: {where}")

    else:
        # Loaders
        train_loader, val_loader, test_loader, train_prebatcher = get_loaders(args, train_dataset, val_dataset, test_dataset)

        # Model
        model = get_model(args, train_dataset, dims_for_linear_layers)

        # Logging
        if args.log:
            run = wandb.init(
                project="charm",
                config=args)
        else:
            run = None

        checkpoint_path = None
        if args.checkpoint_best:  # Experiment naming
            tokenwise_str = 'tokenwise' if args.tokenwise else ""
            nick = f"{run.name}_{run.id}" if run is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"charm_{args.seed}_{nick}_{tokenwise_str}_{args.data.replace('/', '_').replace('.', '_')}"
            os.makedirs(args.checkpoint_folder, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_folder, f'{experiment_name}.pt')
            checkpoint_pack = {
                'checkpoint_path': checkpoint_path,
                'args': args,
                'git_sha': get_git_sha()}

        # Training
        result_dict = train_model(model, train_loader, val_loader, test_loader, num_epochs=args.num_epochs, lr=args.learning_rate, scheduler=args.scheduler, weight_decay=args.weight_decay, weight_decay_target=args.weight_decay_target, weight_decay_for_target=args.weight_decay_for_target, balance=args.balance, train_prebatcher=train_prebatcher, patience=args.patience, log=args.log, checkpoint_pack=checkpoint_pack, verbose=args.verbose)

        # Shutdown loaders
        if train_prebatcher is not None:
            train_prebatcher.shutdown()
        else:
            del train_loader
        del val_loader
        del test_loader
        
        # Close logger
        if args.log:
            wandb.finish()
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data and task configuration
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str)
    parser.add_argument("--tokenwise", action='store_true')
    parser.add_argument("--disable_marks", action='store_true')
    parser.add_argument("--mark_prompt_nodes", action='store_true')
    parser.add_argument("--mark_start_prompt", action='store_true')
    parser.add_argument("--mark_prompt_edges", action='store_true')
    parser.add_argument("--attention_threshold", type=float, default=0.05)
    parser.add_argument("--llm_layers", type=str, default='all')

    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--scheduler", type=str, default='none')
    parser.add_argument("--balance", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_prebatch", action='store_true')
    parser.add_argument("--train_prebatch_interval", type=int, default=10)
    parser.add_argument("--eval_prebatch", action='store_true')
    
    # Regularisation hypers
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--weight_decay_for_target", type=float, default=0.001)
    parser.add_argument("--weight_decay_target", type=str, default="attr_encoders.act")
    
    # Architectural hypers
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--flow", type=str, default='source_to_target')
    parser.add_argument("--readout", type=str, default='mean')
    parser.add_argument("--aggr", type=str, default='mean')
    parser.add_argument("--activation", type=str, default='none')
    parser.add_argument("--layer", type=str, default='custom')
    parser.add_argument("--encoder", type=str, default='linear')
    parser.add_argument("--batch_norm", type=str, default='yes')
    parser.add_argument("--residual", type=str, default='yes')
    parser.add_argument("--attr_encoder_location", type=str, default='beginning')
    parser.add_argument("--cat_attr", type=str, default='yes')

    # Additional artifacts
    parser.add_argument("--on_x", action='store_true')
    parser.add_argument("--acts", type=str)
    parser.add_argument("--act_llm_layers", type=str, default="24")
    
    # Logging
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Checkpointing
    parser.add_argument("--checkpoint_folder", type=str, default='./checkpoints')
    parser.add_argument("--checkpoint_best", action='store_true')

    # On-disk data
    parser.add_argument("--dump_data_only", action="store_true")
    parser.add_argument("--on_disk", action="store_true")

    # Parse and pring args
    args = parser.parse_args()
    _ = main(args)