import os, copy, torch
import torch.nn.utils.prune as prune
from config import MODELS_DIR, PRUNING_AMOUNTS
from models import get_model
from utils import save_model, load_model


def _layers(model):
    return [(m,'weight') for m in model.modules()
            if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))]

def apply_pruning(model, amount):
    pm = copy.deepcopy(model)
    prune.global_unstructured(_layers(pm), pruning_method=prune.L1Unstructured, amount=amount)
    for m,_ in _layers(pm): prune.remove(m,'weight')
    zeros = sum((m.weight==0).sum().item() for m,_ in _layers(pm))
    total = sum(m.weight.numel()           for m,_ in _layers(pm))
    print(f"[prune] {amount*100:.0f}% -> sparsity: {100.*zeros/total:.2f}%")
    return pm

def prune_and_save(baseline_path, pruning_amounts=None, device='cpu', dataset='cifar10'):
    if pruning_amounts is None: pruning_amounts = PRUNING_AMOUNTS
    baseline = get_model()
    baseline = load_model(baseline, baseline_path, device=device)
    pruned = {}
    for amt in pruning_amounts:
        pm = apply_pruning(baseline, amt); pm.eval()
        save_model(pm, os.path.join(MODELS_DIR, f'pruned_{int(amt*100)}_{dataset}.pt'))
        pruned[amt] = pm
    return pruned
