#!/usr/bin/env python3
# main.py — CLI pipeline (use app.py for web version)
# Usage: python main.py [--skip-train] [--dataset cifar10|mnist|diabetes|cifar100]
import argparse, os, torch
from config import DEVICE, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, PRUNING_AMOUNTS, NUM_INFERENCE_RUNS
from models import get_model
from utils import get_loaders, load_model, ensure_dirs
from train    import train
from prune    import prune_and_save
from quantize import quantize_and_save
from evaluate import evaluate_model, save_results
from visualize import generate_plots

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--device', default=DEVICE)
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10','mnist','diabetes','cifar100'])
    args = p.parse_args()
    ensure_dirs(MODELS_DIR, RESULTS_DIR, PLOTS_DIR)

    _, test_loader  = get_loaders(args.dataset)
    baseline_path   = os.path.join(MODELS_DIR, f'baseline_best_{args.dataset}.pt')

    if args.skip_train and os.path.exists(baseline_path):
        baseline = get_model(); baseline = load_model(baseline, baseline_path, args.device)
    else:
        baseline = train(device=args.device, dataset=args.dataset)
        baseline = load_model(baseline, baseline_path, args.device)

    pruned  = prune_and_save(baseline_path, PRUNING_AMOUNTS, args.device, args.dataset)
    q_model = quantize_and_save(source_model=baseline, dataset=args.dataset)

    results = [evaluate_model(baseline, 'Baseline', test_loader, args.device, NUM_INFERENCE_RUNS)]
    for amt, pm in pruned.items():
        results.append(evaluate_model(pm, f'Pruned {int(amt*100)}%', test_loader, args.device, NUM_INFERENCE_RUNS))
    results.append(evaluate_model(q_model, 'Quantized INT8', test_loader, args.device, NUM_INFERENCE_RUNS))

    save_results(results)
    generate_plots()
    print("\n[OK] Pipeline complete!  Results -> results/  Plots -> plots/")
    print("Start web app: python app.py")

if __name__ == '__main__':
    main()
