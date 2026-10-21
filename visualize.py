import os, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import PLOTS_DIR, RESULTS_CSV
from utils import ensure_dirs

PALETTE = ['#2563EB','#16A34A','#DC2626','#D97706','#7C3AED']
DPI = 300

def _load(csv_path):
    with open(csv_path, newline='') as f: return list(csv.DictReader(f))

def _bar(ax, labels, values, title, ylabel, colors, fmt='.2f'):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors[:len(labels)], width=0.55, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
    ax.set_title(title, fontsize=12, fontweight='bold'); ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f'{val:{fmt}}', ha='center', va='bottom', fontsize=8, fontweight='bold')

def generate_plots(csv_path=RESULTS_CSV, plots_dir=PLOTS_DIR):
    ensure_dirs(plots_dir)
    if not os.path.exists(csv_path): print("[plot] No CSV yet"); return
    res     = _load(csv_path)
    labels  = [r['label'] for r in res]
    acc     = [float(r['accuracy_%'])        for r in res]
    sizes   = [float(r['model_size_mb'])     for r in res]
    times   = [float(r['inference_time_ms']) for r in res]
    energy  = [float(r['energy_kwh'])        for r in res]
    specs = [(acc,'Accuracy (%)','accuracy_comparison.png','.2f'),
             (sizes,'Size (MB)','model_size_comparison.png','.4f'),
             (times,'Time (ms)','inference_time_comparison.png','.4f'),
             (energy,'Energy (kWh)','energy_comparison.png','.2e')]
    for vals, ylabel, fname, fmt in specs:
        fig, ax = plt.subplots(figsize=(8,5), dpi=DPI)
        _bar(ax, labels, vals, ylabel, ylabel, PALETTE, fmt)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2,2, figsize=(14,9), dpi=DPI)
    fig.suptitle('Performance–Energy Trade-off', fontsize=14, fontweight='bold')
    for ax,(vals,ylabel,_,fmt) in zip(axes.flat, specs): _bar(ax,labels,vals,ylabel,ylabel,PALETTE,fmt)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir,'summary_grid.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] All plots saved -> {plots_dir}/")
