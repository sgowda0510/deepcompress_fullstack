import os, csv, torch
from config import DEVICE, RESULTS_CSV, RESULTS_DIR, NUM_INFERENCE_RUNS
from utils import get_model_size_mb, ensure_dirs
from energy import measure_energy_and_time

def compute_accuracy(model, loader, device='cpu'):
    model.eval(); device = torch.device(device); model = model.to(device)
    c, t = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += model(x).max(1)[1].eq(y).sum().item(); t += y.size(0)
    return round(100.0*c/t, 4)

def evaluate_model(model, label, test_loader, device=DEVICE, num_runs=NUM_INFERENCE_RUNS):
    print(f"\n[eval] {label}")
    acc  = compute_accuracy(model, test_loader, device)
    size = get_model_size_mb(model)
    e    = measure_energy_and_time(model, test_loader, device=device, num_runs=num_runs, label=label)
    r    = {'label': label, 'accuracy_%': acc, 'model_size_mb': size,
            'inference_time_ms': e['inference_time_ms'], 'energy_kwh': e['energy_kwh'], 'co2_kg': e['co2_kg']}
    print(f"  Acc:{acc:.2f}%  Size:{size:.4f}MB  Time:{e['inference_time_ms']:.2f}ms")
    return r

def save_results(results, csv_path=RESULTS_CSV):
    ensure_dirs(RESULTS_DIR)
    fields = ['label','accuracy_%','model_size_mb','inference_time_ms','energy_kwh','co2_kg']
    with open(csv_path,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
    print(f"[eval] Results -> {csv_path}")
