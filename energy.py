import time, torch

def measure_energy_and_time(model, dataloader, device='cpu', num_runs=100, label='model'):
    model.eval(); device = torch.device(device); model = model.to(device)
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(project_name=label, output_dir='results',
                                   log_level='error', save_to_file=True, tracking_mode='process')
        tracker.start()
    except Exception:
        tracker = None

    times, count = [], 0
    with torch.no_grad():
        for x, *_ in dataloader:
            if count >= num_runs: break
            x = x.to(device)
            t0 = time.perf_counter(); model(x)
            times.append((time.perf_counter()-t0)*1000); count += 1

    energy_kwh, co2_kg = 0.0, 0.0
    if tracker:
        co2_kg = round(float(tracker.stop() or 0.0), 8)
        try:
            import pandas as pd, glob
            files = sorted(glob.glob('results/emissions*.csv'))
            if files: energy_kwh = round(float(pd.read_csv(files[-1])['energy_consumed'].iloc[-1]), 8)
        except Exception: pass

    return {'energy_kwh': energy_kwh,
            'inference_time_ms': round(sum(times)/len(times),4) if times else 0.0,
            'co2_kg': co2_kg}
