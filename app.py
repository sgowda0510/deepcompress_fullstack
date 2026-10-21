"""
app.py — DeepCompress Full Stack Web Application
─────────────────────────────────────────────────
Run:  python app.py
Open: http://localhost:5000
"""
import os, csv, json, glob, time, threading
from flask import Flask, render_template, request, jsonify, send_file
try:
    from flask_cors import CORS
    has_cors = True
except ImportError:
    has_cors = False

app = Flask(__name__)
if has_cors:
    CORS(app)

# ── Global pipeline state ────────────────────────────────────────────────────
state = {
    'running':  False,
    'step':     'idle',
    'percent':  0,
    'logs':     [],
    'results':  [],
    'complete': False,
    'error':    None,
    'dataset':  'cifar10',
}

for d in ['uploads','results','models','plots']:
    os.makedirs(d, exist_ok=True)


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── API routes ────────────────────────────────────────────────────────────────
@app.route('/api/status')
def api_status():
    return jsonify(state)


@app.route('/api/datasets')
def api_datasets():
    default_datasets = [
        {'id':'cifar10',  'name':'CIFAR-10',         'type':'Image',   'classes':10,  'samples':'60,000', 'size':'170 MB', 'desc':'Standard benchmark · 10 object categories'},
        {'id':'mnist',    'name':'MNIST',             'type':'Image',   'classes':10,  'samples':'70,000', 'size':'11 MB',  'desc':'Handwritten digits · Perfect for live demo'},
        {'id':'diabetes', 'name':'Diabetes (Medical)','type':'Tabular', 'classes':2,   'samples':'768',    'size':'<1 MB',  'desc':'8 health features · Predict diabetes'},
        {'id':'cifar100', 'name':'CIFAR-100',         'type':'Image',   'classes':100, 'samples':'60,000', 'size':'170 MB', 'desc':'Tiny-ImageNet proxy · 100 categories'},
    ]
    custom_reg = os.path.join('data', 'custom_datasets.json')
    if os.path.exists(custom_reg):
        try:
            with open(custom_reg, 'r') as f:
                custom_list = json.load(f)
                default_datasets.extend(custom_list)
        except Exception:
            pass
    return jsonify(default_datasets)


@app.route('/api/datasets/upload', methods=['POST'])
def api_upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV dataset file (.csv)'}), 400

    import re, pandas as pd
    ds_name = request.form.get('name', '').strip()
    if not ds_name:
        ds_name = os.path.splitext(file.filename)[0].replace('_', ' ').title()
    
    ds_id = re.sub(r'[^a-zA-Z0-9_]', '', ds_name.lower().replace(' ', '_'))
    if not ds_id:
        ds_id = f"custom_{int(time.time())}"

    save_path = os.path.join('data', f"{ds_id}.csv")
    os.makedirs('data', exist_ok=True)
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)
        samples = len(df)
        features = df.shape[1] - 1
        classes = max(len(df.iloc[:, -1].unique()), 2)
        desc = request.form.get('desc', '').strip() or f"{features} features, {classes} classes"
        file_size_kb = round(os.path.getsize(save_path) / 1024, 1)
        size_str = f"{file_size_kb} KB" if file_size_kb < 1024 else f"{round(file_size_kb/1024, 1)} MB"
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400

    entry = {
        'id': ds_id,
        'name': ds_name,
        'type': 'Tabular',
        'classes': int(classes),
        'samples': f"{samples:,}",
        'size': size_str,
        'desc': desc,
        'custom': True
    }

    custom_reg = os.path.join('data', 'custom_datasets.json')
    custom_list = []
    if os.path.exists(custom_reg):
        try:
            with open(custom_reg, 'r') as f:
                custom_list = json.load(f)
        except Exception:
            pass

    custom_list = [c for c in custom_list if c['id'] != ds_id]
    custom_list.append(entry)
    with open(custom_reg, 'w') as f:
        json.dump(custom_list, f, indent=2)

    return jsonify({'success': True, 'dataset': entry})


@app.route('/api/models')
def api_models():
    files = glob.glob('models/*.pt')
    return jsonify([{
        'name':    os.path.basename(f),
        'size_mb': round(os.path.getsize(f)/(1024**2), 2),
        'modified':time.strftime('%d %b %Y', time.localtime(os.path.getmtime(f)))
    } for f in sorted(files)])


@app.route('/api/results')
def api_results():
    p = 'results/experiment_results.csv'
    if not os.path.exists(p): return jsonify([])
    with open(p, newline='') as f:
        return jsonify(list(csv.DictReader(f)))


@app.route('/api/run', methods=['POST'])
def api_run():
    global state
    if state['running']:
        return jsonify({'error': 'Already running'}), 400
    data = request.json or {}
    state = {
        'running': True, 'step': 'Starting...', 'percent': 0,
        'logs': [], 'results': [], 'complete': False,
        'error': None, 'dataset': data.get('dataset','cifar10'),
    }
    threading.Thread(
        target=_run_pipeline,
        args=(data.get('dataset','cifar10'),
              data.get('skip_train', False),
              data.get('pruning', [0.2, 0.4]),
              data.get('device', 'cpu')),
        daemon=True
    ).start()
    return jsonify({'status': 'started'})


@app.route('/api/download/<path:filename>')
def api_download(filename):
    p = os.path.join('models', filename)
    if os.path.exists(p): return send_file(p, as_attachment=True)
    return jsonify({'error': 'Not found'}), 404


@app.route('/api/reset', methods=['POST'])
def api_reset():
    global state
    state = {'running':False,'step':'idle','percent':0,'logs':[],
             'results':[],'complete':False,'error':None,'dataset':'cifar10'}
    return jsonify({'status': 'reset'})


# ── Pipeline runner ───────────────────────────────────────────────────────────
def _log(msg, level='info'):
    state['logs'].append({'time': time.strftime('%H:%M:%S'), 'msg': msg, 'level': level})
    try:
        print(f"  [{level.upper()}] {msg}")
    except Exception:
        try:
            print(f"  [{level.upper()}] {msg.encode('ascii', 'replace').decode('ascii')}")
        except Exception:
            pass

def _step(name, pct):
    state['step'] = name
    state['percent'] = pct
    _log(f"-> {name}", 'info')


def _run_pipeline(dataset, skip_train, pruning_amounts, device):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        import torch
        from config import MODELS_DIR, DATA_DIR
        from utils import get_loaders, load_model, ensure_dirs
        from utils.helpers import get_model_size_mb, save_model
        from models import get_model
        from models.diabetes_net import get_diabetes_model
        from train    import train as do_train
        from prune    import apply_pruning
        from quantize import quantize_model, get_quantized_size_mb
        from evaluate import compute_accuracy
        from energy   import measure_energy_and_time

        ensure_dirs(MODELS_DIR, 'results', 'plots')
        baseline_path = os.path.join(MODELS_DIR, f'baseline_best_{dataset}.pt')

        # ── 1. Load dataset ───────────────────────────────────────────────────
        _step('Loading dataset', 5)
        train_loader, test_loader = get_loaders(dataset)
        _log(f'{dataset.upper()} loaded successfully', 'success')

        # ── 2. Train ──────────────────────────────────────────────────────────
        _step('Training baseline model', 10)
        if skip_train and os.path.exists(baseline_path):
            _log('Loading saved checkpoint (skip-train)', 'info')
            custom_csv = os.path.join(DATA_DIR, f"{dataset}.csv")
            if dataset == 'diabetes':
                baseline = get_diabetes_model()
            elif os.path.exists(custom_csv):
                import pandas as pd
                df = pd.read_csv(custom_csv)
                feats = df.shape[1] - 1
                cls_count = max(len(df.iloc[:, -1].unique()), 2)
                baseline = get_diabetes_model(input_features=feats, num_classes=cls_count)
            else:
                num_cls = 100 if dataset == 'cifar100' else 10
                baseline = get_model(num_classes=num_cls)
            baseline = load_model(baseline, baseline_path, device=device)
        else:
            epochs = 5 if dataset == 'diabetes' else 3
            if skip_train and not os.path.exists(baseline_path):
                _log(f'No checkpoint found for {dataset}. Training initial baseline ({epochs} epochs)...', 'warning')
            else:
                _log(f'Training on {dataset.upper()} for {epochs} epochs...', 'info')
            baseline = do_train(num_epochs=epochs, device=device, dataset=dataset)
            baseline = load_model(baseline, baseline_path, device=device)
        _log('Training complete [OK]', 'success')

        # ── 3. Baseline evaluation ────────────────────────────────────────────
        _step('Evaluating baseline', 42)
        base_acc  = compute_accuracy(baseline, test_loader, device)
        base_size = get_model_size_mb(baseline)
        e_b = measure_energy_and_time(baseline, test_loader, device=device, num_runs=25, label='baseline')
        _log(f'Baseline -> Acc:{base_acc:.2f}%  Size:{base_size:.4f}MB  Time:{e_b["inference_time_ms"]:.2f}ms', 'success')

        results = [{'label':'Baseline','accuracy_%':base_acc,'model_size_mb':base_size,
                    'inference_time_ms':e_b['inference_time_ms'],'energy_kwh':e_b['energy_kwh'],'co2_kg':e_b['co2_kg']}]
        state['results'] = results[:]

        # ── 4. Pruning ────────────────────────────────────────────────────────
        _step('Pruning models', 55)
        for amt in pruning_amounts:
            _log(f'Pruning at {int(amt*100)}%...', 'info')
            pm      = apply_pruning(baseline, amt); pm.eval()
            p_acc   = compute_accuracy(pm, test_loader, device)
            p_size  = get_model_size_mb(pm)
            e_p     = measure_energy_and_time(pm, test_loader, device=device, num_runs=25, label=f'p{int(amt*100)}')
            save_model(pm, os.path.join(MODELS_DIR, f'pruned_{int(amt*100)}_{dataset}.pt'))
            results.append({'label':f'Pruned {int(amt*100)}%','accuracy_%':p_acc,'model_size_mb':p_size,
                            'inference_time_ms':e_p['inference_time_ms'],'energy_kwh':e_p['energy_kwh'],'co2_kg':e_p['co2_kg']})
            state['results'] = results[:]
            _log(f'Pruned {int(amt*100)}% -> Acc:{p_acc:.2f}%  Time:{e_p["inference_time_ms"]:.2f}ms [OK]','success')

        # ── 5. Quantization ───────────────────────────────────────────────────
        _step('Quantizing to INT8', 78)
        _log('Applying dynamic INT8 quantization...', 'info')
        q_model = quantize_model(baseline)
        q_acc   = compute_accuracy(q_model, test_loader, device)
        q_size  = get_quantized_size_mb(q_model)
        e_q     = measure_energy_and_time(q_model, test_loader, device=device, num_runs=25, label='quant')
        torch.save(q_model.state_dict(), os.path.join(MODELS_DIR, f'quantized_{dataset}.pt'))
        results.append({'label':'Quantized INT8','accuracy_%':q_acc,'model_size_mb':q_size,
                        'inference_time_ms':e_q['inference_time_ms'],'energy_kwh':e_q['energy_kwh'],'co2_kg':e_q['co2_kg']})
        state['results'] = results[:]
        _log(f'Quantized INT8 -> Acc:{q_acc:.2f}%  Size:{q_size:.4f}MB [OK]','success')

        # ── 6. Save CSV + plots ───────────────────────────────────────────────
        _step('Saving results and plots', 92)
        fields = ['label','accuracy_%','model_size_mb','inference_time_ms','energy_kwh','co2_kg']
        with open('results/experiment_results.csv','w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
        try:
            from visualize import generate_plots; generate_plots()
        except Exception: pass
        _log('Results and plots saved [OK]', 'success')

        # ── Done ──────────────────────────────────────────────────────────────
        state['complete'] = True
        state['running']  = False
        state['percent']  = 100
        state['step']     = 'Complete'
        _log('Pipeline complete! [OK]', 'success')

    except Exception as ex:
        import traceback
        state['error']   = str(ex)
        state['running'] = False
        _log(f'ERROR: {ex}', 'error')
        print(traceback.format_exc())


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  DeepCompress - Full Stack Web App")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
