import os, copy, io, torch
from config import MODELS_DIR
from models import get_model
from utils import load_model, ensure_dirs

def quantize_model(model):
    q = copy.deepcopy(model); q.eval()
    return torch.quantization.quantize_dynamic(q, {torch.nn.Linear}, dtype=torch.qint8)

def get_quantized_size_mb(q):
    buf = io.BytesIO(); torch.save(q.state_dict(), buf)
    return round(buf.tell()/(1024**2), 4)

def quantize_and_save(source_model=None, source_path=None, save_name=None, device='cpu', dataset='cifar10'):
    ensure_dirs(MODELS_DIR)
    if save_name is None: save_name = f'quantized_{dataset}.pt'
    if source_model is None:
        if source_path is None: source_path = os.path.join(MODELS_DIR, f'baseline_best_{dataset}.pt')
        source_model = get_model()
        source_model = load_model(source_model, source_path, device=device)
    q = quantize_model(source_model)
    torch.save(q.state_dict(), os.path.join(MODELS_DIR, save_name))
    print(f"[quant] Saved -> {save_name}  ({get_quantized_size_mb(q)} MB)")
    return q
