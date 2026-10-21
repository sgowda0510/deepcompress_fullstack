import os, io, torch

def get_model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.tell() / (1024**2), 4)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[save] -> {path}")

def load_model(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device); model.eval()
    return model

def ensure_dirs(*dirs):
    for d in dirs: os.makedirs(d, exist_ok=True)
