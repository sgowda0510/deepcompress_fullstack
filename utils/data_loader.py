# utils/data_loader.py — Supports CIFAR-10, MNIST, Diabetes, CIFAR-100
import os, csv, urllib.request
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS


def get_cifar10_loaders(batch_size=BATCH_SIZE):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_tf)
    print(f"[CIFAR-10] Train:{len(train_ds)}  Test:{len(test_ds)}")
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS), \
           DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


def get_mnist_loaders(batch_size=BATCH_SIZE):
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3)])
    train_ds = datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tf)
    print(f"[MNIST] Train:{len(train_ds)}  Test:{len(test_ds)}")
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS), \
           DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


def get_diabetes_loaders(batch_size=64):
    csv_path = os.path.join(DATA_DIR, "diabetes.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(csv_path):
        try:
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
            print("[Diabetes] Downloading...")
            urllib.request.urlretrieve(url, csv_path)
        except Exception:
            _create_sample_diabetes(csv_path)
    X, y = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) == 9:
                try:
                    X.append([float(v) for v in row[:8]])
                    y.append(int(float(row[8])))
                except ValueError:
                    continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    n_tr = int(0.8 * len(ds))
    tr, te = random_split(ds, [n_tr, len(ds)-n_tr], generator=torch.Generator().manual_seed(42))
    print(f"[Diabetes] Train:{n_tr}  Test:{len(ds)-n_tr}  Features:{X.shape[1]}")
    return DataLoader(tr, batch_size=batch_size, shuffle=True), \
           DataLoader(te, batch_size=batch_size, shuffle=False)


def _create_sample_diabetes(path):
    import random; random.seed(42)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        for _ in range(600):
            w.writerow([random.randint(0,15), random.randint(70,200),
                        random.randint(40,120), random.randint(0,60),
                        random.randint(0,500), round(random.uniform(18,60),1),
                        round(random.uniform(0.08,2.5),3), random.randint(18,80),
                        random.randint(0,1)])


def get_cifar100_loaders(batch_size=BATCH_SIZE):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR100(root=DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_tf)
    print(f"[CIFAR-100] Train:{len(train_ds)}  Test:{len(test_ds)}")
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS), \
           DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


def get_custom_loaders(dataset_id, batch_size=64):
    csv_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file {csv_path} not found")
    import pandas as pd
    df = pd.read_csv(csv_path)
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    if y_raw.dtype == object or str(y_raw.dtype) == 'category':
        classes, y_raw = np.unique(y_raw, return_inverse=True)
    y_raw = y_raw.astype(np.int64)
    X_scaled = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
    ds = TensorDataset(torch.FloatTensor(X_scaled), torch.LongTensor(y_raw))
    n_tr = int(0.8 * len(ds))
    tr, te = random_split(ds, [n_tr, len(ds) - n_tr], generator=torch.Generator().manual_seed(42))
    return DataLoader(tr, batch_size=batch_size, shuffle=True), DataLoader(te, batch_size=batch_size, shuffle=False)


def get_loaders(dataset='cifar10', batch_size=BATCH_SIZE):
    d = dataset.lower().strip()
    if   d == 'cifar10':   return get_cifar10_loaders(batch_size)
    elif d == 'mnist':     return get_mnist_loaders(batch_size)
    elif d == 'diabetes':  return get_diabetes_loaders()
    elif d == 'cifar100':  return get_cifar100_loaders(batch_size)
    elif os.path.exists(os.path.join(DATA_DIR, f"{d}.csv")):
        return get_custom_loaders(d, batch_size)
    else:
        print(f"Unknown dataset '{dataset}', using CIFAR-10")
        return get_cifar10_loaders(batch_size)
