# train.py — Training pipeline (supports all datasets)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODELS_DIR, DATA_DIR
from models import get_model
from models.diabetes_net import get_diabetes_model
from utils import get_loaders, save_model, ensure_dirs


def train(num_epochs=NUM_EPOCHS, device=DEVICE, dataset='cifar10', progress_cb=None):
    """
    progress_cb: optional callback(step, percent, msg) for live web updates
    """
    ensure_dirs(MODELS_DIR)
    device = torch.device(device)
    train_loader, test_loader = get_loaders(dataset)

    if dataset == 'diabetes':
        model = get_diabetes_model().to(device)
    elif os.path.exists(os.path.join(DATA_DIR, f"{dataset}.csv")):
        import pandas as pd
        df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset}.csv"))
        feats = df.shape[1] - 1
        cls_count = max(len(df.iloc[:, -1].unique()), 2)
        model = get_diabetes_model(input_features=feats, num_classes=cls_count).to(device)
    elif dataset == 'cifar100':
        model = get_model(num_classes=100).to(device)
    else:
        model = get_model(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += out.max(1)[1].eq(y).sum().item()
            total   += y.size(0)
        tr_acc = 100.0 * correct / total

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                vc += model(x).max(1)[1].eq(y).sum().item()
                vt += y.size(0)
        val_acc = 100.0 * vc / vt
        scheduler.step()

        msg = f"Epoch [{epoch}/{num_epochs}]  Train: {tr_acc:.2f}%  Val: {val_acc:.2f}%"
        print(msg)
        if progress_cb:
            pct = int(10 + (epoch / num_epochs) * 30)
            progress_cb('training', pct, msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, os.path.join(MODELS_DIR, f'baseline_best_{dataset}.pt'))

    save_model(model, os.path.join(MODELS_DIR, f'baseline_final_{dataset}.pt'))
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    return model


if __name__ == '__main__':
    train()
