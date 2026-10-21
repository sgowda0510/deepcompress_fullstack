# models/cnn.py — SimpleCNN for image datasets
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1     = nn.Linear(64 * 8 * 8, 256)
        self.fc2     = nn.Linear(256, 128)
        self.fc3     = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_model(num_classes=10):
    return SimpleCNN(num_classes=num_classes)
