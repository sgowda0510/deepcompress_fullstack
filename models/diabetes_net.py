# models/diabetes_net.py — FC Network for tabular/medical data
import torch.nn as nn
import torch.nn.functional as F


class DiabetesNet(nn.Module):
    def __init__(self, input_features=8, num_classes=2):
        super().__init__()
        self.fc1     = nn.Linear(input_features, 64)
        self.fc2     = nn.Linear(64, 32)
        self.fc3     = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_diabetes_model(input_features=8, num_classes=2):
    return DiabetesNet(input_features, num_classes)
