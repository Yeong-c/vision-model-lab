import torch
import torch.nn as nn

class RotNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Head 4개 각도
        self.head = nn.Linear(model.num_features, 4)

    def forward(self, batch):
        x, _ = batch