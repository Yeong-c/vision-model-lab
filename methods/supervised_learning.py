import torch
import torch.nn as nn
class SupervisedLearning(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone

        # Head(Classifier)
        self.head = nn.Linear(backbone.num_features, num_classes)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        x, y = batch
        # Features 추출
        features = self.backbone(x)
        # Outputs
        output = self.head(features)
        # Loss 계산
        loss = self.criterion(output, y)

        return loss
    
    def predict(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output