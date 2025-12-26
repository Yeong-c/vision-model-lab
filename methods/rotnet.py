import torch
import torch.nn as nn


def rotate_batch(images, device):

    batch_size = images.size(0)
    #dims = 배치, 채널, 높 , 넙
    x0 = images
    y0 = torch.zeros(batch_size, dtype=torch.long, device=device)

    x90 = torch.rot90(images, k=1, dims=[2,3])
    y90 = torch.ones(batch_size, dtype=torch.long, device=device)

    x180 = torch.rot90(images, k=2, dims=[2,3])
    y180 = torch.full((batch_size,), 2, dtype=torch.long, device=device)

    x270 = torch.rot90(images, k=3, dims=[2,3])
    y270 = torch.full((batch_size,), 3, dtype=torch.long, device=device)

    imgs = torch.cat([x0,x90,x180,x270], dim=0)
    labs = torch.cat([y0,y90,y180,y270], dim=0)

    return imgs, labs


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
        inputs, labels = rotate_batch(x, x.device)

        features = self.model(inputs)
        logits = self.head(features)

        loss = self.criterion(logits, labels)

        return loss
    
    def predict(self, x):
        features = self.model(x)
        logits = self.head(features)
        return logits
