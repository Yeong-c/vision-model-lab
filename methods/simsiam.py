import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiam(nn.Module):
    def __init__(self, encoder, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()

        self.encoder = encoder
        prev_dim = encoder.num_features

        #projector(3MLP)
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False) 
        )

        #Predictor(2MLP)
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, batch):
        x, _ = batch
        x1, x2 = x[:, 0], x[:, 1]
        #feature
        f1, f2 = self.encoder(x1), self.encoder(x2)
        #projector only
        z1, z2 = self.projector(f1), self.projector(f2)
        #predictor +
        p1, p2 = self.predictor(z1), self.predictor(z2)
        
        loss_1 = -F.cosine_similarity(p1, z2.detach(), dim=1).mean()
        loss_2 = -F.cosine_similarity(p2, z1.detach(), dim=1).mean()

        return 0.5 * loss_1 + 0.5 * loss_2

    def predict(self, x):
        features = self.encoder(x)
        output = self.projector(features)
        return output
