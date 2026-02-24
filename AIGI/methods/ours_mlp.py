import torch
import torch.nn.functional as F

class MultiLayerRAE_MLP: 
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def get_layer_score(self, x):
        """
        [Output] List of scalar scores (Layer 0 ~ Layer 12)
        """
        # 1. Reconstruction
        x_rec = self.model(x)
        
        # 2. Extract Features (All Layers)
        feats_real = self.model.get_dino_features(x)
        feats_fake = self.model.get_dino_features(x_rec)
        
        layer_scores = []
        
        # 3. Layer-wise Cosine Distance
        for f_real, f_fake in zip(feats_real, feats_fake):
            sim = F.cosine_similarity(f_real, f_fake, dim=-1)
            dist = 1.0 - sim
            score = dist.mean(dim=1) # (Batch,)
            layer_scores.append(score)
            
        return layer_scores

    @torch.no_grad()
    def get_layer_score_map(self, x):
        """
        [Visualization] Map 형태 반환
        """
        x_rec = self.model(x)
        feats_real = self.model.get_dino_features(x)
        feats_fake = self.model.get_dino_features(x_rec)
        
        layer_maps = []
        
        for f_real, f_fake in zip(feats_real, feats_fake):
            sim = F.cosine_similarity(f_real, f_fake, dim=-1)
            dist = 1.0 - sim
            
            # CLS 토큰 제거 (257개일 경우)
            if dist.shape[1] == 257:
                dist = dist[:, 1:]
                
            b, n = dist.shape
            h = w = int(n**0.5)
            
            dist_map = dist.view(b, 1, h, w)
            layer_maps.append(dist_map)
            
        return layer_maps