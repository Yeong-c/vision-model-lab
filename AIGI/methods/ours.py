import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class MultiLayerRAE:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def get_layer_features(self, x):
        _, all_layers = self.model.encoder.dinov2_forward(x, return_all_layers=True)
        return all_layers


    """
    메인 구조
    - 원본 이미지와 rae를 통해 복원된 이미지를 각각 준비
    각 이미지를 dino에 통과시켜 13개의 feature map을 모두 추출
    이 때 각 이미지는 가중치가 고정된 dino를 통과하므로, 각 레이어는 동일한 수준의 정보를 추출함.
    따라서 해당 레이어상에서의 거리는 해당 레벨의 왜곡 정도를 볼 수 있는 지표

    실험결과 biggan 등의 모델에선 초반 레이어 쪽에서, sdv4등의 모델에선 후반 레이어에서 고득점 = 차이가 컸음
    """
    @torch.no_grad()
    def get_layer_score(self, x):
        """
         Cosine Distance 사용
        """
        # 1. RAE로 복원
        recon = self.model(x)

        # 2. 정규화
        x_norm = self.normalize(x)
        recon_norm = self.normalize(recon)

        # 3. 특징 추출
        real_feats = self.get_layer_features(x_norm)    
        fake_feats = self.get_layer_features(recon_norm)
        
        layer_scores = []
        
        for f_real, f_fake in zip(real_feats, fake_feats):
            # f_real, f_fake shape: (Batch, Tokens, Channels)
            
            # Cosine Similarity 사용
            # Cosine Distance = 1 - Cosine Similarity
            # dim=-1 (채널 방향)으로 유사도 계산 -> (Batch, Tokens)
            sim = F.cosine_similarity(f_real, f_fake, dim=-1)
            
            # 토큰들의 평균을 내서 배치별 점수 산출 -> (Batch,)
            # 1에서 빼주므로, "차이가 클수록(거리가 멀수록)" 점수가 커짐 (Rigid와 동일 논리)
            dist = 1.0 - torch.mean(sim, dim=1)
            
            layer_scores.append(dist)
            
        return layer_scores

    @torch.no_grad()
    def get_layer_score_map(self, x):

        recon = self.model(x)
        x_norm = self.normalize(x)
        recon_norm = self.normalize(recon)
        
        real_feats = self.get_layer_features(x_norm)
        fake_feats = self.get_layer_features(recon_norm)

        score_maps = []
        for f_real, f_fake in zip(real_feats, fake_feats):
            # (Batch, N_patches, Channels)
            # 채널 방향 Cosine Distance
            sim = F.cosine_similarity(f_real, f_fake, dim=-1) # (Batch, N_patches)
            dist = 1.0 - sim 
            
            h = w = int(dist.shape[1] ** 0.5)
            dist_map = dist.view(x.shape[0], 1, h, w)
            score_maps.append(dist_map)
            
        return score_maps
