import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DecoderAnalysisMethod:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # ImageNet Normalization (DINOv2 입력용)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def normalize(self, x):
        # 입력 x (0~1) -> ImageNet Norm
        return (x - self.mean) / self.std

    @torch.no_grad()
    def get_dino_features(self, x):
        # RAE Encoder(DINOv2)의 모든 레이어 Feature 추출
        # (Batch, Tokens, Dim) 리스트 반환
        return self.model.encoder.dinov2_forward(self.normalize(x), return_all_layers=True)[1]

    @torch.no_grad()
    def get_layer_score(self, x):
        """
        [Winning Logic] Complexity Ratio
        Score = Spatial Error (L2) / (Style Error (Mean/Std) + eps)
        """
        # 1. Reconstruction
        x_rec = self.model(x)
        
        # 2. Extract Features
        feats_real = self.get_dino_features(x)
        feats_fake = self.get_dino_features(x_rec)
        
        layer_scores = []
        
        for f_real, f_fake in zip(feats_real, feats_fake):
            # A. Spatial Error (구조적 차이) -> 분자
            # (Batch, Tokens)
            spatial_dist = torch.norm(f_real - f_fake, p=2, dim=-1)
            
            # B. Style Error (통계적 차이) -> 분모
            mu_real = f_real.mean(dim=1, keepdim=True)
            std_real = f_real.std(dim=1, keepdim=True)
            mu_fake = f_fake.mean(dim=1, keepdim=True)
            std_fake = f_fake.std(dim=1, keepdim=True)
            
            # (Batch, 1)
            style_diff = torch.abs(mu_real - mu_fake).mean(dim=-1) + \
                         torch.abs(std_real - std_fake).mean(dim=-1)
            
            # C. Ratio Calculation (핵심)
            # 분모가 0이 되는 것을 막기 위해 eps 추가
            ratio_map = spatial_dist / (style_diff + 1e-6)
            
            # NaN/Inf 제거 (안전장치)
            ratio_map = torch.nan_to_num(ratio_map, nan=0.0, posinf=0.0, neginf=0.0)
            
            # D. Top-K Mean (상위 30%만 반영)
            # 이미지 전체가 아니라, 가짜의 징후가 가장 뚜렷한 부분만 봅니다.
            k = max(1, int(ratio_map.size(1) * 0.30))
            topk_vals, _ = torch.topk(ratio_map, k, dim=1)
            score = torch.mean(topk_vals, dim=1)
            
            layer_scores.append(score)
            
        return layer_scores

    @torch.no_grad()
    def get_visualization_data(self, x):
        """
        [Visualization] Ratio Map 추출
        L5 레이어(중간층)의 Ratio Map을 시각화용으로 반환
        """
        x_rec = self.model(x)
        feats_real = self.get_dino_features(x)
        feats_fake = self.get_dino_features(x_rec)
        
        # L5 레이어 사용 (가장 밸런스 좋음)
        target_idx = 5 if len(feats_real) > 5 else len(feats_real) - 1
        
        f_real = feats_real[target_idx]
        f_fake = feats_fake[target_idx]
        
        # Ratio Map 계산
        spatial_dist = torch.norm(f_real - f_fake, p=2, dim=-1)
        
        mu_real = f_real.mean(dim=1, keepdim=True)
        std_real = f_real.std(dim=1, keepdim=True)
        mu_fake = f_fake.mean(dim=1, keepdim=True)
        std_fake = f_fake.std(dim=1, keepdim=True)
        
        style_diff = torch.abs(mu_real - mu_fake).mean(dim=-1) + \
                     torch.abs(std_real - std_fake).mean(dim=-1)
                     
        ratio_map = spatial_dist / (style_diff + 1e-6)
        ratio_map = torch.nan_to_num(ratio_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # CLS 토큰 제거
        if ratio_map.shape[1] == 257:
            ratio_map = ratio_map[:, 1:]
            
        # Reshape to 2D
        b, n = ratio_map.shape
        h = w = int(n**0.5)
        ratio_map = ratio_map.view(b, 1, h, w)
        
        return x, x_rec, ratio_map