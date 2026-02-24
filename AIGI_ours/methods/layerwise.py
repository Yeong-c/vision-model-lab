import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerWiseErrorDetector(nn.Module):
    def __init__(self, model_id='dinov2_vitb14', device='cuda', use_abs=True):
        super(LayerWiseErrorDetector, self).__init__()
        self.device = device
        # 차이의 절대값을 사용할지 여부
        self.use_abs = use_abs

        # Frozen DINOv2 Backbone 로드
        print(f">> Loading Frozen Backbone: {model_id}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_id)
        self.backbone.to(device)
        self.backbone.eval()
        
        # 백본 파라미터 고정
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.embed_dim # ViT-B: 768
        self.num_layers = len(self.backbone.blocks) # 12 layers
        
        # 입력 차원: 12개 레이어 * 768차원 = 9216
        # 만약 차이(delta)와 절대값(abs_delta)을 모두 쓴다면 2배
        input_dim = self.embed_dim * self.num_layers
        if self.use_abs:
            input_dim *= 2 

        # Trainable Linear Head (이것만 학습됨)
        """
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # 최종 Score (Real 0 ~ Fake 1)
        ).to(device)
        """
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 1)
        ).to(device)

        self.norm = nn.LayerNorm(input_dim).to(device)
        
        # ImageNet 정규화 파라미터
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def get_all_layers_cls(self, x):
        # return_class_token=True를 쓰면 각 레이어 결과가 (patches, cls_token) 튜플로 나옴
        intermediate_outputs = self.backbone.get_intermediate_layers(
            x, n=range(self.num_layers), return_class_token=True
        )
        
        # 각 레이어의 CLS 토큰
        cls_tokens = [out[1] for out in intermediate_outputs]

        # 결과: (B, 9216)
        return torch.cat(cls_tokens, dim=-1)

    def forward(self, x_orig, x_aug):
        # 정규화
        x_orig = (x_orig - self.mean) / self.std
        x_aug = (x_aug - self.mean) / self.std

        # 모든 레이어 특징 추출
        with torch.no_grad():
            z_orig = self.get_all_layers_cls(x_orig) # (B, 9216)
            z_aug = self.get_all_layers_cls(x_aug) # (B, 9216)

        # Layer-wise Error 계산
        delta = z_orig - z_aug
        
        if self.use_abs:
            # delta과 abs(delta)를 모두 학습에 활용(concat)
            error_pattern = torch.cat([delta, torch.abs(delta)], dim=-1)
        else:
            error_pattern = delta

        # Linear Head 분류기
        error_pattern = self.norm(error_pattern)
        logit = self.classifier(error_pattern)
        
        # return torch.sigmoid(logit) # 0(Real) ~ 1(Fake)
        return logit

    def get_score(self, x_orig, x_aug):
        # 스코어 반환(Sigmoid)
        with torch.no_grad():
            logit = self.forward(x_orig, x_aug)
            score = torch.sigmoid(logit)
        return score