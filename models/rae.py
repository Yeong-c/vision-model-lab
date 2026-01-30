import torch
import torch.nn as nn
from .vit import ViT

class RAE_ViT(ViT): #디코더로 쓰일 ViT Base (DINO 인코더 이후)
    def __init__(self, patch_size=16, img_size=224, dropout=0.1, noise=0.8):
        # outputs: 768, num_layers: 12, n_heads: 12
        super().__init__(768, 12, 12, patch_size, img_size, dropout)
        # Layer Norm(DINO 출력에)
        self.layer_norm = nn.LayerNorm(768)
        # DINO 출력을 projection 768 -> 1152
        self.input_proj = nn.Identity()
        # 가우시안 노이즈 weight
        self.noise = noise

    def forward(self, x):
        # Layer Norm 진행
        x = self.layer_norm(x)
        # 필요시 Projection, 지금은 identity
        x = self.input_proj(x)
        # 가우시안 노이즈(학습시에만)
        if self.training:
            x = x + torch.randn_like(x) * self.noise
        # cls 확장 / 앞에 추가
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 포지션 추가
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 트랜스포머
        x = self.transformer(x)
        # 패치 토큰들 추출
        out = x[:, 1:]
        out = self.norm(out)
        return out

class RAE(nn.Module):
    def __init__(self, patch_size=16, img_size=224, dropout=0.1):
        super().__init__()
        # Encoder: DINO
        self.encoder = self.get_encoder("dinov2_vitb14")
        # Encoder 얼리기
        for p in self.encoder.parameters():
            p.requires_grad = False
        # Decoder: ViT
        self.decoder = RAE_ViT(patch_size=patch_size, img_size=img_size, dropout=dropout)

        # Decoder Pixel Head(768차원을 3*patch_size^2로)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 3*patch_size**2)
            )

    def to_image(self, x):
        x = torch.reshape(x, (-1, 16, 16, 16, 16, 3))
        x = torch.permute(x, (0, 5, 1, 3, 2, 4))
        x = x.contiguous()
        x = torch.reshape(x, (-1, 3, 256, 256))
        return x

    def get_encoder(self, dino_name):
        return torch.hub.load('facebookresearch/dinov2', dino_name)
    
    def forward(self, x):
        # 이미지 크기를 224x224로
        x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic')
        # Encoder 통과
        x = self.encoder(x)
        # Decoder 통과
        x = self.decoder(x[:, 1:, :])
        # Pixel Head 통과
        x = self.final_layer(x)
        # To Image
        x = self.to_image(x)

        return x
    
