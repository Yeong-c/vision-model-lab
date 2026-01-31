import torch
import torch.nn as nn
import numpy as np
from .vit import ViT

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False, device='cuda'):
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
    grid = torch.stack(grid, dim=0)  # [2, grid_size, grid_size]

    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # embed_dim의 절반은 h, 절반은 w에 할당
    omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device)
    omega /= (embed_dim // 4)
    omega = 1.0 / (10000**omega)  # (D/4,)

    pos_h = grid[0].reshape(-1)
    pos_w = grid[1].reshape(-1)
    
    out_h = torch.einsum("m,d->md", pos_h, omega)
    out_w = torch.einsum("m,d->md", pos_w, omega)

    # sin, cos 결합
    pos_embed = torch.cat([
        torch.sin(out_h), torch.cos(out_h),
        torch.sin(out_w), torch.cos(out_w)
    ], dim=1)

    if add_cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim, device=device), pos_embed], dim=0)
    return pos_embed

class RAE_ViT(ViT): #디코더로 쓰일 ViT Base (DINO 인코더 이후)
    def __init__(self, patch_size=16, img_size=256, dropout=0.1, noise=0.8):
        self.config = {
            'outputs': 768,
            'num_layers': 12,
            'n_heads': 12
        }
        super().__init__(self.config['outputs'], self.config['num_layers'], self.config['n_heads'], patch_size, img_size, dropout)

        # pos_embed 덮어쓰기
        num_patches = (img_size // patch_size) ** 2
        pos_embed = get_2d_sincos_pos_embed(self.config['outputs'], int(num_patches**0.5), add_cls_token=True)
        self.pos_embed = nn.Parameter(pos_embed.float().unsqueeze(0), requires_grad=False)

        # Projection 필요하면
        #self.input_proj = nn.Linear(768, 1024)
        self.input_proj = nn.Identity()

        # Layer Norm(DINO 출력에)
        self.layer_norm = nn.LayerNorm(768, elementwise_affine=False)

        # 학습 가능한 CLS 토큰
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, self.config['outputs']))

        # 가우시안 노이즈 tau
        self.noise = noise

    def forward(self, x):
        # Layer Norm 진행
        x = self.layer_norm(x)
        # 필요시 Projection, 지금은 identity
        x = self.input_proj(x)
        # 가우시안 노이즈(학습시에만)
        if self.training:
            tau = self.noise
            sigma = torch.randn(x.size(0), 1, 1, device=x.device).abs() * tau
            x = x + torch.randn_like(x) * sigma
        # cls 확장 / 앞에 추가
        batch_size = x.shape[0]
        cls_token = self.trainable_cls_token.expand(batch_size, -1, -1)
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
    def __init__(self, patch_size=16, img_size=256, dropout=0.1):
        super().__init__()
        # Encoder: DINO
        self.encoder = self.get_encoder("dinov2_vitb14_reg")
        # Encoder 얼리기
        for p in self.encoder.parameters():
            p.requires_grad = False
        # Decoder: ViT
        self.decoder = RAE_ViT(patch_size=patch_size, img_size=img_size, dropout=dropout)

        # Decoder Pixel Head(768차원을 3*patch_size^2로)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(self.decoder.config['outputs']),
            nn.Linear(self.decoder.config['outputs'], 3*patch_size**2)
            )

    """def to_image(self, x):
        B = x.shape[0]
        p = 16  # patch_size
        grid_size = 16  # 256 tokens = 16x16 grid
        x = x.reshape(B, grid_size, grid_size, 768)
        x = x.reshape(B, grid_size, grid_size, 3, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.contiguous().reshape(B, 3, 256, 256)
        return x
    """

    # 저자 코드
    def unpatchify(self, x):
            p = 16 # patch_size
            h = w = 16 # 256 / 16
            x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
            x = torch.einsum('nhwpqc->nchpwq', x)
            return x.reshape(shape=(x.shape[0], 3, h * p, w * p))

    def get_encoder(self, dino_name):
        return torch.hub.load('facebookresearch/dinov2', dino_name)
    
    def forward(self, x):
        # 이미지 크기를 224x224로
        x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic')
        # Encoder 통과
        patch_tokens, cls_token = self.encoder.get_intermediate_layers(x, n=1, return_class_token=True)[0]
        # Decoder 통과
        x = self.decoder(patch_tokens)
        # Pixel Head 통과
        x = self.final_layer(x)
        # To Image
        x = self.unpatchify(x)

        return x
    
