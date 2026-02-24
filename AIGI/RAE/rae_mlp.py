import torch
import torch.nn as nn
# [핵심 변경] 원본 decoder 파일의 GeneralDecoder 사용
from .decoder import GeneralDecoder 
from .encoder_mlp import ARCHS_MLP
from transformers import AutoConfig, AutoImageProcessor
from math import sqrt
from typing import Optional

class RAE_MLP(nn.Module):
    def __init__(self, 
        encoder_cls: str = 'Dinov2withNorm_MLP', 
        encoder_config_path: str = 'facebook/dinov2-with-registers-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        decoder_config_path: str = './RAE/configs', # 로컬 경로 권장
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        # 1. Encoder Init (MLP 버전 사용)
        encoder_cls_type = ARCHS_MLP[encoder_cls] 
        self.encoder = encoder_cls_type(**encoder_params)
        
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2
        
        # 2. Decoder Init (원본 GeneralDecoder 사용)
        try:
            decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        except:
            print(f"Warning: Could not load config from {decoder_config_path}. Using vit-mae-base.")
            decoder_config = AutoConfig.from_pretrained('facebook/vit-mae-base')

        decoder_config.hidden_size = self.latent_dim 
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        
        # 원본 Decoder 호출
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys: {keys.missing_keys}")

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        self.do_normalization = False 

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        z = self.encoder(x)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        
        # 원본 Decoder는 drop_cls_token 인자를 받으므로 그대로 사용
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

    @torch.no_grad()
    def get_dino_features(self, x: torch.Tensor):
        x = self.preprocess(x)
        # encoder_mlp의 기능 사용
        _, all_layers = self.encoder.dinov2_forward(x, return_all_layers=True)
        return all_layers