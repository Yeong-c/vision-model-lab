import torch
import torch.nn as nn
from .decoder import GeneralDecoder
from .encoder import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class RAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z
    
    def decode(self, z: torch.Tensor, return_internals: bool = False, latent_noise_scale: float = 0.0):
        
        # 1. Latent Perturbation (핵심: 잠재 공간 흔들기)
        # z는 encode를 거쳐서 이미 (Batch, Channel, H, W) 형태입니다.
        if latent_noise_scale > 0:
            noise = torch.randn_like(z) * latent_noise_scale
            z = z + noise

        # 2. Denormalization (정규화 해제)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
            
        # 3. Reshape back to Sequence (4D -> 3D)
        if self.reshape_to_2d:
            # 여기서 에러가 났던 이유는 z가 3D로 들어왔기 때문입니다.
            # 이제 self.encode를 쓰면 z가 4D로 보장되므로 안전합니다.
            b, c, h, w = z.shape 
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
            
        # 4. Transformer Decoder Forward (Attention Off, Hidden Only)
        outputs = self.decoder(
            z, 
            drop_cls_token=False, 
            output_attentions=False, 
            output_hidden_states=return_internals,
            return_dict=True
        )
        
        logits = outputs.logits
        x_rec = self.decoder.unpatchify(logits)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        
        if return_internals:
            return x_rec, outputs.hidden_states
            
        return x_rec

    # [수정] 인코딩 과정을 self.encode로 위임하여 Shape 불일치 해결
    @torch.no_grad()
    def get_latent_stress_internals(self, x: torch.Tensor, latent_noise: float = 0.0):
        """
        Input -> Encode (Standard) -> [Add Noise to Latent] -> Decode -> Return Hidden States
        """
        # 1. Encode
        # self.encode 메서드는 이미 Interpolation, Normalize, Reshaping(4D)을 다 처리해줍니다.
        # 따라서 여기서 반환된 z는 decode가 좋아하는 (B, C, H, W) 형태입니다.
        z = self.encode(x)
        
        # 2. Decode with Perturbation
        # 내부적으로 latent_noise_scale만큼 z를 흔들고 Hidden State를 뱉습니다.
        _, hidden_states = self.decode(z, return_internals=True, latent_noise_scale=latent_noise)
        
        return hidden_states
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec