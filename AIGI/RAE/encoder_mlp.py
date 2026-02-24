import torch
from torch import nn
from transformers import Dinov2WithRegistersModel
from typing import Callable, Dict, Optional, Type, Union

# 이름 충돌 방지를 위한 별도 레지스트리
ARCHS_MLP: Dict[str, Type] = {}

def register_encoder_mlp(cls: Optional[Type] = None, *, name: Optional[str] = None):
    def decorator(inner_cls: Type) -> Type:
        encoder_name = name or inner_cls.__name__
        ARCHS_MLP[encoder_name] = inner_cls
        return inner_cls
    if cls is None:
        return decorator
    return decorator(cls)

@register_encoder_mlp()
class Dinov2withNorm_MLP(nn.Module):
    def __init__(self, dinov2_path: str, normalize: bool = True):
        super().__init__()
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
            
        self.encoder.requires_grad_(False)
        
        # Normalization 제거 (Raw Feature 사용)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
            
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        
    def dinov2_forward(self, x: torch.Tensor, return_all_layers: bool = False) -> Union[torch.Tensor, tuple]:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 registers
        
        image_features = x.last_hidden_state[:, unused_token_num:]

        # [핵심] MLP 학습용: 모든 레이어 반환 기능 추가
        if return_all_layers:
            # Tuple -> List 변환 및 토큰 슬라이싱
            all_layers = [layer[:, unused_token_num:] for layer in x.hidden_states]
            return image_features, all_layers
            
        return image_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)