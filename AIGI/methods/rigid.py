import torch
import torch.nn.functional as F
from torchvision import transforms
from RAE.encoder import Dinov2withNorm
from transformers import AutoImageProcessor

class Rigid:
    def __init__(self, device, model_id, register=True):
        self.device = device
        self.register = register

        if self.register:
            print(f">> Initializing RIGID using `RAE.encoder.Dinov2withNorm` on {device}...")
            # DINOv2 모델 로드
            MODEL_ID = model_id
            self.model = Dinov2withNorm(dinov2_path=MODEL_ID, normalize=False)
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_id)

        self.model.to(self.device)
        self.model.eval()

        self.encoder_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.encoder_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        
    @torch.no_grad()
    def get_score(self, x, noise_level=0.05):
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        
        # Normalize - ImageNet
        x = (x - self.encoder_mean) / self.encoder_std

        if self.register: # RAE와 같은 Encoder with Encoder
            f_orig = self.model.full_forward(x)
            # 저자 코드는 CLS 토큰 혹은 avg pooling 된 벡터 하나만 사용. (dino가 벡터 하나만 뱉음)
            f_orig = f_orig[:, 0, :]
        
            img_noisy = x + torch.randn_like(x) * noise_level
            f_noisy = self.model.full_forward(img_noisy)
            f_noisy = f_noisy[:, 0, :]
        
            sim = F.cosine_similarity(f_orig, f_noisy, dim=-1)

        else: # RIGID 논문 식 구현
            f_orig = self.model(x)
            
            img_noisy = x + torch.randn_like(x) * noise_level
            f_noisy = self.model(img_noisy)

            sim = F.cosine_similarity(f_orig, f_noisy, dim=-1)

        
        # 변화량이 클수록(유사도가 낮을수록) 큰 값 반환
        return 1.0 - sim
