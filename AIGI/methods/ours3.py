import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

class CheckerboardBlurDetector:
    def __init__(self, model, patch_size=28, sigma=1.2):
        self.model = model # RAE 클래스
        self.patch_size = patch_size
        self.sigma = sigma

    def get_checkerboard_mask(self, batch_size, h, w, device):
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        row = torch.arange(grid_h, device=device).view(-1, 1) % 2
        col = torch.arange(grid_w, device=device).view(1, -1) % 2
        checkerboard = (row ^ col).to(torch.float32)
        mask = F.interpolate(checkerboard.view(1, 1, grid_h, grid_w), size=(h, w), mode='nearest')
        return mask.expand(batch_size, -1, -1, -1)

    @torch.no_grad()
    def get_score(self, x):
        """
        [B, 3, 256, 256] 입력을 받아 [B] 리스트 점수 반환
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. 원본 특징 추출 (full_forward 사용)
        # output 형태: [B, 329, 768] (1 CLS + 4 Reg + 324 Patches)
        orig_all = self.model.encoder.full_forward(x) 
        
        # [Fix] 앞의 5개 토큰(CLS=1, Reg=4) 제거 후 순수 패치(324개)만 추출
        orig_patches = orig_all[:, 5:, :] 

        # 2. 퐁당퐁당 블러 적용 및 RAE 복원
        mask = self.get_checkerboard_mask(B, H, W, device)
        blurred_x = TF.gaussian_blur(x, [11, 11], [self.sigma, self.sigma])
        x_corrupted = x * (1 - mask) + blurred_x * mask
        
        x_recon = self.model(x_corrupted) #
        
        # 3. 복원된 이미지 특징 추출 및 패치 분리
        recon_all = self.model.encoder.full_forward(x_recon)
        recon_patches = recon_all[:, 5:, :]

        # 4. 코사인 유사도 기반 거리 측정 (RIGID 스타일)
        # sim 형태: [B, 324]
        sim = F.cosine_similarity(orig_patches, recon_patches, dim=-1) 
        
        # 5. [차원 Fix] 324개 패치 평균을 내어 이미지당 하나의 점수로 응축
        scores = 1.0 - sim.mean(dim=1) # [B]
        
        return scores.cpu().tolist()