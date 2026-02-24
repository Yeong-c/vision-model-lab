import torch
import torch.nn.functional as F

class VecAugDetector:
    def __init__(self, model, device="cuda", noise_std=0.01):
        """
        [Rational Design]
        인코더 입력(224)과 디코더 출력(256)의 불일치를 해결하기 위해 
        입력을 256으로 업샘플링하여 비교합니다.
        """
        self.model = model
        self.device = device
        self.noise_std = noise_std
        self.model.eval()
        
        # 디코더가 실제로 뱉어내는 이미지 크기 계산
        # base_patches가 256이라면 16 * 16 = 256이 됩니다.
        from math import sqrt
        self.recon_size = int(self.model.encoder_patch_size * sqrt(self.model.base_patches))
        
        print(f"[VecAugDetector] Sync Version (Target: {self.recon_size}). Noise: {self.noise_std}")

    @torch.no_grad()
    def compute_scores(self, x):
        B = x.shape[0]
        x = x.to(self.device)

        # 1. 인코딩 및 잠재 벡터 추출
        # RAE.encode는 내부적으로 224 리사이즈와 정규화를 수행합니다.
        z = self.model.encode(x) 

        # 2. 잠재 공간 섭동 및 복원
        # RAE.decode는 내부적으로 256 patches를 unpatchify하여 256x256 이미지를 생성합니다.
        x_rec = self.model.decode(z, latent_noise_scale=self.noise_std)

        # 3. [핵심] 입력 이미지를 디코더 출력 크기(256)로 리사이즈
        # RuntimeError: (224) must match (256) 문제를 원천 차단합니다.
        x_target = F.interpolate(x, size=(self.recon_size, self.recon_size), mode='bicubic')

        # 4. 지표 계산: MSE
        # 이제 두 텐서 모두 [B, 3, 256, 256] 형태이므로 안전하게 연산됩니다.
        diff = (x_target - x_rec) ** 2
        score = diff.mean(dim=[1, 2, 3])

        return {
            "score_max": score.cpu().numpy()
        }