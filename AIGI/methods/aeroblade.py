import torch
import lpips
from diffusers import AutoencoderKL, VQModel

class Aeroblade():
    def __init__(self, model, device):
        self.rae = model
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval() # 논문: For VGG16, which we mainly use in this work...
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def denormalize(self, x):
        return (x * self.std + self.mean)
    
    def get_layer_score(self, x): # 배치
        with torch.no_grad():
            x_norm = x
            x_rec = self.rae(x)
            # RAE 결과물 찍어보니까 -0.5 - 1.5 사이의 값이 나옴 그래서 clamp
            x_rec = torch.clamp(x_rec, 0, 1)

            x_lpips = x_norm * 2.0 - 1.0
            x_rec_lpips = x_rec * 2.0 - 1.0

            x_lpips = self.lpips.scaling_layer(x_lpips)
            x_rec_lpips = self.lpips.scaling_layer(x_rec_lpips)

            outs_ori = self.lpips.net(x_lpips)
            outs_rec = self.lpips.net(x_rec_lpips)

            # 모든 LPIPS 레이어 사용 후 list로 저장
            lpips_layers_score = []
            for i in range(5):
                feat_ori = lpips.normalize_tensor(outs_ori[i])
                feat_rec = lpips.normalize_tensor(outs_rec[i])

                # 차이 계산 및 가중치 적용
                diff = (feat_ori - feat_rec) ** 2
                
                score_map = self.lpips.lins[i](diff)
                
                score = score_map.mean(dim=(1, 2, 3))

                lpips_layers_score.append(score)

        return lpips_layers_score
    
    def get_layer_score_map(self, x): # 이미지 한 장
        with torch.no_grad():
            x_norm = x
            x_rec = self.rae(x)
            # RAE 결과물 찍어보니까 -0.5 - 1.5 사이의 값이 나옴 그래서 clamp
            x_rec = torch.clamp(x_rec, 0, 1)

            x_lpips = x_norm * 2.0 - 1.0
            x_rec_lpips = x_rec * 2.0 - 1.0

            x_lpips = self.lpips.scaling_layer(x_lpips)
            x_rec_lpips = self.lpips.scaling_layer(x_rec_lpips)

            outs_ori = self.lpips.net(x_lpips)
            outs_rec = self.lpips.net(x_rec_lpips)

            # 모든 LPIPS 레이어 사용 후 list로 저장
            lpips_layers_score_map = []
            for i in range(5):
                feat_ori = lpips.normalize_tensor(outs_ori[i])
                feat_rec = lpips.normalize_tensor(outs_rec[i])

                # 차이 계산 및 가중치 적용
                diff = (feat_ori - feat_rec) ** 2
                
                score_map = self.lpips.lins[i](diff)

                lpips_layers_score_map.append(score_map)

        return lpips_layers_score_map


import torch
import lpips
import torch.nn.functional as F

class Aeroblade_VAE():
    def __init__(self, ae_dict, device):
        self.aes = ae_dict # 로드된 AE 딕셔너리
        self.device = device
        # 논문 권장: VGG16 기반 LPIPS [cite: 102, 332]
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()

    def get_reconstruction(self, ae, x):
        """각 AE 타입에 맞는 재구성 로직"""
        with torch.no_grad():
            # 입력 범위 조정: [0, 1] -> [-1, 1] (VAE 학습 환경 맞춤)
            x_input = x * 2.0 - 1.0
            
            if isinstance(ae, AutoencoderKL):
                # SD 계열 VAE 재구성 [cite: 94]
                latent = ae.encode(x_input).latent_dist.mode()
                rec = ae.decode(latent).sample
            else:
                # Kandinsky VQ-VAE 재구성 [cite: 94]
                latent = ae.encode(x_input).latents
                rec = ae.decode(latent).sample
                
            # 출력 범위 복원: [-1, 1] -> [0, 1]
            rec = (rec + 1.0) / 2.0
            return torch.clamp(rec, 0, 1)

    def get_layer_score(self, x):
        """논문 Eq (2)의 Min(Delta_AE) 구현 [cite: 133]"""
        all_ae_scores = [] # 각 AE별 레이어 점수를 저장

        for name, ae in self.aes.items():
            x_rec = self.get_reconstruction(ae, x)

            # LPIPS 입력용 [-1, 1] 정규화
            x_lpips = x * 2.0 - 1.0
            x_rec_lpips = x_rec * 2.0 - 1.0

            # LPIPS 내부 레이어 피쳐 추출 [cite: 102]
            # (기존 성규님 aeroblade.py 로직 활용)
            outs_ori = self.lpips.net(self.lpips.scaling_layer(x_lpips))
            outs_rec = self.lpips.net(self.lpips.scaling_layer(x_rec_lpips))

            current_ae_layer_scores = []
            for i in range(5):
                feat_ori = lpips.normalize_tensor(outs_ori[i])
                feat_rec = lpips.normalize_tensor(outs_rec[i])
                diff = (feat_ori - feat_rec) ** 2
                score_map = self.lpips.lins[i](diff)
                score = score_map.mean(dim=(1, 2, 3))
                current_ae_layer_scores.append(score)
            
            all_ae_scores.append(torch.stack(current_ae_layer_scores)) 
            # [Layer(5), Batch]

        # 모든 AE 결과 중 최소 에러 선택 (앙상블의 핵심) [cite: 133, 173]
        all_ae_scores_stack = torch.stack(all_ae_scores) # [AE_Num, Layer, Batch]
        min_scores, _ = torch.min(all_ae_scores_stack, dim=0) # AE_Num 축에서 최소값
        
        # 리스트 형태로 반환 (기존 코드 호환)
        return [min_scores[i] for i in range(5)]

    def get_layer_score_map(self, x):
        with torch.no_grad():
            # 시각화 지표로는 SD1.5 VAE를 사용
            ae = self.aes['sd1']
            x_rec = self.get_reconstruction(ae, x)
            
            # LPIPS 입력 범위 조정 [-1, 1]
            x_lpips = x * 2.0 - 1.0
            x_rec_lpips = x_rec * 2.0 - 1.0

            outs_ori = self.lpips.net(self.lpips.scaling_layer(x_lpips))
            outs_rec = self.lpips.net(self.lpips.scaling_layer(x_rec_lpips))

            lpips_layers_score_map = []
            # VGG16의 5개 레이어를 순회하며 에러 맵 추출
            for i in range(5):
                feat_ori = lpips.normalize_tensor(outs_ori[i])
                feat_rec = lpips.normalize_tensor(outs_rec[i])

                # 채널별 차이 계산 및 선형 가중치 적용
                diff = (feat_ori - feat_rec) ** 2
                score_map = self.lpips.lins[i](diff)
                
                lpips_layers_score_map.append(score_map)

        return lpips_layers_score_map