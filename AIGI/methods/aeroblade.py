import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import lpips  
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

class AerobladeEvaluator:
    def __init__(self, model, device):
        """
        experiment.py에서 생성된 RAE 모델을 받아서 초기화
        """
        self.model = model
        self.device = device
        print(">> Initializing LPIPS metric...")
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.loss_fn.eval()

    def denormalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return x * std + mean

    @torch.no_grad()
    def get_score(self, x):
        """
        배치 이미지 x를 받아 AEROBLADE 점수(LPIPS Reconstruction Error) 반환
        """
        # 1. 모델 통과 
        # RAE 모델의 출력이 (B, C, H, W) 형태라고 가정
        recon = self.model(x)

        # 2. Denormalize ([0,1] 범위로 복원)
        in_image = self.denormalize(x)
        out_image = self.denormalize(recon)

        # 3. LPIPS 계산을 위해 [-1, 1] 범위로 
        lpips_in = (in_image.clamp(0, 1) * 2) - 1
        lpips_out = (out_image.clamp(0, 1) * 2) - 1

        # 4. 점수 계산 (배치 내 각 이미지별 점수)
        dists = self.loss_fn(lpips_in, lpips_out)
        return dists.view(-1)

    def get_dataloader(self, root_dir, batch_size=32, img_size=256):
        """폴더 경로에서 데이터로더 생성"""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), # 256x256
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        
        # 500장 제한 (랜덤 샘플링 or 앞에서부터)
        if len(dataset) > 500:
            indices = list(range(500))
            dataset = Subset(dataset, indices)
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def run_experiment(self, real_path, fake_path):
        """
        실제 실험 실행 함수
        """
        print(f"\n[AEROBLADE Experiment Start]")
        print(f"Real Data: {real_path}")
        print(f"Fake Data: {fake_path}")

        # 1. 데이터 로드
        real_loader = self.get_dataloader(real_path)
        fake_loader = self.get_dataloader(fake_path)

        # 2. 점수 측정
        real_scores = []
        fake_scores = []

        print(">> Calculating Scores for Real Images...")
        for x, _ in tqdm(real_loader):
            x = x.to(self.device)
            scores = self.get_score(x)
            real_scores.extend(scores.cpu().tolist())

        print(">> Calculating Scores for Fake Images...")
        for x, _ in tqdm(fake_loader):
            x = x.to(self.device)
            scores = self.get_score(x)
            fake_scores.extend(scores.cpu().tolist())

        # 3. 결과 분석
        real_scores = np.array(real_scores)
        fake_scores = np.array(fake_scores)

        # Threshold: Real 데이터의 하위 5% 지점 (TPR 95% 기준)
        threshold = np.percentile(real_scores, 5)

        # 정확도 계산 (Real >= threshold, Fake < threshold)
        real_acc = np.mean(real_scores >= threshold) * 100
        fake_acc = np.mean(fake_scores < threshold) * 100
        total_acc = (real_acc + fake_acc) / 2

        print("\n" + "="*50)
        print(f" RESULT REPORT")
        print("="*50)
        print(f" Threshold (Real TPR 95%): {threshold:.6f}")
        print("-" * 50)
        print(f" Real Accuracy : {real_acc:.2f}%")
        print(f" Fake Accuracy : {fake_acc:.2f}%")
        print(f" Total Accuracy: {total_acc:.2f}%")
        print("="*50 + "\n")