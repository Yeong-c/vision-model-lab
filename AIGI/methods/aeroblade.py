import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import lpips  
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class AerobladeEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        print(">> Initializing LPIPS metric...")
        self.loss_fn = lpips.LPIPS(net='alex').to(device).eval()

    def denormalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return x * std + mean

    @torch.no_grad()
    def get_score(self, x):
        """AEROBLADE 점수(복원 오차) 계산"""
        recon = self.model(x)
        # LPIPS 입력 범위 [-1, 1]로 조정
        in_image = self.denormalize(x).clamp(0, 1) * 2 - 1
        out_image = self.denormalize(recon).clamp(0, 1) * 2 - 1
        dists = self.loss_fn(in_image, out_image)
        return dists.view(-1)

    def get_dataloader(self, root_dir, batch_size=32):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def run_experiment(self, real_path, fake_root):
        print(f"\n[AEROBLADE Experiment Start]")
        print(f"Real Data: {real_path}")
        print(f"Fake Root: {fake_root}")

        # 1. Real Data (ImageNet) 평가
        real_loader = self.get_dataloader(real_path)
        real_scores = []
        
        print(f">> Calculating Real Scores...")
        count = 0
        for x, _ in tqdm(real_loader, desc="Real"):
            x = x.to(self.device)
            real_scores.extend(self.get_score(x).cpu().tolist())
            count += len(x)
            if count >= 500: break # Real 데이터는 500장만 사용
            
        real_scores = np.array(real_scores[:500])
        threshold = np.percentile(real_scores, 5) # TPR 95%
        print(f"✅ Calculated Threshold (Real TPR 95%): {threshold:.6f}")
        
        # 2. Fake Data 평가
        fake_loader = self.get_dataloader(fake_root)
        
        # 클래스 이름(폴더명) 확인 (예: ['gan_pool', 'mj_pool', ...])
        class_names = fake_loader.dataset.classes 
        results = {name: [] for name in class_names}

        print(f"\n>> Testing against GenImage models: {class_names}")
        print(">> Calculating Scores...")
        
        for x, y in tqdm(fake_loader, desc="Fake"):
            x = x.to(self.device)
            scores = self.get_score(x)
            
            # 배치 결과를 해당 모델 리스트에 분배
            for score, label_idx in zip(scores.cpu().tolist(), y.tolist()):
                class_name = class_names[label_idx]
                results[class_name].append(score)

        # 3. 최종 결과 출력
        print("\n" + "-" * 65)
        print(f"{'Model Name':<20} | {'Acc (%)':<10} | {'Avg Score':<10} | {'Samples'}")
        print("-" * 65)

        for name in class_names:
            scores = np.array(results[name])
            if len(scores) == 0: continue
            
            # Fake는 점수가 Threshold보다 낮아야 정답 
            acc = np.mean(scores < threshold) * 100
            avg_score = np.mean(scores)
            
            print(f"{name:<20} | {acc:<10.2f} | {avg_score:<10.4f} | {len(scores)}")

        print("-" * 65)
        real_acc = np.mean(real_scores >= threshold) * 100
        print(f"Real Accuracy (Target 95%): {real_acc:.2f}%")
        print("=" * 65 + "\n")