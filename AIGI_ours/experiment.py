from datetime import datetime, timedelta, timezone
import sys
import os, argparse
import glob, tqdm
import random

import torch, torchvision
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from PIL import Image

import dataset

def _main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = dataset.get_mixed_dataloader(args, is_train=False)

    # [Logging] 실험 결과 폴더 생성
    os.makedirs("./results", exist_ok=True)
    start_time = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M")
    result_name = f"{start_time}_{args.dataset}_{args.fake}_{args.real}"
    result_dir = f"./results/eval_{result_name}"
    os.makedirs(result_dir, exist_ok=True)

    # 1. 모델 초기화
    from methods.layerwise import LayerWiseErrorDetector
    detector = LayerWiseErrorDetector(model_id=args.model_id, device=device)

    # 2. 체크포인트 로드 (하드코딩 경로)
    # CHECKPOINT_PATH = "./detector_all_r800_f800_middinov2_vitl14_epoch20.pth"
    CHECKPOINT_PATH = args.checkpoint_path
    if os.path.exists(CHECKPOINT_PATH):
        print(f">> Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        detector.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        detector.norm.load_state_dict(checkpoint['norm_state_dict'])
        print(">> Checkpoint loaded successfully.")
    else:
        print(f"!! Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    detector.eval() # 평가 모드 설정

    # 3. 평가 루프
    all_scores = []
    all_labels = []
    all_paths = []
    
    print(f">> Evaluation started on {args.dataset} dataset...")
    pbar = tqdm.tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for imgs, labels, paths in pbar:
            imgs = imgs.to(device)
            
            # 테스트 시에도 학습과 동일한 노이즈 조건 부여 (RIGID 전략)
            # args.noise를 통해 노이즈 강도 조절 가능
            noise_level = args.noise
            noisy_imgs = imgs + torch.randn_like(imgs) * noise_level
            noisy_imgs = torch.clamp(noisy_imgs, 0, 1)

            # 모델 통과 (0~1 사이의 확률값 반환)
            scores = detector(imgs, noisy_imgs)
            
            all_scores.extend(scores.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_paths.extend(paths)

    # 4. 성능 지표 계산
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)

    print(f"\n" + "="*30)
    print(f"Results for {args.dataset}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Avg Precision: {ap:.4f}")
    print("="*30)

    # 5. 결과 저장 (CSV)
    df = pd.DataFrame({
        'label': all_labels,
        'score': all_scores,
        'path': all_paths
    })
    df.to_csv(f"{result_dir}/predictions.csv", index=False)
    
    # 6. 시각화 및 결과 저장 (결과 리포트용)
    print(f">> Saving visualization plots to {result_dir}...")
    
    # 그래프 스타일 설정
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))

    # 1. ROC Curve
    plt.subplot(1, 2, 1)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {args.dataset}')
    plt.legend(loc="lower right")

    # 2. Score Distribution
    plt.subplot(1, 2, 2)
    # Real(0)과 Fake(1) 점수 분포 분리
    real_scores = all_scores[all_labels == 0]
    fake_scores = all_scores[all_labels == 1]
    
    sns.kdeplot(real_scores, fill=True, color="blue", label="Real (ImageNet)", bw_adjust=0.5)
    sns.kdeplot(fake_scores, fill=True, color="red", label=f"Fake ({args.dataset})", bw_adjust=0.5)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Detection Score')
    plt.ylabel('Density')
    plt.title('Score Distribution by Class')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{result_dir}/evaluation_plots.png", dpi=300)
    plt.close()

    # 텍스트 파일 저장
    with open(f"{result_dir}/summary.txt", "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Real Samples: {args.real}\n")
        f.write(f"Fake Samples: {args.fake}\n")
        f.write(f"Noise Level: {args.noise}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Average Precision: {ap:.4f}\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="none")
    parser.add_argument("--dataset", type=str, default="all")
    '''
    dataset: all -> 모든 GenImage 샘플링(모델별)
    dataset: adm, biggan... -> 특정 GenImage 샘플링
    '''
    parser.add_argument("--fake", type=int, default=25000) # GenImage 샘플 갯수
    parser.add_argument("--real", type=int, default=25000) # ImageNet 샘플 갯수
    parser.add_argument("--batch_size", type=int, default=16) # 배치 사이즈
    parser.add_argument("--noise", type=float, default=0.05) # noise 조절
    parser.add_argument("--model_id", type=str, default="dinov2_vitb14") # 백본
    parser.add_argument("--checkpoint_path", type=str, default="") # 체크포인트
    args = parser.parse_args()
    if args.fake % 8 != 0 and args.dataset == "all": print("Fake Sample의 갯수는 8로 나누어 떨어져야 함.")

    _main(args)