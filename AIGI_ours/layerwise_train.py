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

from methods.layerwise import LayerWiseErrorDetector

import dataset

def train_binary_detector(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로더 준비
    # args에는 real, fake, dataset, batch_size 등이 포함되어야 함
    dataloader = dataset.get_mixed_dataloader(args, is_train=False)
    
    # 2. 모델 초기화
    # DINOv2 기반의 모든 레이어 에러 패턴을 분석하는 모델
    model_id = args.model_id # 'dinov2_vitb14'
    model = LayerWiseErrorDetector(model_id=model_id, device=device)
    model.train()
    
    # 3. 학습 설정
    # Classifier와 Norm 레이어만 학습 대상으로 설정
    trainable_params = list(model.classifier.parameters()) + list(model.norm.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    fake_weight = torch.tensor([float(args.real / args.fake)]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=fake_weight)

    print(f">> Training on Mix: Real({args.real}) + Fake({args.fake}, Model: {args.dataset})")
    
    for epoch in range(args.epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epoch}")
        
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            
            # 노이즈 주입 (RIGID 전략: 가짜 이미지의 민감도를 깨우는 시약 역할)
            noise_level = args.noise
            noisy_imgs = imgs + torch.randn_like(imgs) * noise_level
            noisy_imgs = torch.clamp(noisy_imgs, 0, 1)

            # Forward: 원본과 노이즈본의 계층별 차이(Error Pattern)를 판별
            outputs = model(imgs, noisy_imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 통계 계산
            running_loss += loss.item() 

            predicted = (outputs > 0.0).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{running_loss / (pbar.n + 1):.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })


    # 4. 결과 저장
    save_filename = f"detector_{args.dataset}_r{args.real}_f{args.fake}_mid{model_id}_epoch{args.epoch}.pth"
    torch.save({
        'classifier_state_dict': model.classifier.state_dict(),
        'norm_state_dict': model.norm.state_dict(),
        'args': args
    }, save_filename)
    print(f">> Training Complete. Weights saved to {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    '''
    dataset: all -> 모든 GenImage 샘플링(모델별)
    dataset: adm, biggan... -> 특정 GenImage 샘플링
    '''
    parser.add_argument("--fake", type=int, default=25000) # GenImage 샘플 갯수
    parser.add_argument("--real", type=int, default=25000) # ImageNet 샘플 갯수
    parser.add_argument("--batch_size", type=int, default=16) # 배치 사이즈
    parser.add_argument("--epoch", type=int, default=10) # epoch 사이즈
    parser.add_argument("--lr", type=float, default=1e-4) # LR
    parser.add_argument("--noise", type=float, default=0.05) # noise
    parser.add_argument("--model_id", type=str, default="dinov2_vitb14") # 백본
    args = parser.parse_args()

    if args.fake % 8 != 0 and args.dataset == "all": print("Fake Sample의 갯수는 8로 나누어 떨어져야 함.")

    train_binary_detector(args)