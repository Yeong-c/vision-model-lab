from datetime import datetime, timedelta, timezone
import sys
import os, argparse
import glob, tqdm
import random

import torch, torchvision
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from diffusers import AutoencoderKL, VQModel

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from PIL import Image

from methods.aeroblade import Aeroblade, Aeroblade_VAE
from methods.rigid import Rigid
from methods.ours import DecoderAnalysisMethod

# RAE 폴더 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "RAE"))
from RAE.rae import RAE


def get_rae():
    DECODER_CONFIG_DIR = "./RAE/configs"
    PRETRAINED_WEIGHTS = "./RAE/models/decoder_model.pt"
    ENCODER_ID = "facebook/dinov2-with-registers-base"

    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path=ENCODER_ID,
        encoder_params={'dinov2_path': ENCODER_ID, 'normalize': True},
        decoder_config_path=DECODER_CONFIG_DIR,
        pretrained_decoder_path=PRETRAINED_WEIGHTS,
        reshape_to_2d=True,
        noise_tau=0.0
    )
    return model

def calculate_accuracies(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    if np.mean(y_score[y_true == 1]) < np.mean(y_score[y_true == 0]):
        y_score = -y_score
        
    # Standard Acc (0.5)
    y_pred_std = (y_score > 0.5).astype(int)
    std_acc = accuracy_score(y_true, y_pred_std)
    
    # Best Threshold Acc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_acc = 0
    best_tau = 0
    for t in thresholds:
        y_pred_tmp = (y_score > t).astype(int)
        acc = accuracy_score(y_true, y_pred_tmp)
        if acc > best_acc:
            best_acc = acc
            best_tau = t
            
    return std_acc, best_acc, best_tau

def _main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import dataset
    dataloader = dataset.get_mixed_dataloader(args, False)

    import numpy as np

    # [Logging] 실험 결과 폴더 생성
    os.makedirs("./results", exist_ok=True) # results 폴더 생성
    # 실험 이름, 시간 기록
    start_time = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M") #KST로
    result_name = f"{start_time}_{args.dataset}_{args.fake}_{args.real}"
    

    if args.method.lower() == "aeroblade":
        # 실험 저장용 개별 폴더 생성
        result_dir = f"./results/aeroblade_{result_name}"
        os.makedirs(f"{result_dir}", exist_ok=True)

        model = get_rae().to(device)
        model.eval()

        detector = Aeroblade(model, device)

        # ** LPIPS 메인 레이어(1-5)
        MAIN_LAYER = 2

        # Labels
        all_labels = []
        ls = [[] for _ in range(5)] # Layer Score

        for x, label, _ in tqdm.tqdm(dataloader, desc="AEROBLADE TEST"): # idx 0 real, 1 fake
        # for x, label in dataloader: # idx 0 real, 1 fake
            x = x.to(device)
            scores = detector.get_layer_score(x)

            for i in range(5):
                ls[i].extend((-scores[i]).cpu().tolist())
            all_labels.extend(label.tolist())
        
        # All Layer Score 기록 all_layer_scores[LAYER] = LAYER의 점수
        all_layer_scores = ls

        # AUC, AP 메인 레이어 결과 출력
        auc = roc_auc_score(all_labels, all_layer_scores[MAIN_LAYER - 1])
        ap = average_precision_score(all_labels, all_layer_scores[MAIN_LAYER - 1])

        std_acc, best_acc, best_tau = calculate_accuracies(all_labels, all_layer_scores[MAIN_LAYER - 1])

        print_str = ""

        print_str += ("=" * 50) + "\n"
        print_str += (f"[Result]\nMethod: AEROBLADE | Dataset: {args.dataset} | Real/Fake Sample: {args.real}/{args.fake}\n")
        print_str += (f"Main LPIPS Layer: {MAIN_LAYER}\n")
        print_str += (f"  >>  AUC: {auc:.4f} | AP: {ap:.4f}\n")
        print_str += (f"  >>  Acc (0.5): {std_acc:.4f} | Best Acc: {best_acc:.4f} (at tau={best_tau:.4f})\n")
        print_str += ("=" * 50) + "\n"

        # [Logging]
        # 1. Distributions of recunstruction error
        # KDE Plot (Real, Fake의 Error 분포 확인. 논문 Figure 3)
        real_scores = []
        fake_scores = []
        for i in range(len(all_labels)):
            if all_labels[i] == 0: real_scores.append(-(all_layer_scores[MAIN_LAYER - 1][i]))
            else: fake_scores.append(-(all_layer_scores[MAIN_LAYER - 1][i]))

        plt.figure(figsize=(10, 6))
        sns.kdeplot(real_scores, label='Real (ImageNet)', fill=True)
        sns.kdeplot(fake_scores, label=f'Fake ({args.dataset})', fill=True)
        plt.legend()
        plt.title(f"Distributions of recunstruction error (Layer {MAIN_LAYER})")
        plt.xlabel(f"LPIPS{MAIN_LAYER}")
        plt.ylabel("Count")

        plt.savefig(f"{result_dir}/kde_distribution.png")

        # 2. Layer별 AUC, AP (real only, fake only 아닐 때만)
        if len(np.unique(all_labels)) > 1:
            layer_auc = []
            layer_ap = []
            for i in range(5):
                layer_auc.append(roc_auc_score(all_labels, all_layer_scores[i]))
                layer_ap.append(average_precision_score(all_labels, all_layer_scores[i]))
            plt.figure(figsize=(12, 6))
            idx = np.arange(5)
            plt.bar(idx - 0.15, layer_auc, 0.3, color="orange", label="AUC")
            plt.bar(idx + 0.15, layer_ap, 0.3, color="blue", label="AP")
            plt.title(f"AUC, AP by Layer ({args.dataset})")
            plt.legend()
            plt.xticks(idx, ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5"])
            plt.savefig(f"{result_dir}/layer_auc_ap.png")

            print_str += ("[AUC by Layer]\n")
            for i in range(1, 6): print_str += (f"Layer {i}: {layer_auc[i-1]:.5f} ") + ("\n" if i % 2 == 0 else "")
            print_str += "\n\n"
            print_str += ("[AP by Layer]\n")
            for i in range(1, 6): print_str += (f"Layer {i}: {layer_ap[i-1]:.5f} ") + ("\n" if i % 2 == 0 else "")
            print_str += "\n\n"

        # 4. Args 저장
        import json
        with open(f"{result_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # 5. 결과 리포트 출력 및 저장 (텍스트 파일)
        print(print_str)
        report_path = f"{result_dir}/report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(print_str)

    elif args.method.lower() == "aeroblade_vae":
        # 실험 저장용 개별 폴더 생성
        result_dir = f"./results/aeroblade_vae_{result_name}"
        os.makedirs(f"{result_dir}", exist_ok=True)

        ae_models = {
        "sd1": "runwayml/stable-diffusion-v1-5", # SD1.1, 1.5용
        "sd2": "stabilityai/sd-vae-ft-mse",       # SD2.1용 (논문 내 SD2 AE)
        "kd21": "kandinsky-community/kandinsky-2-1", # Kandinsky 2.1용 VQ-VAE
        }
    
        ensemble = dict()
        
        # SD 1.5 VAE
        ensemble['sd1'] = AutoencoderKL.from_pretrained(ae_models['sd1'], subfolder="vae").to(device).eval()
        # SD 2.1 VAE (ft-MSE 버전이 논문의 SD2 성능과 유사함)
        ensemble['sd2'] = AutoencoderKL.from_pretrained(ae_models['sd2']).to(device).eval()
        # Kandinsky 2.1 VQ-VAE 
        ensemble['kd21'] = VQModel.from_pretrained(ae_models['kd21'], subfolder="movq").to(device).eval()

        detector = Aeroblade_VAE(ensemble, device)

        # ** LPIPS 메인 레이어(1-5)
        MAIN_LAYER = 2

        # Labels
        all_labels = []
        ls = [[] for _ in range(5)] # Layer Score

        for x, label, _ in tqdm.tqdm(dataloader, desc="AEROBLADE VAE TEST"): # idx 0 real, 1 fake
        # for x, label in dataloader: # idx 0 real, 1 fake
            x = x.to(device)
            scores = detector.get_layer_score(x)

            for i in range(5):
                ls[i].extend((-scores[i]).cpu().tolist())
            all_labels.extend(label.tolist())
        
        # All Layer Score 기록 all_layer_scores[LAYER] = LAYER의 점수
        all_layer_scores = ls

        # AUC, AP 메인 레이어 결과 출력
        auc = roc_auc_score(all_labels, all_layer_scores[MAIN_LAYER - 1])
        ap = average_precision_score(all_labels, all_layer_scores[MAIN_LAYER - 1])

        std_acc, best_acc, best_tau = calculate_accuracies(all_labels, np.array(all_layer_scores[MAIN_LAYER - 1]))

        print_str = ""

        print_str += ("=" * 50) + "\n"
        print_str += (f"[Result]\nMethod: AEROBLADE | Dataset: {args.dataset} | Real/Fake Sample: {args.real}/{args.fake}\n")
        print_str += (f"Main LPIPS Layer: {MAIN_LAYER}\n")
        print_str += (f"  >>  AUC: {auc:.4f} | AP: {ap:.4f}\n")
        print_str += (f"  >>  Acc (0.5): {std_acc:.4f} | Best Acc: {best_acc:.4f} (at tau={best_tau:.4f})\n")
        print_str += ("=" * 50) + "\n"

        # [Logging]
        # 1. Distributions of recunstruction error
        # KDE Plot (Real, Fake의 Error 분포 확인. 논문 Figure 3)
        real_scores = []
        fake_scores = []
        for i in range(len(all_labels)):
            if all_labels[i] == 0: real_scores.append(-(all_layer_scores[MAIN_LAYER - 1][i]))
            else: fake_scores.append(-(all_layer_scores[MAIN_LAYER - 1][i]))

        plt.figure(figsize=(10, 6))
        sns.kdeplot(real_scores, label='Real (ImageNet)', fill=True)
        sns.kdeplot(fake_scores, label=f'Fake ({args.dataset})', fill=True)
        plt.legend()
        plt.title(f"Distributions of recunstruction error (Layer {MAIN_LAYER})")
        plt.xlabel(f"LPIPS{MAIN_LAYER}")
        plt.ylabel("Count")

        plt.savefig(f"{result_dir}/kde_distribution.png")

        # 2. Layer별 AUC, AP (real only, fake only 아닐 때만)
        if len(np.unique(all_labels)) > 1:
            layer_auc = []
            layer_ap = []
            for i in range(5):
                layer_auc.append(roc_auc_score(all_labels, all_layer_scores[i]))
                layer_ap.append(average_precision_score(all_labels, all_layer_scores[i]))
            plt.figure(figsize=(12, 6))
            idx = np.arange(5)
            plt.bar(idx - 0.15, layer_auc, 0.3, color="orange", label="AUC")
            plt.bar(idx + 0.15, layer_ap, 0.3, color="blue", label="AP")
            plt.title(f"AUC, AP by Layer ({args.dataset})")
            plt.legend()
            plt.xticks(idx, ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5"])
            plt.savefig(f"{result_dir}/layer_auc_ap.png")

            print_str += ("[AUC by Layer]\n")
            for i in range(1, 6): print_str += (f"Layer {i}: {layer_auc[i-1]:.5f} ") + ("\n" if i % 2 == 0 else "")
            print_str += "\n\n"
            print_str += ("[AP by Layer]\n")
            for i in range(1, 6): print_str += (f"Layer {i}: {layer_ap[i-1]:.5f} ") + ("\n" if i % 2 == 0 else "")
            print_str += "\n\n"

        # 4. Args 저장
        import json
        with open(f"{result_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # 5. 결과 리포트 출력 및 저장 (텍스트 파일)
        print(print_str)
        report_path = f"{result_dir}/report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(print_str)

    elif args.method.lower() == "rigid":
 
        detector = Rigid(device, "facebook/dinov2-with-registers-base", register=True)
        # detector = Rigid(device, "facebook/dinov2-with-registers-large", register=True)
        # detector = Rigid(device, "dinov2_vitl14", register=False)

        # 실험 저장용 개별 폴더 생성
        result_dir = f"./results/rigid_{result_name}"
        os.makedirs(f"{result_dir}", exist_ok=True)

        all_scores = []
        all_labels = []

        for x, label, _ in tqdm.tqdm(dataloader, desc="RIGID TEST"):
            x = x.to(device)
            
            # RIGID 점수 계산 (1 - Cosine Similarity)
            # 점수가 높을수록(유사도가 낮을수록) fake일 확률이 높음 (일반적인 경우)
            # sd 모델의 경우 real이 fake보다 더 노이즈에 취약한 부분을 확인 가능.
            scores = detector.get_score(x, noise_level=args.noise)

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()

            all_scores.extend(scores)
            all_labels.extend(label.tolist())

        # 결과 계산
        auc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)

        std_acc, best_acc, best_tau = calculate_accuracies(all_labels, all_scores)

        print_str = ""
        print_str += ("=" * 50) + "\n"
        print_str += (f"[Result]\nMethod: Rigid | Dataset: {args.dataset} | Real/Fake Sample: {args.real}/{args.fake}\n")
        print_str += (f"Noise Level: {args.noise}\n")
        print_str += (f" >>  AUC: {auc:.4f} | AP: {ap:.4f}\n")
        print_str += (f"  >>  Acc (0.5): {std_acc:.4f} | Best Acc: {best_acc:.4f} (at tau={best_tau:.4f})\n")
        print_str += ("=" * 50) + "\n"

        # KDE Plot
        real_scores = []
        fake_scores = []
        for i in range(len(all_labels)):
            if all_labels[i] == 0: real_scores.append(all_scores[i])
            else: fake_scores.append(all_scores[i])

        plt.figure(figsize=(10, 6))
        sns.kdeplot(real_scores, label='Real (ImageNet)', fill=True, color='blue')
        sns.kdeplot(fake_scores, label=f'Fake ({args.dataset})', fill=True, color='red')
        plt.legend()
        plt.title(f"Score Distribution (Rigid, Noise={args.noise})")
        plt.xlabel("Rigid Score (1 - Cosine Sim)")
        plt.ylabel("Density")
        plt.savefig(f"{result_dir}/kde_distribution.png")
        plt.close()

        # 3. Args 저장
        import json
        with open(f"{result_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # 결과 리포트 출력 및 저장
        print(print_str)
        with open(f"{result_dir}/report.txt", "w", encoding="utf-8") as f:
            f.write(print_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="none") # AEROBLADE, RIGID
    parser.add_argument("--dataset", type=str, default="all")
    '''
    dataset: all -> 모든 GenImage 샘플링(모델별)
    dataset: adm, biggan... -> 특정 GenImage 샘플링
    '''
    parser.add_argument("--fake", type=int, default=25000) # GenImage 샘플 갯수
    parser.add_argument("--real", type=int, default=25000) # ImageNet 샘플 갯수
    parser.add_argument("--batch_size", type=int, default=16) # 배치 사이즈
    parser.add_argument("--noise", type=float, default=0.05) # noise 조절
    args = parser.parse_args()

    if args.fake % 8 != 0 and args.dataset == "all": print("Fake Sample의 갯수는 8로 나누어 떨어져야 함.")

    _main(args)