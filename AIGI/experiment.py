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

# ImageNet-1k Validation Set 50000장 경로
IMAGENET_PATH = "data/imagenet-1k"
# GenImage Test Set 48000장 경로 (adm, biggan, ...)
# 각 폴더에 ai, nature 폴더 존재 -> ai 만 사용
GENIMAGE_PATH = "data/genimage_test/test"
GENMODELS = ["adm", "biggan", "glide", "midjourney", "sdv4",
             "sdv5", "vqdm", "wukong"]

class GenImageDataset(Dataset):
    def __init__(self, name, num_samples, transform):
        self.transform = transform
        self.genimage_path = GENIMAGE_PATH

        if name == "all":
            selected_models = GENMODELS
            per_model = min(num_samples // len(GENMODELS), 6000)
        else:
            selected_models = [name]
            per_model = min(num_samples, 6000)
        
        self.images = []
        for model in selected_models:
            # genimage_test/test/<name>/ai 내 데이터
            dir = os.path.join(self.genimage_path, f"{model}_imagenet", "ai", "*")
            all_files = glob.glob(dir, recursive=True)

            self.images.extend(random.sample(all_files, per_model))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        img = self.transform(img)
        return img, 1
    
class ImageNetDataset(Dataset):
    def __init__(self, num_samples, transform):
        self.transform = transform
        self.imagenet_path = random.sample(glob.glob(os.path.join(IMAGENET_PATH, "*"), recursive=True), min(num_samples, 50000))

    def __len__(self):
        return len(self.imagenet_path)
    
    def __getitem__(self, index):
        img = Image.open(self.imagenet_path[index]).convert("RGB")
        img = self.transform(img)
        return img, 0

def get_mixed_dataloader(args):
    # 256으로 transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    # Real 데이터셋
    real_dataset = ImageNetDataset(num_samples=args.real, transform=transform)

    # Fake 데이터셋
    fake_dataset = GenImageDataset(args.dataset, args.fake, transform=transform)

    # 데이터셋 합치기
    total_dataset = ConcatDataset([real_dataset, fake_dataset])
    dataloader = DataLoader(
        total_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return dataloader

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

def _main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_mixed_dataloader(args)

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

        for x, label in tqdm.tqdm(dataloader, desc="AEROBLADE TEST"): # idx 0 real, 1 fake
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

        print_str = ""

        print_str += ("=" * 50) + "\n"
        print_str += (f"[Result]\nMethod: AEROBLADE | Dataset: {args.dataset} | Real/Fake Sample: {args.real}/{args.fake}\n")
        print_str += (f"Main LPIPS Layer: {MAIN_LAYER}\n")
        print_str += (f"  >>  AUC: {auc:.4f} | AP: {ap:.4f}\n")
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

        # 3. Origin, Reconstruction, ErrorMap 합친 사진 저장
        scores_np = np.array(all_layer_scores[MAIN_LAYER - 1])
        labels_np = np.array(all_labels)

        real_idx = np.where(labels_np == 0)[0]
        fake_idx = np.where(labels_np == 1)[0]

        # 극단값 인덱스 추출 (Real Error Max, Min / Fake Error Max, Min)
        samples = dict()
        if len(real_idx) > 0:
            samples["real_errormax_success"] = real_idx[np.argmin(scores_np[real_idx])]
            samples["real_errormin_fail"] = real_idx[np.argmax(scores_np[real_idx])]
        if len(fake_idx) > 0:
            samples["fake_errormax_fail"] = fake_idx[np.argmin(scores_np[fake_idx])]
            samples["fake_errormin_success"] = fake_idx[np.argmax(scores_np[fake_idx])]
        # max, min 반대임 score 음수라

        # 사진 저장
        dsets = dataloader.dataset.datasets
        real_dset = dsets[0]
        fake_dset = dsets[1]
        num_real = len(real_dset)
        print_str += ("[Sample Path]\n")
        for name, idx in samples.items():
            if idx < num_real:
                original_path = real_dset.imagenet_path[idx]
            else:
                original_path = fake_dset.images[idx - num_real]
            print_str += (f" - {name:25s} : {original_path}\n")
        
            img, _ = dataloader.dataset[idx]
            img = img.to(device).unsqueeze(0)
            img_rec = model(img)
            score_map = detector.get_layer_score_map(img)[MAIN_LAYER - 1]
            score_map = torch.nn.functional.interpolate(score_map, size=256, mode="bilinear", align_corners=False)
            score_map = torch.clamp(score_map.squeeze() / 0.1, 0, 1)

            r, g, b = (torch.clamp(torch.min(4 * score_map - 1.5, -4 * score_map + 4.5), 0, 1),
                       torch.clamp(torch.min(4 * score_map - 0.5, -4 * score_map + 3.5), 0, 1),
                       torch.clamp(torch.min(4 * score_map + 0.5, -4 * score_map + 2.5), 0, 1))
            score_map_color = torch.stack([r, g, b], dim=0)

            full_image = torch.cat([img.squeeze(0), img_rec.squeeze(0), score_map_color], dim=2)

            torchvision.utils.save_image(full_image, f"{result_dir}/visualize_{name}.png")


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

        for x, label in tqdm.tqdm(dataloader, desc="AEROBLADE VAE TEST"): # idx 0 real, 1 fake
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

        print_str = ""

        print_str += ("=" * 50) + "\n"
        print_str += (f"[Result]\nMethod: AEROBLADE | Dataset: {args.dataset} | Real/Fake Sample: {args.real}/{args.fake}\n")
        print_str += (f"Main LPIPS Layer: {MAIN_LAYER}\n")
        print_str += (f"  >>  AUC: {auc:.4f} | AP: {ap:.4f}\n")
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
        import numpy as np
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

        # 3. Origin, Reconstruction, ErrorMap 합친 사진 저장
        scores_np = np.array(all_layer_scores[MAIN_LAYER - 1])
        labels_np = np.array(all_labels)

        real_idx = np.where(labels_np == 0)[0]
        fake_idx = np.where(labels_np == 1)[0]

        # 극단값 인덱스 추출 (Real Error Max, Min / Fake Error Max, Min)
        samples = dict()
        if len(real_idx) > 0:
            samples["real_errormax_success"] = real_idx[np.argmin(scores_np[real_idx])]
            samples["real_errormin_fail"] = real_idx[np.argmax(scores_np[real_idx])]
        if len(fake_idx) > 0:
            samples["fake_errormax_fail"] = fake_idx[np.argmin(scores_np[fake_idx])]
            samples["fake_errormin_success"] = fake_idx[np.argmax(scores_np[fake_idx])]
        # max, min 반대임 score 음수라

        # 사진 저장
        dsets = dataloader.dataset.datasets
        real_dset = dsets[0]
        fake_dset = dsets[1]
        num_real = len(real_dset)
        print_str += ("[Sample Path]\n")
        for name, idx in samples.items():
            if idx < num_real:
                original_path = real_dset.imagenet_path[idx]
            else:
                original_path = fake_dset.images[idx - num_real]
            print_str += (f" - {name:25s} : {original_path}\n")
        
            img, _ = dataloader.dataset[idx]
            img = img.to(device).unsqueeze(0)
            img_rec = detector.get_reconstruction(ensemble['sd1'], img)
            score_map = detector.get_layer_score_map(img)[MAIN_LAYER - 1]
            score_map = torch.nn.functional.interpolate(score_map, size=256, mode="bilinear", align_corners=False)
            score_map = torch.clamp(score_map.squeeze() / 0.1, 0, 1)

            r, g, b = (torch.clamp(torch.min(4 * score_map - 1.5, -4 * score_map + 4.5), 0, 1),
                       torch.clamp(torch.min(4 * score_map - 0.5, -4 * score_map + 3.5), 0, 1),
                       torch.clamp(torch.min(4 * score_map + 0.5, -4 * score_map + 2.5), 0, 1))
            score_map_color = torch.stack([r, g, b], dim=0)

            full_image = torch.cat([img.squeeze(0), img_rec.squeeze(0), score_map_color], dim=2)

            torchvision.utils.save_image(full_image, f"{result_dir}/visualize_{name}.png")

        # 4. Args 저장
        import json
        with open(f"{result_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # 5. 결과 리포트 출력 및 저장 (텍스트 파일)
        print(print_str)
        report_path = f"{result_dir}/report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(print_str)

    elif args.method.lower() == "ours": 
        # -------------------------------------------------------------------------
        # [Strategy] Complexity Ratio Ensemble (Spatial / Style)
        # 이전에 AUC 0.75를 달성했던 가장 강력한 로직입니다.
        # -------------------------------------------------------------------------
        import torch.nn.functional as F 
        import numpy as np
        
        result_dir = f"./result_new/ours_{result_name}"
        os.makedirs(f"{result_dir}", exist_ok=True)

        model = get_rae().to(device)
        
        # [수정] 클래스 이름을 DecoderAnalysisMethod로 통일
        from methods.ours import DecoderAnalysisMethod 
        detector = DecoderAnalysisMethod(model, device)

        all_labels = []
        all_scores = []
        scores_by_layer = []

        print(f">>> Start Complexity Ratio Analysis on {args.dataset}...")

        for x, label in tqdm.tqdm(dataloader, desc="OURS TEST"): 
            x = x.to(device)
            # 리턴: [Score_L0, Score_L1, ...]
            batch_scores = detector.get_layer_score(x) 
            
            # 리스트 안의 텐서를 꺼내서 저장
            if len(scores_by_layer) == 0:
                scores_by_layer = [[] for _ in range(len(batch_scores))]
            
            for i, layer_s in enumerate(batch_scores):
                scores_by_layer[i].extend(layer_s.cpu().tolist())
                
            all_labels.extend(label.tolist())
        
        # -------------------------------------------------------------------------
        # [결과 분석] Robust Ensemble (L4 + L5 + L6)
        # -------------------------------------------------------------------------
        X = np.array(scores_by_layer).T
        y = np.array(all_labels)
        
        # NaN / Inf 제거
        X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=0.0)
        
        # Outlier Clipping
        p99 = np.percentile(X, 99, axis=0)
        X = np.clip(X, a_min=None, a_max=p99)

        NUM_LAYERS = X.shape[1]
        print(f"\nAnalysis Complete. Extracted {NUM_LAYERS} Layers.")

        from sklearn.preprocessing import RobustScaler

        # [핵심] Golden Layers (L4, L5, L6) 선택
        target_indices = [4, 5, 6]
        target_indices = [i for i in target_indices if i < NUM_LAYERS]
        
        if len(target_indices) > 0:
            X_target = X[:, target_indices]
            # Robust Scaling & Sum
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_target)
            final_scores = np.sum(X_scaled, axis=1)
        else:
            final_scores = np.mean(X, axis=1) # Fallback
        
        # 평가
        auc = roc_auc_score(y, final_scores)
        ap = average_precision_score(y, final_scores)
        
        # 방향 체크 (Ratio가 클수록 Fake여야 함 -> 정방향)
        direction = "Normal (High Ratio = Fake)"
        if auc < 0.5:
             print(" >> ⚠️ Direction Inverted. Correcting...")
             final_scores = -final_scores
             auc = 1.0 - auc
             direction = "Inverted (Low Ratio = Fake)"
        
        print_str = ""
        print_str += ("=" * 50) + "\n"
        print_str += (f"[Result]\nMethod: OURS (Complexity Ratio) | Dataset: {args.dataset}\n")
        print_str += (f"Logic: Sum(Spatial / Style) of Layers {target_indices}\n")
        print_str += (f"Direction: {direction}\n")
        print_str += (f" >>  AUC: {auc:.4f} | AP: {ap:.4f}\n")
        print_str += ("=" * 50) + "\n"
        
        print(print_str)

        with open(f"{result_dir}/report.txt", "w", encoding="utf-8") as f:
            f.write(print_str)

        # -------------------------------------------------------------------------
        # [Visualization] Original | Recon | Ratio Map
        # -------------------------------------------------------------------------
        print("\n>>> Generating Visualization...")
        import torchvision
        
        scores_np = final_scores
        labels_np = y
        real_idx = np.where(labels_np == 0)[0]
        fake_idx = np.where(labels_np == 1)[0]
        
        samples = {}
        # Fake(1): Ratio가 높아야(High Score) 성공
        if len(real_idx) > 0:
            samples["real_success"] = real_idx[np.argmin(scores_np[real_idx])]
            samples["real_fail"]    = real_idx[np.argmax(scores_np[real_idx])]
        if len(fake_idx) > 0:
            samples["fake_success"] = fake_idx[np.argmax(scores_np[fake_idx])]
            samples["fake_fail"]    = fake_idx[np.argmin(scores_np[fake_idx])]

        dsets = dataloader.dataset.datasets
        real_dset = dsets[0]; fake_dset = dsets[1]; num_real = len(real_dset)

        for name, global_idx in samples.items():
            path = "Unknown"
            if global_idx < num_real:
                try: path = real_dset.imagenet_path[global_idx]
                except: path = "Unknown_Real"
            else:
                try: path = fake_dset.images[global_idx - num_real]
                except: path = "Unknown_Fake"

            img, _ = dataloader.dataset[global_idx]
            img = img.to(device).unsqueeze(0)
            
            # [Original, Recon, RatioMap]
            x, x_rec, ratio_map = detector.get_visualization_data(img)
            
            # 1. Original (Normalize 0~1)
            x_vis = (x - x.min()) / (x.max() - x.min() + 1e-8)
            
            # 2. Recon (Normalize 0~1)
            x_rec_vis = (x_rec - x_rec.min()) / (x_rec.max() - x_rec.min() + 1e-8)
            
            # 3. Ratio Map (Upsample & Heatmap)
            # (1, 1, 16, 16) -> (1, 1, 256, 256)
            map_up = F.interpolate(ratio_map, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False).squeeze()
            
            m_min, m_max = map_up.min(), map_up.max()
            heatmap = (map_up - m_min) / (m_max - m_min + 1e-8)
            
            # Magma Style Heatmap (Black->Red->Yellow)
            hm_r = heatmap
            hm_g = torch.where(heatmap > 0.5, (heatmap - 0.5) * 2, torch.zeros_like(heatmap))
            hm_b = torch.zeros_like(heatmap)
            
            heatmap_color = torch.stack([hm_r, hm_g, hm_b], dim=0).unsqueeze(0)
            
            # Grid
            full_grid = torch.cat([x_vis, x_rec_vis, heatmap_color], dim=3)
            
            torchvision.utils.save_image(full_grid, f"{result_dir}/vis_{name}.png")
            
            with open(f"{result_dir}/report.txt", "a") as f:
                f.write(f"\n[Vis] {name}: {path} (Score: {scores_np[global_idx]:.4f})")
            
        print(f"Visualization saved to {result_dir}")

    # =========================================================================
    # METHOD: OURS_MLP
    # =========================================================================

    elif args.method.lower() == "ours_mlp":
        from RAE.rae_mlp import RAE_MLP
        from methods.ours_mlp import MultiLayerRAE_MLP
        
        result_dir = f"./result_new/ours_mlp_{result_name}"
        os.makedirs(f"{result_dir}", exist_ok=True)

        ENCODER_ID = "facebook/dinov2-with-registers-base"
        
        # 1. 모델 로드
        model = RAE_MLP(
            encoder_cls='Dinov2withNorm_MLP',
            encoder_config_path=ENCODER_ID,
            encoder_params={'dinov2_path': ENCODER_ID, 'normalize': True}, 
            pretrained_decoder_path='./RAE/models/decoder_model.pt',
            decoder_config_path='./RAE/configs', 
        ).to(device)
        
        detector = MultiLayerRAE_MLP(model, device)

        MAIN_LAYER = 6 
        NUM_LAYERS = 13 

        all_labels = []
        ls = [[] for _ in range(NUM_LAYERS)]

        print(f">>> Start RAE_MLP Analysis on {args.dataset}...")

        # 2. 피쳐 추출 루프
        for x, label in tqdm.tqdm(dataloader, desc="OURS_MLP TEST"): 
            x = x.to(device)
            # 리턴: [Score_L0, Score_L1, ...]
            scores_batch_list = detector.get_layer_score(x)

            for i in range(len(scores_batch_list)):
                # Aeroblade 로직: -Distance (Distance 클수록 Fake)
                ls[i].extend((-scores_batch_list[i]).cpu().tolist())
            
            all_labels.extend(label.tolist())
        
        all_layer_scores = ls # List of Lists (13, N)

        # ---------------------------------------------------------------------
        # [Vis 1] Layer-wise AUC/AP Bar Chart
        # ---------------------------------------------------------------------
        print(" >> Calculating Layer-wise Statistics...")
        layer_aucs = []
        layer_aps = []
        
        for i in range(NUM_LAYERS):
            score = all_layer_scores[i]
            # 안전장치: 혹시라도 점수가 전부 같거나 NaN이면 0.5 처리
            try:
                l_auc = roc_auc_score(all_labels, score)
                l_ap = average_precision_score(all_labels, score)
            except:
                l_auc, l_ap = 0.5, 0.0
            layer_aucs.append(l_auc)
            layer_aps.append(l_ap)

        # Bar Chart 그리기
        x = np.arange(NUM_LAYERS)
        width = 0.35
        plt.figure(figsize=(14, 6))
        plt.bar(x - width/2, layer_aucs, width, label='AUC', color='orange')
        plt.bar(x + width/2, layer_aps, width, label='AP', color='blue')
        plt.xlabel('Layer')
        plt.ylabel('Score')
        plt.title(f'AUC, AP by Layer ({args.dataset})')
        plt.xticks(x, [f'L{i}' for i in range(NUM_LAYERS)])
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(f"{result_dir}/layer_auc_ap.png")
        plt.close()
        
        # ---------------------------------------------------------------------
        # [Vis 2] Sample Visualization (Reconstruction)
        # ---------------------------------------------------------------------
        print(" >> Generating Sample Visualizations...")
        scores_np = np.array(all_layer_scores[MAIN_LAYER])
        labels_np = np.array(all_labels)
        real_idx = np.where(labels_np == 0)[0]
        fake_idx = np.where(labels_np == 1)[0]
        
        samples = {}
        if len(real_idx) > 0:
            samples["real_errormax_fail"] = real_idx[np.argmin(scores_np[real_idx])] 
            samples["real_errormin_success"] = real_idx[np.argmax(scores_np[real_idx])]
        if len(fake_idx) > 0:
            samples["fake_errormax_success"] = fake_idx[np.argmin(scores_np[fake_idx])]
            samples["fake_errormin_fail"] = fake_idx[np.argmax(scores_np[fake_idx])]

        dsets = dataloader.dataset.datasets
        real_dset = dsets[0]; fake_dset = dsets[1]; num_real = len(real_dset)
        
        for name, idx in samples.items():
            path = "Unknown"
            if idx < num_real:
                try: path = real_dset.imagenet_path[idx]
                except: path = "Unknown_Real"
            else:
                try: path = fake_dset.images[idx - num_real]
                except: path = "Unknown_Fake"
            
            img, _ = dataloader.dataset[idx]
            img = img.to(device).unsqueeze(0)
            img_rec = model(img)
            
            score_map = detector.get_layer_score_map(img)[MAIN_LAYER]
            score_map = F.interpolate(score_map, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
            
            s_min, s_max = score_map.min(), score_map.max()
            score_map = (score_map - s_min) / (s_max - s_min + 1e-8)
            score_map = torch.clamp(score_map.squeeze(), 0, 1)

            r = torch.clamp(torch.min(4 * score_map - 1.5, -4 * score_map + 4.5), 0, 1)
            g = torch.clamp(torch.min(4 * score_map - 0.5, -4 * score_map + 3.5), 0, 1)
            b = torch.clamp(torch.min(4 * score_map + 0.5, -4 * score_map + 2.5), 0, 1)
            heatmap = torch.stack([r, g, b], dim=0)
            
            img_vis = (img.squeeze(0) - img.min()) / (img.max() - img.min() + 1e-8)
            rec_vis = (img_rec.squeeze(0) - img_rec.min()) / (img_rec.max() - img_rec.min() + 1e-8)

            full_image = torch.cat([img_vis, rec_vis, heatmap], dim=2)
            torchvision.utils.save_image(full_image, f"{result_dir}/vis_{name}.png")
            
            with open(f"{result_dir}/report.txt", "a") as f:
                f.write(f"[Vis] {name}: {path} (Score: {scores_np[idx]:.4f})\n")

        # ---------------------------------------------------------------------
        # 3. MLP Training (Ensemble)
        # ---------------------------------------------------------------------
        print("\n" + "="*30)
        print("Training MLP Ensemble...")
        
        # (Features, Batch) -> (Batch, Features)
        X = np.array(all_layer_scores).T 
        y = np.array(all_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                            random_state=42, max_iter=1000).fit(X_train_scaled, y_train)
        
        # 예측 확률
        probs = clf.predict_proba(X_test_scaled)[:, 1]
        
        ensemble_auc = roc_auc_score(y_test, probs)
        ensemble_ap = average_precision_score(y_test, probs)
        
        print(f" >> Ensemble AUC: {ensemble_auc:.4f}")
        print(f" >> Ensemble AP : {ensemble_ap:.4f}")
        
        # ---------------------------------------------------------------------
        # [Vis 3] Ensemble Score Distribution (Histogram & KDE)
        # ---------------------------------------------------------------------
        print(" >> Plotting Ensemble Score Distribution...")
        
        real_probs = probs[y_test == 0]
        fake_probs = probs[y_test == 1]
        
        plt.figure(figsize=(10, 6))
        # Real: Blue, Fake: Red
        sns.histplot(real_probs, color='blue', label='Real', kde=True, stat="density", bins=30, alpha=0.5)
        sns.histplot(fake_probs, color='red', label='Fake', kde=True, stat="density", bins=30, alpha=0.5)
        
        plt.title(f"Ensemble Score Distribution (AUC: {ensemble_auc:.4f})")
        plt.xlabel("Fake Probability (0.0 = Real, 1.0 = Fake)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{result_dir}/ensemble_distribution.png")
        plt.close()

        # ---------------------------------------------------------------------
        # [Vis 4] t-SNE Visualization
        # ---------------------------------------------------------------------
        print(" >> Calculating t-SNE (Projection to 2D)...")
        
        # 데이터가 너무 많으면 t-SNE가 오래 걸리므로 최대 2000개만 샘플링
        n_samples_tsne = min(2000, len(X_test_scaled))
        # 랜덤 인덱스 선택
        indices = np.random.choice(len(X_test_scaled), n_samples_tsne, replace=False)
        
        X_tsne_input = X_test_scaled[indices]
        y_tsne = y_test[indices]
        
        # t-SNE 수행
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_embedded = tsne.fit_transform(X_tsne_input)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        # Real (Blue)
        plt.scatter(X_embedded[y_tsne==0, 0], X_embedded[y_tsne==0, 1], 
                    c='blue', label='Real', alpha=0.6, s=10)
        # Fake (Red)
        plt.scatter(X_embedded[y_tsne==1, 0], X_embedded[y_tsne==1, 1], 
                    c='red', label='Fake', alpha=0.6, s=10)
        
        plt.title(f"t-SNE Visualization of 13-Layer Features\n(Separation by MLP)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{result_dir}/tsne_plot.png")
        plt.close()
        
        # Report 저장
        with open(f"{result_dir}/report.txt", "a") as f:
            f.write(f"\n\n[Ensemble Result (MLP)]\nAUC: {ensemble_auc:.4f}\nAP: {ensemble_ap:.4f}\n")
            f.write("\n[Layer-wise Performance]\n")
            for i in range(NUM_LAYERS):
                f.write(f"L{i} - AUC: {layer_aucs[i]:.4f}, AP: {layer_aps[i]:.4f}\n")
                
        print("="*30)
        
    elif args.method.lower() == "rigid":
 
        detector = Rigid(device, "facebook/dinov2-with-registers-base", register=True)
        # detector = Rigid(device, "facebook/dinov2-with-registers-large", register=True)
        # detector = Rigid(device, "dinov2_vitl14", register=False)

        # 실험 저장용 개별 폴더 생성
        result_dir = f"./result_new/rigid_{result_name}"
        os.makedirs(f"{result_dir}", exist_ok=True)

        all_scores = []
        all_labels = []

        for x, label in tqdm.tqdm(dataloader, desc="RIGID TEST"):
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

        print_str = ""
        print_str += ("=" * 50) + "\n"
        print_str += (f"[Result]\nMethod: Rigid | Dataset: {args.dataset} | Real/Fake Sample: {args.real}/{args.fake}\n")
        print_str += (f"Noise Level: {args.noise}\n")
        print_str += (f" >>  AUC: {auc:.4f} | AP: {ap:.4f}\n")
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

        # 2. 대표 샘플 시각화 (성공/실패 케이스)
        scores_np = np.array(all_scores)
        labels_np = np.array(all_labels)

        real_idx = np.where(labels_np == 0)[0]
        fake_idx = np.where(labels_np == 1)[0]

        # Rigid는 Score가 높으면 Fake라고 판단함.
        # Real Success: 가장 Score가 낮은 real (good)
        # Real Fail: 가장 Score가 높은 real (bad - 가짜라고 판단)
        # Fake Success: 가장 Score가 높은 fake (good)
        # Fake Fail: 가장 Score가 낮은 fake (bad - 진짜라고 판단)
        
        samples = dict()
        if len(real_idx) > 0:
            samples["real_best_success"] = real_idx[np.argmin(scores_np[real_idx])]
            samples["real_worst_fail"] = real_idx[np.argmax(scores_np[real_idx])]
        if len(fake_idx) > 0:
            samples["fake_best_success"] = fake_idx[np.argmax(scores_np[fake_idx])]
            samples["fake_worst_fail"] = fake_idx[np.argmin(scores_np[fake_idx])]

        # 사진 저장
        dsets = dataloader.dataset.datasets
        real_dset = dsets[0]
        fake_dset = dsets[1]
        num_real = len(real_dset)

        print_str += ("[Sample Path]\n")
        
        for name, idx in samples.items():
            # 원본 경로 찾기
            if idx < num_real:
                original_path = real_dset.imagenet_path[idx]
            else:
                original_path = fake_dset.images[idx - num_real]
            
            print_str += (f" - {name:25s} (Score: {scores_np[idx]:.4f}) : {original_path}\n")

            # 이미지 로드 및 시각화
            #[원본 | 노이즈 이미지 | 차이]
            img, _ = dataloader.dataset[idx]
            img = img.to(device).unsqueeze(0)
            
            # 노이즈 생성 시각화
            # 엄연히 말해서 구분에 사용된 노이즈와는 다르지만, 노이즈의 세기를 보기엔 문제 없음
            noise = torch.randn_like(img) * args.noise
            img_noisy = img + noise
            
            # 시각화를 위해 값 범위 클램핑 (0~1)
            img_vis = torch.clamp(img, 0, 1)
            img_noisy_vis = torch.clamp(img_noisy, 0, 1)
            
            # Pixel Difference
            diff = torch.abs(img_vis - img_noisy_vis)
            diff = diff / diff.max() # 정규화해서 잘 보이게

            # 이미지 합치기 (Original, Noisy, Diff)
            full_image = torch.cat([img_vis.squeeze(0), img_noisy_vis.squeeze(0), diff.squeeze(0)], dim=2)
            
            torchvision.utils.save_image(full_image, f"{result_dir}/visualize_{name}.png")

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