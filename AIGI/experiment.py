import torch
import sys
import os, argparse
import glob, tqdm
import random

from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score, average_precision_score

from PIL import Image

from methods.aeroblade import Aeroblade

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
        shuffle=True,
        num_workers=4,
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

    all_scores = []
    all_labels = []
    
    if args.method.lower() == "aeroblade":
        model = get_rae().to(device)
        model.eval()

        detector = Aeroblade(model, device)

        for x, label in tqdm.tqdm(dataloader, desc="AEROBLADE TEST"): # idx 0 real, 1 fake
        # for x, label in dataloader: # idx 0 real, 1 fake
            x = x.to(device)

            scores = detector.get_score(x)

            all_scores.extend((-scores).cpu().tolist())
            all_labels.extend(label.tolist())

        # AUC, AP
        auc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)

        print(f"\n[Result] Method: AEROBLADE | Dataset: {args.dataset}")
        print(f"AUC: {auc:.4f} | AP: {ap:.4f}")

        '''
        작성중.........
        '''

        
    elif args.method.lower() == "rigid":
        pass

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
    args = parser.parse_args()

    if args.method.lower() not in ["aeroblade", "rigid"]:
        print("Method는 \"AEROBLADE\" 또는 \"RIGID\"여야 함.")
        exit()

    if args.fake % 8 != 0: print("Fake Sample의 갯수는 8로 나누어 떨어져야 함.")

    _main(args)