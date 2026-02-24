import os
import glob
import random

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms

# ImageNet-1k Validation Set 50000장 경로
IMAGENET_PATH = "data/imagenet-1k"
# GenImage Test Set 48000장 경로 (adm, biggan, ...)
# 각 폴더에 ai, nature 폴더 존재 -> ai 만 사용
GENIMAGE_PATH = "data/genimage"
GENMODELS = ["adm", "biggan", "glide", "midjourney", "sdv4",
             "sdv5", "vqdm", "wukong"]
             # ntire_val은 in wild dataset

class GenImageDataset(Dataset):
    def __init__(self, name, is_train, num_samples, transform):
        self.transform = transform
        if is_train:
            self.genimage_path = os.path.join(GENIMAGE_PATH, "train")
            self.max_each = 1000
        else:
            self.genimage_path = os.path.join(GENIMAGE_PATH, "test")
            self.max_each = 5000

        selected_models = [name]
        if name == "all": 
            selected_models = GENMODELS[:-1]
            per_model = min(num_samples // len(selected_models), self.max_each)
        else:
            per_model = min(num_samples, self.max_each)
        
        self.images = []
        for model in selected_models:
            # genimage/train or val/<name>/ 내 데이터
            dir = os.path.join(self.genimage_path, model, "*")
            all_files = glob.glob(dir, recursive=True)

            self.images.extend(random.sample(all_files, per_model))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, 1, os.path.relpath(img_path, start=".")
    
class ImageNetDataset(Dataset):
    def __init__(self, is_train, num_samples, transform):
        if is_train:
            self.imagenet_path = os.path.join(IMAGENET_PATH, "train")
            self.max_each = 10000
        else:
            self.imagenet_path = os.path.join(IMAGENET_PATH, "test")
            self.max_each = 50000
        self.transform = transform
        self.imagenet_path = random.sample(glob.glob(os.path.join(self.imagenet_path, "*"), recursive=True), min(num_samples, self.max_each))

    def __len__(self):
        return len(self.imagenet_path)
    
    def __getitem__(self, index):
        img_path = self.imagenet_path[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, 0, os.path.relpath(img_path, start=".")

class ChameleonDataset(Dataset):
    def __init__(self, real_num_samples, fake_num_samples, transform):
        self.transform = transform
        
        # 경로 설정
        base_path = "data/chameleon" 
        # 항상 테스트용임
        real_dir = os.path.join(base_path, "real")
        fake_dir = os.path.join(base_path, "fake")

        # 파일 목록
        real_files = glob.glob(os.path.join(real_dir, "*.jpg"))
        fake_files = glob.glob(os.path.join(fake_dir, "*.jpg"))
        
        # 가용한 데이터보다 요청량이 많을 경우를 대비해 min 처리
        sampled_real = random.sample(real_files, min(real_num_samples, len(real_files)))
        sampled_fake = random.sample(fake_files, min(fake_num_samples, len(fake_files)))

        # Real: 0, Fake: 1
        self.all_images = []
        for path in sampled_real:
            self.all_images.append((path, 0))
            
        for path in sampled_fake:
            self.all_images.append((path, 1))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        img_path, label = self.all_images[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        return img, label, os.path.relpath(img_path, start=".")

def get_mixed_dataloader(args, is_train):
    transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    if args.dataset.lower() == "chameleon": # 카멜레온 셋이면
        # 여기서 알아서 샘플링
        total_dataset = ChameleonDataset(args.real, args.fake, transform)
        dataloader = DataLoader(
            total_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        return dataloader

    # Real 데이터셋
    real_dataset = ImageNetDataset(is_train=is_train, num_samples=args.real, transform=transform)

    # Fake 데이터셋
    fake_dataset = GenImageDataset(args.dataset, is_train, args.fake, transform=transform)

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