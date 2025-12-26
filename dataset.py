import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class SimCLRTransform:
    # SimSiam 참고
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # Call시 x를 x1, x2로 Augmentation
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return torch.stack((x1, x2))

class SupervisedTransform:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            # Supervised도 간단한 증강은 성능이 좋아진다고 함
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __call__(self, x):
        return self.transform(x)

class RotNetTransform:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __call__(self, x):
        return self.transform(x)


def get_dataloader(dataset_name, method, batch_size):
    # Size 미리 결정
    # Dataset 클래스 정하기
    if dataset_name == "cifar10":
        size = 32
        Dataset = datasets.CIFAR10
    elif dataset_name == "stl10":
        size = 96
        Dataset = datasets.STL10
    elif dataset_name == "imagenet":
        size = 224
        Dataset = datasets.ImageFolder

    # Method에 따라 Transform
    if method == "supervised":
        train_transform = SupervisedTransform(size)
    elif method == "simclr":
        train_transform = SimCLRTransform(size)
    elif method == "rotnet":
        train_transform = RotNetTransform(size)

    # Test Transform
    # ImageNet은 크기가 제각각이라 Resize 후 Crop
    if dataset_name == "imagenet":
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        test_transform = transforms.Compose([ # 정규화
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # Dataset 로드
    if dataset_name == "cifar10":
        train_set = Dataset("./data", train=True, transform=train_transform, download=True)
        test_set = Dataset("./data", train=False, transform=test_transform, download=True)
    elif dataset_name == "stl10":
        train_split = "train" if method == "supervised" else "unlabeled"
        train_set = Dataset("./data", split=train_split, transform=train_transform, download=True)
        test_set = Dataset("./data", split="test", transform=test_transform, download=True)
    elif dataset_name == "imagenet":
        train_set = Dataset("./data/imagenet/train", transform=train_transform)
        test_set = Dataset("./data/imagenet/val", transform=test_transform)
    
    # 테스트 용 데이터셋 축소
    #train_set, _ = torch.utils.data.random_split(train_set, [500, 49500])
    #test_set, _ = torch.utils.data.random_split(test_set, [500, 9500])

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # DataLoader 튜플 리턴
    return train_loader, test_loader
    
