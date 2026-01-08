import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class TwoTransform:
    # 들어온 transform으로 transform
    def __init__(self, transform):
        self.transform = transform

    # Call시 x를 x1, x2로 Augmentation
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return torch.stack((x1, x2))

def get_dataloader(dataset_name, type, batch_size, num_workers):
    # Size 미리 결정
    # Dataset 클래스 정하기
    if dataset_name == "cifar10":
        Dataset = datasets.CIFAR10
    elif dataset_name == "stl10":
        Dataset = datasets.STL10
    elif dataset_name == "imagenet":
        Dataset = datasets.ImageFolder

    # Type에 따라 Train Transform
    # Type: General Learning(기본), Constrastive Learning
    if type == "general":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    elif type == "contrastive":
        train_transform = TwoTransform(transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]))

    # Test Transform
    # Resize(32) CenterCrop(32) => 32x32
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Dataset 로드
    if dataset_name == "cifar10":
        train_set = Dataset("./data", train=True, transform=train_transform, download=True)
        val_set = Dataset("./data", train=True, transform=test_transform, download=True)
        test_set = Dataset("./data", train=False, transform=test_transform, download=True)

    elif dataset_name == "stl10":
        train_split = "train" if type == "general" else "unlabeled"
        train_split2 = "train"

        train_set = Dataset("./data", split=train_split, transform=train_transform, download=True)
        val_set = Dataset("./data", split=train_split2, transform=test_transform, download=True)
        test_set = Dataset("./data", split="test", transform=test_transform, download=True)

    elif dataset_name == "imagenet":
        train_set = Dataset("./data/imagenet/train", transform=train_transform)
        val_set = Dataset("./data/imagenet/train", transform=test_transform)
        test_set = Dataset("./data/imagenet/val", transform=test_transform)
    
    # 테스트 용 데이터셋 축소
    train_set, _ = torch.utils.data.random_split(train_set, [500, 49500])
    test_set, _ = torch.utils.data.random_split(test_set, [500, 9500])

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set , batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    shuffle=False)

    # DataLoader 튜플 리턴
    return train_loader, val_loader, test_loader
    
