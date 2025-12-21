import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __call__(self, x):
        return torch.stack([self.transform(x), self.transform(x)], dim=0)

def load_cifar10(transform):
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return (trainset, testset)

def load_dataset(dataset):
    print("Load dataset")

    if(dataset == "simclr_cifar10"):
        transform = SimCLRTransform()
    else:
        transform = transforms.Compose( # 정규화
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        )

    return load_cifar10(transform) # 일단 cifar10 고정(simclr 전용일 수도)
    # return (trainset, testset)
