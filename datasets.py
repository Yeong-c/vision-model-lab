from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

def load_cifar10(transform):
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return (trainset, testset)

def load_dataset(dataset):
    print("Load dataset")
    transform = transforms.Compose( # 정규화
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )
    return load_cifar10(transform) # 일단 cifar10 고정

    # return (trainset, testset)
