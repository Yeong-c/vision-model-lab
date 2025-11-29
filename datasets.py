from torchvision import datasets
from torchvision.transforms import ToTensor

def load_cifar10():
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=ToTensor())
    return (trainset, testset)

def load_dataset(dataset):
    print("Load dataset")
    return load_cifar10() # 일단 cifar10 고정

    # return (trainset, testset)
