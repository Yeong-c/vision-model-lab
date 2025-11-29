import argparse
import datasets
from torch.utils.data import DataLoader

def get_model(model_name):
    if model_name == "resnet32":
        from models.resnet32 import ResNet32
        print("Select resnet32")
        return ResNet32()
    elif model_name == "densenet":
        from models.densenet import DenseNet
        print("Select densenet")
        return DenseNet()
    elif model_name == "preresnet":
        from models.preresnet import PreResNet
        print("Select preresnet")
        return PreResNet()
    elif model_name == "resnet18":
        from models.resnet18 import ResNet18
        print("Select ResNet18")
        return ResNet18()
    else:
        # Default ResNet32
        from models.resnet32 import ResNet32
        print("Unknown Model, Default ResNet32")
        return ResNet32()

def _main(args):
    print("args: ", args)
    
    trainset, testset = datasets.load_dataset(args.dataset) #일단 그냥 cifar10 고정
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size)
    test_dataloader = DataLoader(testset, batch_size=args.batch_size)

    # Models
    model = get_model(args.model)
    print(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet32")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    _main(args)