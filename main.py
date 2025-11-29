import argparse
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
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


## Training Model Function
def train_model(model, device, train_dataloader, epoch_):
    net = model.to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    # 300 에폭 기준

    for epoch in range(epoch_):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # forward backward optimize
            outputs = net(inputs) # input network 통과
            loss = criterion(outputs, labels) # loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        scheduler.step()    

    print("Finished Training")

# Test Model Function
def test_model(model, device, test_loader):
    print("\n\n**Test Start**")
    model.eval()

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("-" * 30)
    print(f'Total Accuracy: {100 * correct / total:.2f} %')
    print("-" * 30)

    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.1f} %')
    
    print("-" * 30)


def _main(args):
    print("args: ", args)
    
    device = torch.device(args.device)

    trainset, testset = datasets.load_dataset(args.dataset) #일단 그냥 cifar10 고정
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Models
    model = get_model(args.model)
    # print(model)

    # Train
    train_model(model, device, train_dataloader, args.epoch)

    # TEST!!
    test_model(model, device, test_dataloader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet32")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=32)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    _main(args)