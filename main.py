import argparse
import torch
import models, methods, dataset, optims # 우리 것들
import tqdm

# Dataset별 num_classes
dataset_num_classes = {
    "cifar10": 10,
    "stl10": 10,
    "imagenet": 1000
}

# String : Optimizer
optimizers = {
    "LARS": ("LARS", torch.optim.SGD),
    "AdamW": ("AdamW", torch.optim.AdamW),
    "SGD": ("SGD", torch.optim.SGD)
}

# Method, Architecture별 기본 Optimizers
# Optimizers
def get_optimizers(method_name, model_name):
    if model_name == "vit":
        # ViT는 AdamW
        return optimizers["AdamW"]
    if method_name == "simclr":
        # SimCLR은 LARS
        return optimizers["LARS"]
    return optimizers["SGD"]

# MAIN
def _main(args):
    # Set Device
    device = torch.device(args.device)

    # Get DataLoader
    train_loader, test_loader = dataset.get_dataloader(args.dataset, args.method, args.batch_size)

    # Get Model
    model = models.get_model(args.model)

    # Method를 Model에 씌우기
    if args.method == "supervised":
        model = methods.SupervisedLearning(model, num_classes=dataset_num_classes[args.dataset])
    elif args.method == "simclr":
        model = methods.SimCLR(model)
        # SimCLR은 Batch Size에 따라 LR 중요 (중요)
        args.lr = 0.3 * args.batch_size / 256

    # 완전히 형성된 Model을 Device로
    model.to(device)

    # Optimizer, Scheduler, lr 설정
    # Default나 optimizer가 없는 경우에는 기본적 optimizer 사용
    if args.optimizer == "default" or not args.optimizer in optimizers.keys():
        optimizer_name, optimizer = get_optimizers(args.method, args.model)
    else:
        # 아니면 임의의 optimizer
        optimizer_name, optimizer = optimizers[args.optimizer]

    # Optimizer 하이퍼 파라미터 설정
    if optimizer_name == "SGD":
        # SGD 하이퍼 파라미터
        optimizer = optimizer(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "LARS":
        # LARS 하이퍼 파라미터
        optimizer = optims.Lars(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    elif optimizer_name == "AdamW":
        # AdamW 하이퍼 파라미터
        optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=0.05)

    # Scheduler에 Optimizer 매핑
    if optimizer_name == "SGD":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epoch * len(train_loader)
        )
    elif optimizer_name == "LARS" or optimizer_name == "AdamW":
        scheduler = optims.CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps = args.epoch * len(train_loader),
            cycle_mult = 1.0,
            max_lr = args.lr,
            min_lr = 1e-6,
            warmup_steps = int(args.epoch * len(train_loader) * 0.1),
            gamma = 1.0
        )

    # 학습!!!!
    train_model(args, test_loader=test_loader, train_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler, device=device)

def train_one_epoch(args, dataloader, model, optimizer, scheduler, device, epoch_cnt):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader,
                                                desc=f"Epoch {epoch_cnt}",
                                                colour="green")):
        # 1. Device로 넣기
        x, y = batch
        x, y = x.to(device), y.to(device)
        batch_on_device = (x, y)

        # 2. Forward
        loss = model(batch_on_device)

        # 3. Backward, Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. Scheduler Step
        scheduler.step()

        # 5. Loss 누적
        total_loss += loss.item()

        # 10 배치마다 로그 출력
        if (batch_idx + 1) % 10 == 0:
            print(f"  [Batch {batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    # return 에포크 평균 Loss
    return total_loss / len(dataloader)

# Main Training 함수
# train_one_epoch를 Epoch만큼 반복.
def train_model(args, test_loader, train_loader, model, optimizer, scheduler, device):
    print("="*50)
    print("**START TRAINING**")
    print(f"Model: {args.model}\t\tMethod: {args.method}\t\tDataset: {args.dataset}")
    print(f"Epoch: {args.epoch}\t\tBatch Size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}\tScheduler: {args.scheduler}\t\tLR: {args.lr}")
    print("="*50)

    for ep in range(args.epoch):
        train_one_epoch(args, train_loader, model, optimizer, scheduler, device, epoch_cnt=ep+1)
        test_model_accuracy(args, test_loader, model, device)

    print("="*50)
    print("**FINISH TRAINING**")
    print("="*50)


def test_model_accuracy(args, dataloader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Predict로 Output Get
            output = model.predict(x)

            # Accuracy 계산
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    acc = correct / total * 100
    print(f"Accuracy: {acc:.2f}%")
    return acc

def test_model_KNN(args, dataloader, model, device):
    pass



# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--method", type=str, default="supervised")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epoch", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--optimizer", type=str, default="default")
    parser.add_argument("--scheduler", type=str, default="default")
    parser.add_argument("--lr", type=float, default=0.1)

    args = parser.parse_args()
    _main(args)