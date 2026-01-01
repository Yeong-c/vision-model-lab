import argparse
import os
import torch
import models, methods, dataset, optims # 우리 것들
import tqdm, math

#rotnet 평가용
from methods.rotnet import rotate_batch

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


dataset_setting = {
    "cifar10":  {"size": 32,  "channels": 3, "classes": 10},
    "stl10":    {"size": 96,  "channels": 3, "classes": 10},
    "imagenet": {"size": 224, "channels": 3, "classes": 1000}
}

# MAIN
def _main(args):
    # Set Device
    device = torch.device(args.device)

    setting = dataset_setting.get(args.dataset, {"size": 224, "channels": 3, "classes": 1000})
    img_size = setting["size"]
    input_shape = (setting["channels"], img_size, img_size)

    train_loader, train_loader2, test_loader = dataset.get_dataloader(args.dataset, args.method, args.batch_size, img_size, args.num_workers)

    model = models.get_model(args.model, input_shape)

    # Method를 Model에 씌우기
    if args.method == "supervised":
        model = methods.SupervisedLearning(model, num_classes=dataset_num_classes[args.dataset])
    elif args.method == "simclr":
        model = methods.SimCLR(model)
        # SimCLR은 Batch Size에 따라 LR 중요 (중요)
        args.lr = 0.3 * args.batch_size / 256
    elif args.method == "rotnet":
        model = methods.RotNet(model)
        args.lr = 0.1
        args.momentum = 0.9
        args.weight_decay=5e-4
        args.nesterov=True

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
        if args.lr>0.01: args.lr = 1e-3 #학습불가 방지
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
    train_model(args, test_loader=test_loader, train_loader=train_loader, train_loader2=train_loader2, model=model, optimizer=optimizer, scheduler=scheduler, device=device)

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
        if (batch_idx + 1) % 5 == 0:
            #print(f"\n[Batch {batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
            pass

    # return 에포크 평균 Loss
    return total_loss / len(dataloader)

# Main Training 함수
# train_one_epoch를 Epoch만큼 반복.
def train_model(args, test_loader, train_loader, train_loader2, model, optimizer, scheduler, device):
    print("="*50)
    print("**START TRAINING**")
    print(f"Model: {args.model}\t\tMethod: {args.method}\t\tDataset: {args.dataset}")
    print(f"Epoch: {args.epoch}\t\tBatch Size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}\tScheduler: {args.scheduler}\t\tLR: {args.lr}")
    print("="*50)

    # 한 Epoch 학습 실행
    for ep in range(args.epoch):
        # 1 Epoch 학습 후 Train Loss 출력
        train_loss = train_one_epoch(args, train_loader, model, optimizer, scheduler, device, epoch_cnt=ep+1)
        print(f"   [Epoch {ep+1}] Train Loss: {train_loss:.4f}")


        # Test(Eval) 진행
        if args.method == "supervised":
            # supervised면 epoch마다 Accuracy 측정 후 출력
            acc = test_model_accuracy(args, test_loader, model, device)
            print(f"   [Epoch {ep+1}] Accuracy: {acc:.2f}%")

        elif args.method == "rotnet":
            # rotnet이면 Rotation Accuracy랑 Feature Accuracy(KNN) 전부 측정
            # 대신 KNN은 5 Epoch마다

            # Rotation Acc
            acc = test_model_accuracy(args, test_loader, model, device)
            print(f"   [Epoch {ep+1}] Rotation Accuracy: {acc:.2f}%")

            # Feature Acc(KNN)
            if (ep + 1) % 5 == 0:
                print("   **Calculating KNN Accuracy**")
                acc = test_model_KNN(args, train_loader, train_loader2, test_loader, model, device, 200)
                print(f"   [Epoch {ep+1}] KNN Accuracy: {acc:.2f}%")

        else:
            # 아니면 Epoch 5개마다 KNN으로 측정
            if (ep + 1) % 5 == 0:
                if args.method == "simclr":
                    print("   **Calculating KNN Accuracy**")
                    acc = test_model_KNN(args, train_loader, train_loader2, test_loader, model, device, 200)
                    print(f"   [Epoch {ep+1}] KNN Accuracy: {acc:.2f}%")

        # 모델 저장(checkpoint)
        os.makedirs("./checkpoints", exist_ok=True)
        save_path = f"./checkpoints/{args.model}_{args.method}_{args.dataset}.pth"

        # 계속 덮어쓰기
        if args.method == "supervised":
            torch.save(model.state_dict(), save_path)
        else:
            # Supervised 아닌건 Backbone State를 저장
            torch.save(model.model.state_dict(), save_path)

    print("="*50)
    print("**FINISH TRAINING**")
    print("="*50)

# Accuracy Test 함수
def test_model_accuracy(args, test_loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch

            if args.method=="rotnet":
                x=x.to(device)
                x,y=rotate_batch(x,device)
            else:
                x, y = x.to(device), y.to(device)

            # Predict로 Output Get
            output = model.predict(x)

            # Accuracy 계산
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    acc = correct / total * 100
    return acc

# KNN Test 함수
def test_model_KNN(args, train_loader, train_loader2, test_loader, model, device, K):
    model.eval()
    correct = 0
    total = 0

    train_features = []
    train_labels = []

    # stl10이면 train_loader2를 사용(unsupervised여도 확인할 땐 labeled로 확인해야 됨)
    if train_loader2 != None: train_loader = train_loader2

    # Train Set Feature들 저장
    with torch.no_grad():
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            if x.dim() == 5:
                x = x[:, 0, :, :, :]
            
            # Backbone에 TrainSet 넣어서 Feature 추출 (model.model)
            feature = model.model(x)
            # 정규화
            feature = torch.nn.functional.normalize(feature, dim=1)

            train_features.append(feature)
            train_labels.append(y)

        train_features = torch.cat(train_features, dim=0).t()
        train_labels = torch.cat(train_labels, dim=0)

        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Test Feature 추출
            test_feature = model.model(x)
            test_feature = torch.nn.functional.normalize(test_feature, dim=1)

            # test feature와 Train Feature들에 대한 거리 계산(행렬곱)
            # (Test, Dim) X (Dim, Train) -> (Test, Train)
            sim_matrix = torch.matmul(test_feature, train_features)

            # 제일 가까운 K개 이웃 찾기 (거리, 인덱스)
            distances, indices = sim_matrix.topk(K, dim=1, largest=True, sorted=True)

            # 인덱스로 라벨 가져오기
            neighbors = train_labels[indices] # [Batch Size, K]

            # 가장 많이 나온 라벨이 뭐지?
            knn, _ = torch.mode(neighbors, dim=1)

            total += x.size(0)
            correct += (knn == y).sum().item()
        
    acc = correct / total * 100
    return acc



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
