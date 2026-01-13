import argparse
import os
import torch
import models, methods, dataset, optims # 우리 것들
import tqdm
from datetime import datetime, timedelta, timezone
from torch.utils.tensorboard import SummaryWriter

#rotnet 평가용
from methods.rotnet import rotate_batch

# Dataset별 num_classes
dataset_num_classes = {
    "cifar10": 10,
    "stl10": 10,
    "imagenet": 1000
}

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

    # General, Constrastive Transform Type 선택(Method에 따라)
    if args.method in ["simclr", "moco"]:
        transform_type = "contrastive"
    else:
        transform_type = "general"

    # Type에 맞춰 Data Loader Load
    train_loader, val_loader, test_loader = dataset.get_dataloader(args.dataset, transform_type, args.batch_size, args.num_workers)

    # Backbone 모델 가져오기
    model = models.get_model(args.model, input_shape)

    # Method를 Model에 씌우기 (최종체)
    model = methods.wrap_method(args.method, model, num_classes=dataset_num_classes[args.dataset])

    # 완전히 구성된 Model을 Device로
    model.to(device)

    # Get Optimizer, Optimzer 세팅
    optimizer = optims.get_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    # Scheduler 세팅 (CosineAnnealingLR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch * len(train_loader)
    )

    # 학습!!!!
    train_model(args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, model=model, optimizer=optimizer, scheduler=scheduler, device=device)

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
def train_model(args, train_loader, val_loader, test_loader, model, optimizer, scheduler, device):
    print("="*50)
    print("**START TRAINING**")
    print(f"Model: {args.model}\t\tMethod: {args.method}\t\tDataset: {args.dataset}")
    print(f"Epoch: {args.epoch}\t\tBatch Size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}\t\tLR: {args.lr}")
    print("="*50)

    # 실험 저장 폴더 생성
    os.makedirs("./experiments", exist_ok=True) # experiments 폴더 생성

    # 실험 이름, 시간 기록
    start_time = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M") #KST로
    exp_name = f"{start_time}_{args.model}_{args.method}_{args.dataset}"
    exp_dir = f"./experiments/{exp_name}"

    # 실험 저장용 개별 폴더 생성
    os.makedirs(f"{exp_dir}", exist_ok=True)

    # 실험 Arguments 파일로 저장
    save_arguments(exp_dir, exp_name, start_time, args)

    # [Logging] TensorBoard Writer 생성
    writer = SummaryWriter(log_dir=f"{exp_dir}/logs")

    # Best Accuracy 기록용
    best_all_acc = 0.0

    # 한 Epoch 학습 실행
    for ep in range(args.epoch):
        # 1 Epoch 학습 후 Train Loss 출력
        train_loss = train_one_epoch(args, train_loader, model, optimizer, scheduler, device, epoch_cnt=ep+1)
        print(f"   [Epoch {ep+1}] Train Loss: {train_loss:.4f}")

        # [Logging] Train Loss 기록
        writer.add_scalar("Loss/Train", train_loss, ep)

        # Test(Eval) 진행
        test_loss, acc_result = methods.test_model(args, train_loader, val_loader, test_loader, model, device)

        # Test Loss 출력
        print(f"   [Epoch {ep+1}] Test Loss: {test_loss:.4f}")

        # [Logging] Test Loss 기록
        writer.add_scalar("Loss/Test", test_loss, ep)

        all_acc = 0.0

        # Accuracy Dict로 받아온 값들 전부 출력
        for key, value in acc_result.items():
            all_acc += value

            print(f"   [Epoch {ep+1}] {key}: {value:.2f}%")
            # [Logging] Accuracy 기록
            writer.add_scalar(f"Accuracy/{key}", value, ep)

        # [Logging] Learning Rate 기록 (CosineAnnealing 확인용)
        writer.add_scalar("Params/Learning_Rate", optimizer.param_groups[0]["lr"], ep)

        # Last Checkpoint 저장
        save_checkpoint(exp_dir, "last", ep, model, optimizer, scheduler, all_acc, False)

        # Best Accuracy Update
        if all_acc > best_all_acc:
            best_all_acc = all_acc
            # Best Checkpoint 저장
            save_checkpoint(exp_dir, "best", ep, model, optimizer, scheduler, all_acc, False if args.method == "supervised" else True)

    print("="*50)
    print("**FINISH TRAINING**")
    print("="*50)

def save_arguments(dir, exp_name, start_time, args):
    # arguments 파일로 저장
    with open(f"{dir}/args.txt", "w") as f:
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Start Time: {start_time}\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epoch: {args.epoch}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")

def save_checkpoint(dir, checkpoint_name, epoch, model, optimizer, scheduler, score, only_encoder=False):
    # Full Model 저장 / Encoder 저장
    # Encoder 저장하는데, 실제 Encoder가 있으면
    if only_encoder and hasattr(model, "encoder"):
        model = model.encoder
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "score": score
    }

    torch.save(checkpoint, f"{dir}/{checkpoint_name}.pth")


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
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=0.1)

    args = parser.parse_args()
    _main(args)
