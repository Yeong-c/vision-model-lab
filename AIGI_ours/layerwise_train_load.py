import os, argparse, tqdm
import torch
from torch.utils.data import Subset
from methods.layerwise import LayerWiseErrorDetector
import dataset # 성규님의 기존 데이터 로더 모듈

def few_shot_tuning(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 초기화 및 체크포인트 로드
    model_id = args.model_id
    model = LayerWiseErrorDetector(model_id=model_id, device=device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        model.norm.load_state_dict(checkpoint['norm_state_dict'])
        print(f">> Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"❌ Error: Checkpoint {args.checkpoint} not found.")
        return

    model.train()

    # 2. 데이터 로더 준비 (Chameleon 데이터셋 대상)
    # args.dataset을 'chameleon'으로 설정하여 호출한다고 가정
    full_dataloader = dataset.get_mixed_dataloader(args, is_train=True)
    full_dataset = full_dataloader.dataset

    # [핵심] 딱 100개(Few-shot)의 샘플만 무작위로 추출
    indices = torch.randperm(len(full_dataset))[:args.shot_size]
    few_shot_dataset = Subset(full_dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        few_shot_dataset, batch_size=args.batch_size, shuffle=True
    )

    # 3. 학습 설정 (이미 학습된 모델이므로 아주 낮은 LR 사용 권장)
    trainable_params = list(model.classifier.parameters()) + list(model.norm.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f">> Few-shot Tuning on {args.dataset}: {args.shot_size} samples only")
    
    for epoch in range(args.epoch):
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm.tqdm(dataloader, desc=f"Few-shot Epoch {epoch+1}/{args.epoch}")
        
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            
            # 노이즈 주입 전략 유지
            noisy_imgs = torch.clamp(imgs + torch.randn_like(imgs) * args.noise, 0, 1)

            outputs = model(imgs, noisy_imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            predicted = (outputs > 0.0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{running_loss/(pbar.n+1):.4f}", 'acc': f"{100*correct/total:.2f}%"})

    # 4. 적응된 결과 저장
    save_filename = f"tuned_fewshot_{args.shot_size}_{args.checkpoint.split('/')[-1]}"
    torch.save({
        'classifier_state_dict': model.classifier.state_dict(),
        'norm_state_dict': model.norm.state_dict(),
        'args': args
    }, save_filename)
    print(f">> Adaptation Complete. Weights saved to {save_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained .pth")
    parser.add_argument("--dataset", type=str, default="chameleon")
    parser.add_argument("--shot_size", type=int, default=100) # 딱 100장만 사용
    parser.add_argument("--fake", type=int, default=50) # 로더 내부 샘플링용
    parser.add_argument("--real", type=int, default=50) 
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--epoch", type=int, default=5) # 아주 적게 반복
    parser.add_argument("--lr", type=float, default=1e-5) # 매우 낮은 학습률 (Knowledge 유지)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--model_id", type=str, default="dinov2_vitb14")
    args = parser.parse_args()

    few_shot_tuning(args)