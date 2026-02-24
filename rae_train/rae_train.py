import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from torch.utils.data import Dataset

import argparse
import os
import tqdm
import models.rae, methods.rae

transform = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(256, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.jpeg', '.jpg'))])

        # 테스트 용
        # self.file_list = self.file_list[:4096]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def _main(args):
    device = torch.device(args.device)
    ae_model = models.rae.RAE()
    model = methods.rae.RAE_Method(ae_model).to(device)

    # 디코더 옵티마이저
    opt_decoder = optim.Adam(
        model.AE.parameters(),
        lr=2e-4,
        betas=(0.5, 0.9)
    )
    # Disc 옵티마이저
    opt_disc = optim.Adam(
        model.GAN.parameters(),
        lr=2e-4,
        betas=(0.5, 0.9)
    )

    # 데이터셋 로드(ImageNet-1K)
    dataset = ImageNetDataset(root_dir="./data/imagenet-1k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
    shuffle=True, drop_last=True)
    accumulation_steps = 16 # effective batch
    # scaler, autocast는 메모리를 위해 사용함
    scaler = GradScaler()

    # [Logging] Init
    save_dir = "./AIGI/RAE"
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    best_loss = float('inf')

    global_step = 0

    start_epoch = 1
    best_loss = float('inf')

    # 디코더 스케줄러
    scheduler_decoder = optim.lr_scheduler.CosineAnnealingLR(
        opt_decoder, T_max=args.epoch, eta_min=2e-5
    )

    # Disc 스케줄러
    scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(
        opt_disc, T_max=args.epoch, eta_min=2e-5
    )

    if args.load:
        checkpoint_path = os.path.join(save_dir, "last.pth")
        if os.path.exists(checkpoint_path):
            print(f"==> Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 1. 모델 가중치 복구
            model.load_state_dict(checkpoint['model_state_dict'])
            # 2. 옵티마이저 상태 복구 (Momentum 등 유지 위해 필수)
            opt_decoder.load_state_dict(checkpoint['opt_decoder_state_dict'])
            opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
            # 3. FP16 스케일러 복구
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            # global_step은 대략적으로 계산하거나 checkpoint에 저장하도록 수정 필요
            global_step = (start_epoch - 1) * len(dataloader) // accumulation_steps 

            # [수정] 스케줄러를 현재 epoch에 맞춰 강제 이동
            # scheduler.step()을 start_epoch-1 번 호출하여 LR을 맞춥니다.
            for i in range(start_epoch - 1):
                scheduler_decoder.step()
                if i >= 80: # 80에포크 이후부터만 Disc 스케줄러 업데이트
                    scheduler_disc.step()
            
            print(f"==> Learning Rate Synced: {opt_decoder.param_groups[0]['lr']}")
        else:
            print(f"==> No checkpoint found at {checkpoint_path}")

    # ** 학습 루프
    for epoch in range(start_epoch, args.epoch+1):
        model.train()
        if epoch >= 80:
            model.useDisc = True # Discriminator 학습
        if epoch >= 100:
            model.useGANLoss = True # GAN Loss 반영
        
        for idx, batch in enumerate(tqdm.tqdm(dataloader,
                                                desc=f"Epoch {epoch}",
                                                colour="white")):
            x = batch.to(device)

            step_flag = ((idx + 1) % accumulation_steps == 0)

            with autocast(device_type='cuda'):
                model.update_lambda = True if step_flag or (idx == 0) else False
                loss_dict, out_image, logits_fake = model(x)
                loss_decoder = loss_dict["total_loss"] / accumulation_steps
            
            scaler.scale(loss_decoder).backward()

            if step_flag:
                global_step += 1

                scaler.step(opt_decoder)
                scaler.update()
                opt_decoder.zero_grad()
            
                loss_disc = 0.0
                if model.useDisc and logits_fake is not None:
                    with autocast(device_type='cuda'):
                        logits_real = model.GAN(x)
                        loss_disc = model.hinge_loss(logits_real, logits_fake.detach())

                    scaler.scale(loss_disc).backward()
                    scaler.step(opt_disc)
                    scaler.update()
                    opt_disc.zero_grad()

                # [Logging] 상세 지표 추가 global step 200마다
                if global_step % 25 == 0:
                    writer.add_scalar("Loss/Decoder_Step", loss_dict["total_loss"].item(), global_step)
                    writer.add_scalar("Loss/LPIPS_Only", loss_dict["lpips_loss"].item(), global_step)
                    writer.add_scalar("Weight/Lambda", model.lam, global_step)
                    if "logits_fake" in loss_dict:
                        writer.add_scalar("Logits/Fake_Mean", loss_dict["logits_fake"].item(), global_step)
        
        # LR update
        scheduler_decoder.step()
        if model.useDisc:
            scheduler_disc.step()
        

        # LOSS 출력
        print(f"   [Epoch {epoch}] Decoder Loss: {loss_decoder.item():.4f}, Discriminator Loss: {loss_disc:.4f}")

        # [Logging] Loss 텐서보드 저장
        writer.add_scalar("Loss/Decoder_Epoch", loss_decoder.item(), epoch)
        if model.useDisc and logits_fake is not None: writer.add_scalar("Loss/Discriminator_Epoch", loss_disc, epoch)

        # [Logging] Batch의 첫 번째 이미지를 Encode -> Decode 후 텐서보드 저장(성능 확인용)
        model.eval()
        with torch.no_grad():
            sample = x[0:1]
            sample_out = model.predict(sample)


            res_img = torch.cat([sample[0], sample_out[0]], dim=2).detach().cpu()
            res_img = res_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

            res_err_img = torch.abs(sample[0] - sample_out[0]).detach().cpu()
            res_err_img = res_err_img * 5.0 # 강조

            writer.add_image("Image/Origin_vs_Recons", res_img.clamp(0, 1), epoch)
            writer.add_image("Image/Residual_Error", res_err_img.clamp(0, 1), epoch)

        
        model.train()

        # [Checkpoint]
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_decoder_state_dict': opt_decoder.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss
        }

        torch.save(checkpoint, os.path.join(save_dir, "last.pth"))

        if loss_decoder.item() < best_loss:
            best_loss = loss_decoder.item()
            torch.save(checkpoint, os.path.join(save_dir, "best.pth"))

    writer.close()

# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--load", type=bool, default=False)

    args = parser.parse_args()

    _main(args)