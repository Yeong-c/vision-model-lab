import torch
import torch.nn as nn
import lpips

class DiscHead(nn.Module):
    def __init__(self, embed_dim=384): #DINO-s embed_dim = 384
        super().__init__()
        # Discriminator Head (StyleGAN-T 스타일)
        self.first = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv1 = nn.utils.spectral_norm(nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=9, padding=4))
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.cls = nn.utils.spectral_norm(nn.Conv1d(embed_dim, 1, kernel_size=1))

    def forward(self, x):
        out = self.first(x)
        shortcut = out
        out = self.conv1(out)
        out = self.bn1(out)
        out += shortcut
        out = self.lrelu(out)
        out = self.cls(out)
        return out
    
class RAE_Disc(nn.Module):
    def __init__(self):
        super().__init__()
        # Discriminator
        self.disc = torch.hub.load('facebookresearch/dinov2', "dinov2_vits14")
        for p in self.disc.parameters():
            p.requires_grad = False

        self.disc_head = DiscHead(embed_dim=self.disc.embed_dim)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic')

        feature = self.disc.get_intermediate_layers(x, n=1)[0]
        feature = feature.transpose(1, 2)

        logit = self.disc_head(feature)
        return logit.view(x.size(0), -1)



class RAE_Method(nn.Module):
    def __init__(self, AE):
        super().__init__()
        # Representing Autoencoder
        self.AE = AE

        # Loss Function
        # L1
        self.l1_loss = nn.L1Loss()
        # LPIPS (w:1.0)
        self.lpips_loss = lpips.LPIPS(net='alex')

        # Discriminator을 사용하는지
        self.useDisc = False
        # GAN Loss를 더하는지
        self.useGANLoss = False
        # GAN
        self.GAN = RAE_Disc()

        # lambda 계산
        self.lam = 0.0
        # lambda 계산 flag
        self.update_lambda = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    # L_GAN에 곱해지는 Lambda 계산
    def calculate_lambda(self, loss, gan_loss, last_layer):
        # L1, LPIPS Loss Gradient 계산
        l_grads = torch.autograd.grad(loss, last_layer, retain_graph=True)[0]
        # GAN Loss Gradient 계산
        g_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

        # 논문 식대로 계산
        lambda_weight = torch.norm(l_grads) / (torch.norm(g_grads) + 1e-4)
        lambda_weight = torch.clamp(lambda_weight, 0.0, 1.0).detach()

        return lambda_weight
    
    # GAN Disc 업데이트 위한 Hinge Loss
    def hinge_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(nn.functional.relu(1.0 - logits_real))
        loss_fake = torch.mean(nn.functional.relu(1.0 + logits_fake))
        return loss_real + loss_fake
    
    # LPIPS 위한 Denormalize
    def denormalize(self, tensor):
        return tensor * self.std + self.mean

    def forward(self, batch):
        x = batch

        # Input Image Target
        in_image = self.denormalize(x)

        # Image(Encoder -> Decoder) 압축 후 복원
        # Out Image
        out_image_raw = self.AE(x)
        out_image = self.denormalize(out_image_raw)

        # ㅣ1 Loss 계산
        l1_loss = self.l1_loss(in_image, out_image)

        # LPIPS Loss 계산
        lpips_in = in_image * 2 - 1
        lpips_out = out_image * 2 - 1
        lpips_loss = self.lpips_loss(lpips_in, lpips_out).mean()

        total_loss = 1.5 * l1_loss + lpips_loss
        logits_fake = None



        # GAN
        # 6, 7 에포크부터 useDisc True: Discriminator 사용(별도로 학습)
        # 8 에포크부터 useGANLoss True: GAN Loss가 디코더 학습에 관여
        if self.useDisc:
            logits_fake = self.GAN(out_image_raw)
            if self.useGANLoss:
                gan_loss = -torch.mean(logits_fake)
                
                if self.update_lambda: self.lam = self.calculate_lambda(total_loss, gan_loss, self.AE.final_layer[1].weight)

                total_loss = total_loss + 0.75 * self.lam * gan_loss

        # Dict 저장
        loss_dict = {
            "l1_loss": l1_loss.detach(),
            "lpips_loss": lpips_loss.detach(),
            "total_loss": total_loss
        }

        if self.useDisc:
            loss_dict["logits_fake"] = logits_fake.detach().mean()
            if self.useGANLoss:
                loss_dict["gan_loss"] = gan_loss.detach()

        return loss_dict, out_image, logits_fake
    
    def predict(self, x):
        image = self.AE(x)
        return image