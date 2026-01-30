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
        self.disc = torch.hub.load('facebookresearch/dinov2', "dinov2_vits8")
        for p in self.disc.parameters():
            p.requires_grad = False

        self.disc_head = DiscHead(embed_dim=self.disc.embed_dim)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='area')

        with torch.no_grad():
            feature = self.disc.get_intermediate_layers(x, n=1)[0]
            feature = feature.transpose(1, 2)

        logit = self.disc_head(feature)
        return logit.view(x.size(0), 256)



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

        # GAN을 사용하는지
        self.is_GAN = False
        # GAN
        self.GAN = RAE_Disc()
    
    # L_GAN에 곱해지는 Lambda 계산
    def calculate_lambda(self, loss, gan_loss, last_layer):
        # L1, LPIPS Loss Gradient 계산
        l_grads = torch.autograd.grad(loss, last_layer, retain_graph=True)[0]
        # GAN Loss Gradient 계산
        g_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

        # 논문 식대로 계산
        lambda_weight = torch.norm(l_grads) / (torch.norm(g_grads) + 1e-4)
        lambda_weight = torch.clamp(lambda_weight, 0.0, 1e4).detach()

        return lambda_weight

    def forward(self, batch):
        x, y = batch
        # Input Image Target
        in_image = nn.functional.interpolate(x, size=(256, 256), mode='bicubic')

        # Image(Encoder -> Decoder) 압축 후 복원
        # Out Image
        out_image = self.AE(x)

        # Loss 계산
        l1_loss = self.l1_loss(in_image, out_image)
        lpips_loss = self.lpips_loss(in_image, out_image)

        # GAN(is_GAN이라면)
        if self.is_GAN:
            logits = self.GAN(out_image)
            gan_loss = -torch.mean(logits)
        ##################
        ##################
        # 여기 Hinge Loss 구하는 거 추가해야됨

        return loss
    
    def predict(self, x):
        image = self.AE(x)
        return image