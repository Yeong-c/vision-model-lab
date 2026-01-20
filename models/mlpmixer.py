import torch
import torch.nn as nn

class MlpBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None):
        super(MlpBlock, self).__init__()
        if out_dim is None:
            out_dim = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim):
        super(MixerBlock, self).__init__()

        # 1. Token Mixing (패치들끼리 섞기)
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = MlpBlock(num_patches, token_dim, num_patches)

        # 2. Channel Mixing (채널끼리 섞기)
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = MlpBlock(dim, channel_dim, dim)

    def forward(self, x):
        # x shape: [Batch, Patches, Channels]
        
        # Token Mixing
        residual = x
        x = self.token_norm(x)
        x = x.transpose(1, 2)   # [N, P, C] -> [N, C, P]
        x = self.token_mlp(x)
        x = x.transpose(1, 2)   # [N, C, P] -> [N, P, C]
        x = x + residual        # Skip Connection

        # Channel Mixing
        residual = x
        x = self.channel_norm(x)
        x = self.channel_mlp(x) # 그대로 적용 가능
        x = x + residual        # Skip Connection
        
        return x

class MLPMixer(nn.Module):
    def __init__(self, in_channels=3, dim=512, num_blocks=8, patch_size=4, img_size=32):
        super(MLPMixer, self).__init__()
        
        # 패치 개수 계산
        num_patches = (img_size // patch_size) ** 2
        self.num_features = dim

        # 1. Patch Embedding
        self.patch_emb = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # 2. Mixer Blocks
        # 논문 비율(0.5, 4.0) 적용
        token_dim = int(dim * 0.5) 
        channel_dim = int(dim * 4.0)

        self.blocks = nn.ModuleList([
            MixerBlock(dim, num_patches, token_dim, channel_dim)
            for _ in range(num_blocks)
        ])

        # 3. Final Norm
        self.norm = nn.LayerNorm(dim)

        # 가중치 초기화 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_emb(x) 
        
        # 2. Flatten & Transpose
        x = x.flatten(2).transpose(1, 2)

        # 3. Blocks
        for block in self.blocks:
            x = block(x)

        # 4. Final Norm
        x = self.norm(x)

        # [Batch, Patches, Dim] -> [Batch, Dim]
        x = x.mean(dim=1)

        return x

def mlpmixer_512_8():
    return MLPMixer(dim=512, num_blocks=8, patch_size=4, img_size=32)

def mlpmixer_768_12():
    return MLPMixer(dim=768, num_blocks=12, patch_size=4, img_size=32)

def mlpmixer_1024_24():
    return MLPMixer(dim=1024, num_blocks=24, patch_size=4, img_size=32)