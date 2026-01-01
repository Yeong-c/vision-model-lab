import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, inputs, outputs, patch_size, img_size):

        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        #input을 output으로 변환
        self.proj = nn.Conv2d(inputs, outputs, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class ViT(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), outputs=128, num_Layers=6, n_heads=4, dropout=0.1):

        super(ViT, self).__init__()

        inputs = input_shape[0]
        img_size = input_shape[1]
        
        #패치 사이즈 결정
        if img_size <= 32:
            patch_size = 4
        elif img_size <= 96:
            patch_size = 8
        else:
            patch_size = 16

        #패치 임베딩
        self.patch_embed = PatchEmbedding(inputs, outputs, patch_size, img_size)
        now_features = outputs
        #cls랑 포지션
        self.cls_token = nn.Parameter(torch.zeros(1, 1, now_features))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, now_features))
        self.pos_drop = nn.Dropout(p=dropout)


        dim_feedforward = now_features * 4
        #레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=now_features, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_Layers)

        self.norm = nn.LayerNorm(now_features)

        self.num_features = now_features

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        #패치 임베딩
        x = self.patch_embed(x)
        #cls 확장 / 앞에 추가
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        #포지션 추가
        x = x + self.pos_embed
        x = self.pos_drop(x)
        #트랜스포머
        x = self.transformer(x)
        #cls추출
        out = x[:, 0]
        out = self.norm(out)
        return out
