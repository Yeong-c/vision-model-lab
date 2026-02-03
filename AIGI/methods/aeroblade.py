import torch
import lpips

class Aeroblade():
    def __init__(self, model, device):
        self.rae = model
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval() # 논문: For VGG16, which we mainly use in this work...
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def denormalize(self, x):
        return (x * self.std + self.mean)
    
    def get_score(self, x):
        with torch.no_grad():
            x_norm = x
            x_rec = self.rae(x)
            # RAE 결과물 찍어보니까 -0.5 - 1.5 사이의 값이 나옴 그래서 clamp
            x_rec = torch.clamp(x_rec, 0, 1)

            x_lpips = x_norm * 2.0 - 1.0
            x_rec_lpips = x_rec * 2.0 - 1.0

            outs_ori = self.lpips.net(self.lpips.scaling_layer(x_lpips))
            outs_rec = self.lpips.net(self.lpips.scaling_layer(x_rec_lpips))

            # LPIPS2만 사용
            feat_ori = lpips.normalize_tensor(outs_ori[1])
            feat_rec = lpips.normalize_tensor(outs_rec[1])

            # 5. 차이 계산 및 가중치 적용
            diff = (feat_ori - feat_rec) ** 2
            
            score_map = self.lpips.lins[1](diff)
            
            score = score_map.mean(dim=(1, 2, 3))

        return score