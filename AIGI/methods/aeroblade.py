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
    
    def get_layer_score(self, x): # 배치
        with torch.no_grad():
            x_norm = x
            x_rec = self.rae(x)
            # RAE 결과물 찍어보니까 -0.5 - 1.5 사이의 값이 나옴 그래서 clamp
            x_rec = torch.clamp(x_rec, 0, 1)

            x_lpips = x_norm * 2.0 - 1.0
            x_rec_lpips = x_rec * 2.0 - 1.0

            x_lpips = self.lpips.scaling_layer(x_lpips)
            x_rec_lpips = self.lpips.scaling_layer(x_rec_lpips)

            outs_ori = self.lpips.net(x_lpips)
            outs_rec = self.lpips.net(x_rec_lpips)

            # 모든 LPIPS 레이어 사용 후 list로 저장
            lpips_layers_score = []
            for i in range(5):
                feat_ori = lpips.normalize_tensor(outs_ori[i])
                feat_rec = lpips.normalize_tensor(outs_rec[i])

                # 차이 계산 및 가중치 적용
                diff = (feat_ori - feat_rec) ** 2
                
                score_map = self.lpips.lins[i](diff)
                
                score = score_map.mean(dim=(1, 2, 3))

                lpips_layers_score.append(score)

        return lpips_layers_score
    
    def get_layer_score_map(self, x): # 이미지 한 장
        with torch.no_grad():
            x_norm = x
            x_rec = self.rae(x)
            # RAE 결과물 찍어보니까 -0.5 - 1.5 사이의 값이 나옴 그래서 clamp
            x_rec = torch.clamp(x_rec, 0, 1)

            x_lpips = x_norm * 2.0 - 1.0
            x_rec_lpips = x_rec * 2.0 - 1.0

            x_lpips = self.lpips.scaling_layer(x_lpips)
            x_rec_lpips = self.lpips.scaling_layer(x_rec_lpips)

            outs_ori = self.lpips.net(x_lpips)
            outs_rec = self.lpips.net(x_rec_lpips)

            # 모든 LPIPS 레이어 사용 후 list로 저장
            lpips_layers_score_map = []
            for i in range(5):
                feat_ori = lpips.normalize_tensor(outs_ori[i])
                feat_rec = lpips.normalize_tensor(outs_rec[i])

                # 차이 계산 및 가중치 적용
                diff = (feat_ori - feat_rec) ** 2
                
                score_map = self.lpips.lins[i](diff)

                lpips_layers_score_map.append(score_map)

        return lpips_layers_score_map