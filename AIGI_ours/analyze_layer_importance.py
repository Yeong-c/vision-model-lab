import torch
import matplotlib.pyplot as plt
import numpy as np

def analyze_layer_importance(checkpoint_path, model_type='vits14'):
    configs = {
        'vits14': {'embed_dim': 384, 'num_layers': 12},
        'vitb14': {'embed_dim': 768, 'num_layers': 12},
        'vitl14': {'embed_dim': 1024, 'num_layers': 24}
    }
    dim = configs[model_type]['embed_dim']
    num_layers = configs[model_type]['num_layers']
    
    # weights_only=False로 보안 정책 우회
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['classifier_state_dict']
    
    # 가중치 키 자동 찾기: 'weight'로 끝나고 차원이 1인 텐서 선택
    weight_key = None
    for key in state_dict.keys():
        if 'weight' in key and state_dict[key].ndim == 2: # Linear 레이어 가중치
            weight_key = key
            break
            
    if weight_key is None:
        raise KeyError(f"가중치를 찾을 수 없습니다. 키 목록: {list(state_dict.keys())}")
        
    print(f">> 분석 중인 가중치 키: {weight_key} (from {checkpoint_path})")
    weights = state_dict[weight_key].abs().numpy().flatten()
    
    layer_importance = []
    for i in range(num_layers):
        # delta 부분 (9216 or 18432 or 49152의 앞쪽 절반)
        delta_idx = weights[i*dim : (i+1)*dim]
        # abs_delta 부분 (뒤쪽 절반)
        abs_idx = weights[(num_layers + i)*dim : (num_layers + i + 1)*dim]
        
        importance = (np.mean(delta_idx) + np.mean(abs_idx)) / 2
        layer_importance.append(importance)
        
    return layer_importance

# 실행 및 시각화
models = [
    ('./detector_all_r800_f800_middinov2_vits14_epoch20.pth', 'vits14', 'DINOv2-Small'),
    ('./detector_all_r800_f800_middinov2_vitb14_epoch20.pth', 'vitb14', 'DINOv2-Base'),
    ('./detector_all_r800_f800_middinov2_vitl14_epoch20.pth', 'vitl14', 'DINOv2-Large')
]

plt.figure(figsize=(12, 6))
for path, m_type, label in models:
    importance = analyze_layer_importance(path, m_type)
    plt.plot(range(len(importance)), importance, marker='o', label=label)

plt.xlabel('Layer Index')
plt.ylabel('Average Weight Magnitude (Importance)')
plt.title('Layer-wise Importance Analysis by Model Scale')
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig('layer_importance.png', dpi=300)
plt.show()