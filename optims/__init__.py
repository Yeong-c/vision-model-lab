import torch

def get_optimizer(optimizer_name, model, lr, weight_decay):
    # optimzer_name에 따라 알맞는 optimizer 리턴
    # lr, weight_decay만 arg로 받고, 나머지는 많이 사용하는 옵션으로
    if optimizer_name == "SGD":
        # SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        # AdamW
        if lr > 0.01: lr = 1e-3 # 학습불가 방지
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    return optimizer