import torch
import torch.nn as nn
from models.resnet50 import ResNet50 

# Loss 함수 
class MoCoLoss(nn.Module):
    def __init__(self):
        super(MoCoLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target_ignored):
        # main.py에서 주는 정답(target)은 무시하고
        # 모델 안에서 계산한 logits와 labels를 씁니다.
        logits, labels = output
        loss = self.criterion(logits, labels)
        return loss

class MoCoResNet50(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.999, T=0.07):
        """
        dim: 임베딩 벡터 크기 
        K: 큐 크기 (Dictionary Size)
        m: 모멘텀 계수
        T: 온도 (Temperature)
        """
        super(MoCoResNet50, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # 1. 쿼리(Query) 인코더: 기존 ResNet50 활용
        self.encoder_q = ResNet50(input_shape=(3, 32, 32), num_classes=dim)

        # 2. 키(Key) 인코더: 구조 똑같이 생성
        self.encoder_k = ResNet50(input_shape=(3, 32, 32), num_classes=dim)

        # 3. Projection Head 교체 (ResNet50은 fc 입력이 2048)
        # 2048 -> 512 -> 128(dim) 구조의 MLP로 바꿔줍니다.
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim, bias=False)
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim, bias=False)
        )

        # 4. 파라미터 초기화 & Key 인코더 그래디언트 끊기
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 값 복사
            param_k.requires_grad = False     # Key 인코더는 그래디언트 계산 안 함

        # 5. 큐(Queue) 생성
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0) # 정규화 작업

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 6. Loss 함수, Optimizer, Scheduler
        self.criterion = MoCoLoss()

        self.optimizer = torch.optim.SGD(
            self.encoder_q.parameters(),
            lr=0.03, # CIFAR10이라 학습률 조금 조정
            momentum=0.9,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        수도코드: f_k.params = m * f_k.params + (1-m) * f_q.params
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 큐 업데이트 (공간 모자르면 잘라서 넣기)
        batch_size = min(keys.shape[0], self.K - ptr)
        
        # Transpose해서 저장 (dim x batch)
        self.queue[:, ptr:ptr + batch_size] = keys.T[:, :batch_size]
        
        ptr = (ptr + batch_size) % self.K 
        self.queue_ptr[0] = ptr

    def forward(self, x):
        
        if x.dim() == 5:
            im_q = x[:, 0]
            im_k = x[:, 1]
        else:
            im_q = x
            im_k = x.clone()

        # 1. Query 인코더 통과
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # 2. Key 인코더 통과 + 모멘텀 업데이트
        with torch.no_grad():
            self._momentum_update_key_encoder() # 모멘텀 업데이트
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        # 3. Logits 계산 (Positive & Negative)
        # Positive: (N, 1)
        l_pos = (q * k).sum(dim=1, keepdim=True)
        
        # Negative: (N, K)
        l_neg = torch.mm(q, self.queue.clone().detach())

        # 합치기
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # 라벨 생성 (Positive는 항상 0번 인덱스)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # 큐 업데이트
        self._dequeue_and_enqueue(k)

        return logits, labels