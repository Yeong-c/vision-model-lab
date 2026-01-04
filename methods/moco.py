import torch
import torch.nn as nn
import copy

class MoCo(nn.Module):
    def __init__(self, model, dim=128, K=4096, m=0.999, T=0.07):
        
        # dim:(dimension), K:(Dictionary Size), m:(momentum), T:(temperature)

        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # 1. 쿼리(Query) 인코더, 키(Key) 인코더
        self.encoder_q = model
        self.encoder_k = copy.deepcopy(model)

        # 2. Projection Head 추가 (MoCo v2 내용 반영)
        self.head_q = nn.Sequential(
            nn.Linear(model.num_features, model.num_features),
            nn.ReLU(inplace=True),
            nn.Linear(model.num_features, dim)
        )
        self.head_k = nn.Sequential(
            nn.Linear(model.num_features, model.num_features),
            nn.ReLU(inplace=True),
            nn.Linear(model.num_features, dim)
        )

        # 3. Key 인코더와 Head는 그래디언트 계산 안 함 
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 값 복사
            param_k.requires_grad = False     # Key는 업데이트 안 함
            
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 4. 큐(Queue) 생성 (Dictionary)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0) # 정규화
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # 포인터

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        키 인코더 모멘텀 업데이트 공식 적용 
        """
        # Backbone 업데이트
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        # Projection Head 업데이트
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # Update the feature queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
       
        assert self.K % batch_size == 0
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    # +) Shuffle BN 구현 
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        batch_size = x.shape[0]  
        # 1. 랜덤 인덱스 생성
        idx_shuffle = torch.randperm(batch_size).to(x.device) 
        # 2. 인덱스 역추적을 위한 unshuffle 인덱스 생성 (나중에 순서 복구용)
        idx_unshuffle = torch.argsort(idx_shuffle)  
        # 3. 이미지 순서 섞기
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        # 다시 원래대로 복구
        return x[idx_unshuffle]

    def forward(self, batch):
        x, _ = batch
        
        im_q = x[:, 0] # 첫 번째 뷰 -> Query
        im_k = x[:, 1] # 두 번째 뷰 -> Key

        # 1. Query 계산
        q = self.encoder_q(im_q) # Backbone
        q = self.head_q(q)       # Projection
        q = nn.functional.normalize(q, dim=1)

        # 2. Key 계산 (Gradient 계산 X)
        with torch.no_grad():
            self._momentum_update_key_encoder()  # 모멘텀 업데이트 실행

            # Shuffle
            im_k_shuffled, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            # Encoder Forward (섞인 이미지로 인코딩)
            k = self.encoder_k(im_k_shuffled)
            k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)

            # Unshuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # 3. Logits 계산
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)      
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # temperature 적용
        logits /= self.T

        # 4. Labels: positives are the 0-th (첫번째가 정답)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # 5. Loss 계산
        loss = self.criterion(logits, labels)

        # 6. Queue 업데이트
        self._dequeue_and_enqueue(k)

        return loss

    def predict(self, x):
        features = self.encoder_q(x) 
        return self.head_q(features)