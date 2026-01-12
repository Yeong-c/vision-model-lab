import torch
from . import rotnet

# Test Loss 계산 함수
def test_loss(test_loader, model, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            batch_on_device = (x, y)

            loss = model(batch_on_device)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

# Accuracy 테스트 함수
# Label 비교를 통해 Accuracy 계산 (Supervised)
def accuracy(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Predict로 Output Get
            output = model.predict(x)

            # Accuracy 계산
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    acc = correct / total * 100
    return acc

# Rotation Accuracy 테스트 함수
# Rotated 이미지를 Label로 해서 Accuracy 계산 (RotNet)
def rot_accuracy(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x=x.to(device)
            x,y=rotnet.rotate_batch(x,device)

            # Predict로 Output Get
            output = model.predict(x)

            # Accuracy 계산
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    acc = correct / total * 100
    return acc

# KNN 테스트 함수
# K = 1로 해도 충분
def KNN(val_loader, test_loader, model, device):
    model.eval()
    correct = 0
    total = 0

    train_features = []
    train_labels = []

    # Train Set Feature들 저장
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Backbone에 TrainSet 넣어서 Feature 추출 (model.encoder)
            feature = model.encoder(x)

            # 정규화
            feature = torch.nn.functional.normalize(feature, dim=1)

            train_features.append(feature)
            train_labels.append(y)

        train_features = torch.cat(train_features, dim=0).t()
        train_labels = torch.cat(train_labels, dim=0)

        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Test Feature 추출
            test_feature = model.encoder(x)
            test_feature = torch.nn.functional.normalize(test_feature, dim=1)

            # test feature와 Train Feature들에 대한 거리 계산(행렬곱)
            # (Test, Dim) X (Dim, Train) -> (Test, Train)
            sim_matrix = torch.matmul(test_feature, train_features)

            # K=1이므로 argmax로 가장 가까운 데이터와 비교
            # argmax로 각 이미지마다 가장 가까운 인덱스를 구하고, Label 가져오기
            top_label = train_labels[torch.argmax(sim_matrix, dim=1)]

            # ** 기존의 K개 뽑아서 KNN 진행 로직
            # 제일 가까운 K개 이웃 찾기 (거리, 인덱스)
            #distances, indices = sim_matrix.topk(K, dim=1, largest=True, sorted=True)
            # 인덱스로 라벨 가져오기
            #neighbors = train_labels[indices] # [Batch Size, K]
            # 가장 많이 나온 라벨이 뭐지?
            #knn, _ = torch.mode(neighbors, dim=1)

            total += x.size(0)
            correct += (top_label == y).sum().item()
        
    acc = correct / total * 100
    return acc