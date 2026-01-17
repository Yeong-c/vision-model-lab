from .supervised_learning import SupervisedLearning
from .simclr import SimCLR
from .rotnet import RotNet
from .moco import MoCo
from .simsiam import SimSiam
from .byol import BYOL

from . import evaluate

def wrap_method(method_name, encoder, num_classes):
    if method_name == "supervised":
        model = SupervisedLearning(encoder, num_classes)
    elif method_name == "simclr":
        model = SimCLR(encoder)
        # SimCLR은 Batch Size에 따라 LR 중요 (중요)
        # args.lr = 0.3 * args.batch_size / 256
    elif method_name == "rotnet":
        model = RotNet(encoder)
        #rotnet args.lr = 0.1, args.momentum = 0.9, args.weight_decay=5e-4, args.nesterov=True
    elif method_name == "moco":
        model = MoCo(encoder)
        #moco args.lr = 0.03
    elif method_name == "simsiam":
        model = SimSiam(encoder)
        #lr=0.05~0.1, momentum=0.9, weight_decay=1e-4
    elif method_name == "byol":
        model = BYOL(encoder)
        # momentum=0.999
    return model

# test_model 함수
# 알맞은 테스트를 진행 후 Test Loss와 Accuracy Dict를 리턴
def test_model(args, train_loader, val_loader, test_loader, model, device):
    only_knn_methods = ["simclr", "moco", "simsiam", "byol"]
    test_loss = evaluate.test_loss(test_loader, model, device) if args.method not in only_knn_methods else 0.0

    acc_result = dict() # 결과를 Dict로 묶어서 리턴(RotNet 처럼 테스트 여러 개 가능한 Method를 위해)
    if args.method in ["supervised"]:
        acc_result["Accuracy"] = evaluate.accuracy(test_loader, model, device)
    elif args.method in ["rotnet"]:
        acc_result["Rotation Accuracy"] = evaluate.rot_accuracy(test_loader, model, device)
        acc_result["KNN Accuracy"] = evaluate.KNN(val_loader, test_loader, model, device)
    elif args.method in only_knn_methods:
        acc_result["KNN Accuracy"] = evaluate.KNN(val_loader, test_loader, model, device)

    return test_loss, acc_result
