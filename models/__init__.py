from .resnet import ResNet
from .densenet import DenseNet
from .vit import ViT

def get_model(model_name, input_shape):

    if model_name == "resnet18":
        model = ResNet(num_layers=18, input_shape=input_shape)  
    elif model_name == "resnet50":
        model = ResNet(num_layers=50, input_shape=input_shape)
    elif model_name == "densenet":
        model = DenseNet(input_shape=input_shape)
    elif model_name == "vit":
        model = ViT(input_shape=input_shape)
    else:
        raise ValueError(f"null: {model_name}")
    
    return model
