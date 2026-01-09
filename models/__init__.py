from .densenet import DenseNet
from .vit import ViT

from . import resnet

def get_model(model_name, input_shape):

    if model_name in resnet.__dict__:
        model = resnet.__dict__[model_name]()
    elif model_name == "densenet":
        model = DenseNet(input_shape=input_shape)
    elif model_name == "vit":
        model = ViT(input_shape=input_shape)
    else:
        raise ValueError(f"null: {model_name}")
    
    return model
