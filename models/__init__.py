from . import resnet
from . import densenet
from . import vit

def get_model(model_name, input_shape):

    if model_name in resnet.__dict__:
        model = resnet.__dict__[model_name]()
    elif model_name in densenet.__dict__:
        model = densenet.__dict__[model_name]()
    elif model_name in vit.__dict__:
        model = vit.__dict__[model_name]()
    else:
        raise ValueError(f"null: {model_name}")
    
    return model
