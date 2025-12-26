from .resnet import ResNet

def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = ResNet(num_layers=18)
    elif model_name == "resnet50":
        model = ResNet(num_layers=50)
    elif model_name == "vit":
        model ="vit"
    elif model_name == "densenet":
        model = "densenet"
    else:
        model = "null"
    return model