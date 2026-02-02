import torch
import sys
import os, argparse
from torchvision import transforms
from PIL import Image

# RAE 폴더 내의 모듈(decoder, encoder) 임포트를 위해 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "RAE"))

from RAE.rae import RAE

def get_rae():
    # config.json 경로
    DECODER_CONFIG_DIR = "./RAE/configs"
    
    # Pre-trained Weight 경로(decoder_model.pt)
    PRETRAINED_WEIGHTS = "./RAE/models/decoder_model.pt"
    
    # DINOv2wRegBase
    ENCODER_ID = "facebook/dinov2-with-registers-base"

    # RAE 객체 생성
    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path=ENCODER_ID,
        encoder_params={
            'dinov2_path': ENCODER_ID, 
            'normalize': True
        },
        decoder_config_path=DECODER_CONFIG_DIR,
        pretrained_decoder_path=PRETRAINED_WEIGHTS,
        reshape_to_2d=True,
        noise_tau=0.0 # 그냥 통과용이라 노이즈 X
    )
    
    return model

def _main(args):
    # 모델
    model = get_rae().to(device)
    model.eval()

    """
    RAE는 생성 후 aeroblade(model) 식으로
    혹은 rigid() <- 이건 그냥 RAE 참고해서 DINO 불러서 쓰면 될 듯
    """

    

    pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="none")

    args = parser.parse_args()

    if args.method.lower() not in ["aeroblade", "rigid"]:
        print("Method must be \"AEROBLADE\" or \"RIGID\"")
        exit()

    _main(args)

    # 할 것
    # 데이터셋 데이터로더
    # 이미지 받아서 모델 통과
    # 로스 확인 등

    # 256x256 입력