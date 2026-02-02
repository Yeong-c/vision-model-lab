import torch
import sys
import os, argparse
from methods.aeroblade import AerobladeEvaluator

# RAE 폴더 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "RAE"))
from RAE.rae import RAE

def get_rae():
    DECODER_CONFIG_DIR = "./RAE/configs"
    PRETRAINED_WEIGHTS = "./RAE/models/decoder_model.pt"
    ENCODER_ID = "facebook/dinov2-with-registers-base"

    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path=ENCODER_ID,
        encoder_params={'dinov2_path': ENCODER_ID, 'normalize': True},
        decoder_config_path=DECODER_CONFIG_DIR,
        pretrained_decoder_path=PRETRAINED_WEIGHTS,
        reshape_to_2d=True,
        noise_tau=0.0
    )
    return model

def _main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_rae().to(device)
    model.eval()
    
    if args.method.lower() == "aeroblade":
        evaluator = AerobladeEvaluator(model, device)
        
        # 경로 설정
        REAL_PATH = os.path.expanduser("./data/imagenet/imagenet_500")
        FAKE_ROOT = os.path.expanduser("./data/genimage_subset/genimage")        
        
        if not os.path.exists(REAL_PATH):
            print(f"Error: Real path not found {REAL_PATH}")
            return
        if not os.path.exists(FAKE_ROOT):
            print(f"Error: Fake root not found {FAKE_ROOT}")
            return

        evaluator.run_experiment(REAL_PATH, FAKE_ROOT)
        
    elif args.method.lower() == "rigid":
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="none")
    args = parser.parse_args()

    if args.method.lower() not in ["aeroblade", "rigid"]:
        print("Method must be \"AEROBLADE\" or \"RIGID\"")
        exit()

    _main(args)