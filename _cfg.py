import os

# TODO: change the default noisyart path: ./noisyart/_cfg.py

# TODO: change the default image folder to be used from
#  preprocessing and feature extraction: ./preprocessing/_cfg.py



GPU_INDEX = 0
SEED = 43
USE_FIXED_SEED = True
USE_GPU = True
DEVICE = "cpu"

FEATS_NAMES = ['resnet50_r255_c224_meansub_stdnorm',
               'resnet101_r255_c224_meansub_stdnorm',
               'resnet152_r255_c224_meansub_stdnorm',
               'vgg16_r255_c224_meansub_stdnorm',
               'vgg19_r255_c224_meansub_stdnorm']

TRAINVAL_FEATS = 'trainval_200'
TEST_FEATS = 'test_200'


HIDDEN_UNITS = [4096]


# Training Technique Configurations:
# Each 'column' of the config table is a full-training procedure that will be run (with the standard config: 9 runs).
# LOGDIR = "./runs/standard_exp"
# LF =        [   -1,     300,    580,    -1,     -1,     -1,     -1,     -1,   -1 ]
# ES =        [   -1,     -1,     -1,     80,     80,     300,    300,    80,   80 ]
# TS =        [   -1,     -1,     -1,     -1,     80,     -1,     300,    -1,   80 ]
# BOOT_EP =   [   -1,     -1,     -1,     -1,     -1,     -1,     -1,     80,   80 ]
# -1: disable the corresponding technique
# int: epoch at which LabelFlip (LF), EntropyScaling (ES), TemperatureScaling (TS) will start,
#      or number of epoch for which training with boot-images only (BOOT_EP).


# # Short standard config:
LOGDIR = "./runs/fast_exp"
LF =        [   -1,     580,   -1,     -1 ]
ES =        [   -1,     -1,    80,     80 ]
TS =        [   -1,     -1,    80,     80 ]
BOOT_EP =   [   -1,     -1,    -1,     80 ]
# # - uncomment to use the short config



if USE_FIXED_SEED:
    LOGDIR += f"_seed_{SEED}"
os.makedirs(LOGDIR, exist_ok=True)
##############################################################################################

def init(gpu_index=GPU_INDEX, seed=SEED, use_fixed_seed=USE_FIXED_SEED, use_gpu=USE_GPU):
    import torch
    import numpy as np
    global GPU_INDEX, SEED, USE_FIXED_SEED, USE_GPU, DEVICE
    GPU_INDEX = gpu_index
    SEED= seed
    USE_FIXED_SEED = use_fixed_seed
    USE_GPU = use_gpu


    if USE_FIXED_SEED:
        np.random.seed(SEED)
        torch.manual_seed(SEED)


    if USE_GPU:
        torch.cuda.set_device(GPU_INDEX)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if USE_FIXED_SEED:
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True

    DEVICE = torch.device("cuda:{}".format(GPU_INDEX) if torch.cuda.is_available() else "cpu")

    print("Torch will use device: {}".format(DEVICE))
    return DEVICE



