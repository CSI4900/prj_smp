#*********************************************************************************************
# FILE   NAME:    config.py
# PROJ   NAME:    Segmentation
# DESCRIPTION:    configuration of training/validation env
#
# HOW TO USE:
#
#
# REVISION HISTORY
# YYYY/MMM/DD     Author       Comments
# 2024 MAR 01     Yu Liu       creation
#
#*********************************************************************************************

ENCODER = 'resnet34' #'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda:0'

TRAIN_DATA_DIR = '../../data/CamVid'

LOAD_BEST_MODEL = False
LATEST_MODE_NM = './params/latest.pt'
BEST_MODEL_NM = './params/best.pth'
