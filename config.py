# *********************************************************************************************
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
# *********************************************************************************************
import torch

ENCODER = 'resnet34'  # 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
# could be None for logits or 'softmax2d' for multiclass segmentation
ACTIVATION = 'sigmoid'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   

TRAIN_DATA_DIR = './training_data/CamVid'

LOAD_BEST_MODEL = False
LATEST_MODE_NM = './params/latest2.pt'
BEST_MODEL_NM = './params/best2.pth'
