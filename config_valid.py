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
CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']
# could be None for logits or 'softmax2d' for multiclass segmentation
ACTIVATION = 'sigmoid'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   

TRAIN_DATA_DIR = './training_data/CamVid'

LOAD_BEST_MODEL = False
LATEST_MODE_NM = './params/resnet34_imagenet_sigmoid_12classes.pt'
BEST_MODEL_NM = './params/resnet34_imagenet_sigmoid_12classes.pth'
