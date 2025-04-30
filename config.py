# =====================================================
# File Name:    config.py
# Project Name: Object Segmentation
# Description:  The configuration file for training or
#               validation
#
# Usage:        Import this file into other python
#               programs.
#
# Contributors: 
# - Zechen Zhou     zzhou186@uottawa.ca
# - Shun Hei Yiu    syiu017@uottawa.ca
# =====================================================
import torch

ENCODER = 'mobileone_s4'  # 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 'car', 'truck', 'bus', 
               'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 
               'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'vegetation', 
               'terrain', 'sky', 'ground', 'dynamic', 'static']
# CLASSES = ['road', 'person', 'rider', 'car', 'motorcycle', 'bicycle',  'building',
# 'guard rail', 'sky', 'ground', 'dynamic', 'static']
# CLASSES = ['rider', 'car', 'motorcycle', 'bicycle', 'dynamic']
# could be None for logits or 'softmax2d' for multiclass segmentation
# The activation function applied on the output layer 
ACTIVATION = 'softmax2d'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   

TRAIN_DATA_DIR = './training_data/Cityscapes/smp#1_part_of_leftImg8bit_foggy'
# TRAIN_DATA_DIR = '~/scratch/training_data/Cityscapes/part_of_leftImg8bit_foggy'
VALID_VIDEO_DIR = './validating_data/1_test_bike_car.mp4'
# VALID_VIDEO_DIR = './validating_data/1_test_bike_car_complete.mp4'
# VALID_VIDEO_DIR = './validating_data/Recording-2025-02-25-121337.mp4'
# VALID_VIDEO_DIR = './validating_data/1_test_bike_car.mp4'
# VALID_VIDEO_DIR = './validating_data/racing_cars.sd.mp4'

LOAD_BEST_MODEL = True
LATEST_MODE_NM = './params/smp#3_Unet_mobileone_s4_30classes_softmax2d_50epochs.pt'
BEST_MODEL_NM = './params/smp#3_Unet_mobileone_s4_30classes_softmax2d_50epochs.pth'