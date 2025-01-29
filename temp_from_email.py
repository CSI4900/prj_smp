import os
import sys
import cv2
import numpy as np

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

from torchinfo import summary


import utils

import transforms as ext_transforms

from collections import OrderedDict


# from torchvision.models.segmentation import deeplabv3_resnet101 as Model

# from torchvision.models.segmentation import deeplabv3_resnet50 as Model

# from torchvision.models.segmentation import fcn_resnet50 as Model

from torchvision.models.segmentation import lraspp_mobilenet_v3_large as Model


params = './params'

if not os.path.exists(params):

    os.mkdir(params)

results = './results'

if not os.path.exists(results):

    os.mkdir(results)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# param = torch.load('model/fcn_resnet50_coco-1167a1af.pth')

model = Model(pretrained=False, progress=False)

# model.load_state_dict(param)

# summary(model)
