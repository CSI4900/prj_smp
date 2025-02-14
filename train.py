# *********************************************************************************************
# FILE   NAME:    train.py
# PROJ   NAME:    Segmentation
# DESCRIPTION:    training SMP (Segmentation Models Pytorch)
#
# HOW TO USE:     download CamVid data to
#               git clone https://github.com/alexgkendall/SegNet-Tutorial ./data
#
#
# REVISION HISTORY
# YYYY/MMM/DD     Author       Comments
# 2024 FEB 29     Yu Liu       creation
#
# *********************************************************************************************
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import pandas as pd

from segmentation_models_pytorch import utils
from torch.utils.data import DataLoader
from dataset.camvid import CamVid
from dataset.augment import *
from argparse import ArgumentParser
from config import *


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--last_lr", type=float, default=1e-6)
    parser.add_argument("--lrs_type", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


args = get_arguments()


def train():
    x_train_dir = os.path.join(TRAIN_DATA_DIR, 'train')
    y_train_dir = os.path.join(TRAIN_DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(TRAIN_DATA_DIR, 'val')
    y_valid_dir = os.path.join(TRAIN_DATA_DIR, 'valannot')

    x_test_dir = os.path.join(TRAIN_DATA_DIR, 'test')
    y_test_dir = os.path.join(TRAIN_DATA_DIR, 'testannot')

    """
    dataset = CamVid(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        classes=['car'],
    )

    for i in range(3):
        image, mask = augmented_dataset[1]
        visualize(image=image, mask=mask.squeeze(-1))
    """

    # create segmentation model with pretrained encoder
    # model = smp.FPN(
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    train_dataset = CamVid(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = CamVid(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = utils.losses.DiceLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5),
    ]

    optim = torch.optim.AdamW([
        dict(params=model.parameters(), lr=args.init_lr),
    ])

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.last_lr)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optim,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for args.epochs
    os.makedirs('params', exist_ok=True)
    max_score = 0
    history = []
    for epoch_cnt in range(0, args.epochs):

        print('\nEpoch: {}'.format(epoch_cnt))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, BEST_MODEL_NM)
            print('Model saved!')

        if args.lrs_type == 0:
            if epoch_cnt == 25:
                train_epoch.optimizer.param_groups[0]['lr'] = 1e-5
        elif args.lrs_type == 1:
            sched.step()
        else:
            raise NotImplemented('not defined lr scheduler')

        # Save logs in history
        history.append({
            'epoch': epoch_cnt,
            'lr': train_epoch.optimizer.param_groups[0]['lr'],
            'train_loss': train_logs['dice_loss'],
            'valid_loss': valid_logs['dice_loss'],
            'train_iou': train_logs['iou_score'],
            'valid_iou': valid_logs['iou_score'],
        })

        torch.save({
            'model_param': model.state_dict(),
            'optim_param': optim.state_dict(),
            'sched_param': sched.state_dict(),
            'epoch_count': epoch_cnt,
        }, LATEST_MODE_NM)

    pd.DataFrame(history).to_csv('training_log.csv', index=False)

if __name__ == '__main__':

    train()
