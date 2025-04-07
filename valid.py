# *********************************************************************************************
# FILE   NAME:    valid.py
# PROJ   NAME:    Segmentation
# DESCRIPTION:    validation of SMP (Segmentation Models Pytorch)
#
# HOW TO USE:     download CamVid data to
#               git clone https://github.com/alexgkendall/SegNet-Tutorial ./data
#
# REFERENCE: https://github.com/qubvel-org/segmentation_models.pytorch
#
# *********************************************************************************************
import time
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from config import *
from collections import OrderedDict

# results = './results_CityScapes_30Classes_val_bike_cars_2nd_test'
# results = './results_CityScapes_30Classes_val_bike_cars_complete'
results = './results'
# results = './results_CityScapes_12Classes_val_racing_cars'
# results = './results_CityScapes_12Classes_val_bike_cars'
# results = './results_CityScapes_30Classes_val_racing_cars_2nd_test'
os.makedirs(results, exist_ok=True)


def valid():
    N_CLASSES = len(CLASSES)
    # create segmentation model with pretrained encoder
    model = smp.Unet(  # smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=N_CLASSES,
        activation=ACTIVATION,
    ).to(DEVICE)

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    if LOAD_BEST_MODEL:
        param = torch.load(BEST_MODEL_NM, map_location=DEVICE, weights_only=False)
        model.load_state_dict(param['model_param'])
    else:
        param = torch.load(LATEST_MODE_NM, map_location=DEVICE, weights_only=False)
        model.load_state_dict(param['model_param'])

    # Set to eval mode
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    # Video setup
    video_file = VALID_VIDEO_DIR
    cap = cv2.VideoCapture(video_file)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vsize = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hsize = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    list_time = []

    frame_cnt = 0
    for frame_cnt in range(n_frame):
        print('.', end='')

        ret, image = cap.read()
        if not ret:
            break
        H, W, C = image.shape
        """
        image = cv2.medianBlur(image, 7)
        edge = cv2.Laplacian(image, cv2.CV_32F) #edge = cv2.Scharr(image, cv2.CV_32F, 0, 1)
        image = image.astype(np.float32)
        image += edge
        """
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
        din = preprocessing_fn(image, input_space='BGR')
        din = torch.from_numpy(din).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Replace NVTX range_push and range_pop with time-based profiling
            start_time = time.time()  # Start time before the model computation
            dout = model(din)
            end_time = time.time()  # End time after the model computation
            step_time = end_time-start_time

        list_time.append(step_time)

        # Print or log the time taken for the operation
        print(f"Frame {frame_cnt} took {end_time - start_time:.4f} seconds")

        # multi-class segmentation
        if N_CLASSES > 1:
            _, label = torch.max(dout, 1)
            scale = 255 // (N_CLASSES-1)
        # single-class segmentation: set a threshold
        else:
            label = (dout > 0.9).int()[0]
            scale = 200
        label = label.permute(1, 2, 0).cpu().numpy()

        image = (label * scale).astype(np.uint8())
        image = cv2.resize(image, (W, H))
        '''
        cv2.imwrite(os.path.join(results, 'label_%04d.png' % frame_cnt), image)
        '''

        color_map = OrderedDict([
            # Roads and infrastructure
            ('road', (128, 64, 128)),  # Purple color for roads
            ('sidewalk', (244, 35, 232)),  # Pink color for sidewalks
            ('parking', (250, 170, 160)),  # Light red color for parking areas
            ('rail track', (230, 150, 140)),  # Light brown color for rail tracks

            # People and vehicles
            ('person', (220, 20, 60)),  # Red color for a person
            ('rider', (255, 0, 0)),  # Bright red color for a rider (e.g., on a bike or motorcycle)

            ('car', (0, 0, 142)),  # Dark blue color for cars
            ('truck', (0, 0, 70)),  # Darker blue color for trucks
            ('bus', (0, 60, 100)),  # Blue-green color for buses
            ('on rails', (0, 80, 100)),  # Bluish color for vehicles on rails
            ('motorcycle', (0, 0, 230)),  # Bright blue color for motorcycles
            ('bicycle', (34, 139, 34)),  # Bright green color for bicycles (updated)
            ('caravan', (0, 0, 90)),  # Dark blue color for caravans
            ('trailer', (0, 0, 110)),  # Dark blue color for trailers

            # Structures and buildings
            ('building', (70, 70, 70)),  # Medium gray color for buildings
            ('wall', (102, 102, 156)),  # Bluish-gray color for walls
            ('fence', (190, 153, 153)),  # Light reddish-gray color for fences
            ('guard rail', (180, 165, 180)),  # Light gray with a slight purple hue for guard rails
            ('bridge', (150, 100, 100)),  # Reddish-brown color for bridges
            ('tunnel', (150, 120, 90)),  # Muted brown color for tunnels

            # Poles and traffic elements
            ('pole', (153, 153, 153)),  # Neutral gray color for poles
            ('pole group', (153, 153, 153)),  # Neutral gray color for a group of poles
            ('traffic sign', (220, 220, 0)),  # Bright yellow color for traffic signs
            ('traffic light', (250, 170, 30)),  # Yellow-orange color for traffic lights

            # Natural elements
            ('vegetation', (107, 142, 35)),  # Green color for vegetation
            ('terrain', (152, 251, 152)),  # Pale green color for terrain

            # Sky and environmental features
            ('sky', (70, 130, 180)),  # Sky-blue color for the sky

            # Ground and object types
            ('ground', (81, 0, 81)),  # Deep purple color for ground
            ('dynamic', (111, 74, 0)),  # Brownish-yellow color for dynamic objects (moving)
            ('static', (0, 0, 0))  # Black color for static objects (non-moving)
        ])

        r = np.zeros_like(label)
        g = np.zeros_like(label)
        b = np.zeros_like(label)
        for k, color in enumerate(color_map.values()):
            r[label==k] = color[0]
            g[label==k] = color[1]
            b[label==k] = color[2]
        image = np.concatenate((b,g,r), axis=2).astype(np.uint8())
        image = cv2.resize(image, (hsize, vsize))
        cv2.imwrite(os.path.join(results, 'color_%04d.png'%frame_cnt), image)
        
    average_time = sum(list_time)/len(list_time)

    print(f"Frame {frame_cnt} took {end_time - start_time:.4f} seconds")
    print("Average time: {:.4f} seconds".format(average_time))

if __name__ == '__main__':

    valid()
