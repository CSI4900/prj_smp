#*********************************************************************************************
# FILE   NAME:    valid.py
# PROJ   NAME:    Segmentation
# DESCRIPTION:    validation of SMP (Segmentation Models Pytorch)
#
# HOW TO USE:     download CamVid data to
#               git clone https://github.com/alexgkendall/SegNet-Tutorial ./data
#
#
# REVISION HISTORY
# YYYY/MMM/DD     Author       Comments
# 2024 MAR 01     Yu Liu       creation
#
#*********************************************************************************************
import os, cv2, torch, numpy as np
import segmentation_models_pytorch as smp
from config import *

results='./results'
os.makedirs(results, exist_ok=True)

def valid():
    N_CLASSES = len(CLASSES)
    # create segmentation model with pretrained encoder
    model = smp.Unet(#smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=N_CLASSES, 
        activation=ACTIVATION,
    )

    if LOAD_BEST_MODEL:
        param = torch.load(LATEST_MODE_NM)
        model.load_state_dict(param)
    else:
        param = torch.load(LATEST_MODE_NM)
        model.load_state_dict(param['model_param'])


    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # start inference
    video_file = '../../RacingCars/racing_cars.sd.mp4'
    cap = cv2.VideoCapture(video_file)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vsize = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hsize = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_cnt = 0
    for frame_cnt in range(n_frame):
        print('.', end='')

        ret, image = cap.read()
        if not ret:
            break
        H,W,C = image.shape
        """
        image = cv2.medianBlur(image, 7)
        edge = cv2.Laplacian(image, cv2.CV_32F) #edge = cv2.Scharr(image, cv2.CV_32F, 0, 1)
        image = image.astype(np.float32)
        image += edge
        """
        image = cv2.resize(image, (384,384), interpolation = cv2.INTER_CUBIC)
        din = preprocessing_fn(image, input_space='BGR')
        din = torch.from_numpy(din).float().permute(2,0,1).unsqueeze(0)

        with torch.no_grad():
            torch.cuda.nvtx.range_push('frame%03d'%frame_cnt)
            dout = model(din)
            torch.cuda.nvtx.range_pop()

        # multi-class segmentation
        if N_CLASSES >1:
            _, label = torch.max(dout, 1)
            scale = 255 // (N_CLASSES-1)
        # single-class segmentation: set a threshold
        else:
            label = (dout > 0.9).int()[0]
            scale = 200
        label = label.permute(1,2,0).cpu().numpy()

        image = (label * scale).astype(np.uint8())
        image = cv2.resize(image, (W, H))
        cv2.imwrite(os.path.join(results, 'label_%04d.png'%frame_cnt), image)
        
        """
        r = np.zeros_like(label)
        g = np.zeros_like(label)
        b = np.zeros_like(label)
        for k, color in enumerate(color_map.values()):
            r[label==k] = color[0]
            g[label==k] = color[1]
            b[label==k] = color[2]
        image = np.concatenate((b,g,r), axis=2).astype(np.uint8())
        image = cv2.resize(image, (hsize, vsize))
        cv2.imwrite(os.path.join(args.results, 'color_%04d.png'%frame_cnt), image)
        """

if __name__=='__main__':

    valid()
