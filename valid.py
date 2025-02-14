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
from collections import OrderedDict

import segmentation_models_pytorch as smp
from config import *

from collections import OrderedDict
results = './results_3'
os.makedirs(results, exist_ok=True)

def valid():
    N_CLASSES = len(CLASSES)
    # create segmentation model with pretrained encoder
    # Adjust the following settings in the config.py
    model = smp.Unet(#smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=N_CLASSES, 
        activation=ACTIVATION,
    )

    # !!!Please remember to change the saved params name everytime we try a different configuration
    if LOAD_BEST_MODEL:
        parama = torch.load('./params/best.pth')
        model.load_state_dict(parama)
    else:
        param = torch.load('./params/latest.pt')
        model.load_state_dict(param['model_param'])


    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # start inference
    video_file = 'racing_cars.sd.mp4'
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
        H, W, C = image.shape
        
        image = cv2.medianBlur(image, 7)
        edge = cv2.Laplacian(image, cv2.CV_32F) #edge = cv2.Scharr(image, cv2.CV_32F, 0, 1)
        image = image.astype(np.float32)
        image += edge
        
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
        din = preprocessing_fn(image, input_space='BGR')
        din = torch.from_numpy(din).float().permute(2,0,1).unsqueeze(0)

        with torch.no_grad():
            torch.cuda.nvtx.range_push('frame%03d'%frame_cnt)
            dout = model(din)
            torch.cuda.nvtx.range_pop()

        # multi-class segmentation
        if N_CLASSES >1:
            _, label = torch.max(dout, 1)

            # The below method kthvalue 2nd argument '11' stands for
            # show the 11th smallest likelihood (that is 2nd biggest likelihood for a class of 12)
            # number of classes can be configured at config.py and camvid.py)
            # The first and last argument doesn't need to be adjusted

            #_, label_second = torch.kthvalue(dout,11,1)

            # The below method topk 2nd argument '2' stands for
            # record the first 2 biggest likelihood and store it as a tensor
            # The first and last argument doesn't need to be adjusted

            #_,lbt = torch.topk(dout,2,1)

            scale = 255 // (N_CLASSES-1)
        # single-class segmentation: set a threshold
        else:
            label = (dout > 0.9).int()[0]
            scale = 200
        label = label.permute(1,2,0).cpu().numpy()

        # !!!Please do not forget to change the saved png name when we try a different configuration
        image = (label * scale).astype(np.uint8())
        image = cv2.resize(image, (W, H))
        cv2.imwrite(os.path.join(results, 'U_twice_seed_1.4_multiclass_label_%04d.png'%frame_cnt), image)

        # color map for 12 classes
        # The order of class should be the same as config.py
        # As it is orderedDict
        results_colored = results+'colored'
        os.makedirs(results, exist_ok=True)

        color_map = OrderedDict([
        ('sky', (190, 255, 255)),  # sky blue ( a little bit of white)
        ('tree', (0, 255, 100)),  # green
        ('signsymbol', (255, 0, 0)),  # red
        ('car', (0, 0, 255)),  # blue
        ('pedestrian', (255, 255, 255)),  # white
        ('bicycle', (119, 11, 32)),  # dark red
        ('unlabeled', (0, 0, 0))  # black
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
        


        # print the 2nd likelihood using the topk method
        # Please see the "test.py" to see the reason why and example for these methods
        # !!!do not forget to change the saved png name everytime we try a different configuration
        """
        # Unlike kthvalue method, the topk return a tensor with mulitiple value (which contains first few biggest likelihood)
        # For example: for a single pixel's first 2 likelihood: tensor([[0.XXX , 0.YYY]])
        # This way can also be used to show the 2nd/3rd.... likelihood
        # We need a "index_select" method to select the 2nd likelihood
        # We need do an extra transpose and unsqueeze to get the form we need to do a color labelling
        # Please see the "test.py" to see the reason why and example for these methods
        
        lb_tk = torch.index_select(lbt,1, torch.tensor([1]))
        lb_tk = torch.transpose(lb_tk,1,0).squeeze(0)
        lb_tk = lb_tk.permute(1, 2, 0).cpu().numpy()
        image_tk = (lb_tk * scale).astype(np.uint8())
        image_tk = cv2.resize(image_tk, (W, H))

        r = np.zeros_like(lb_tk)
        g = np.zeros_like(lb_tk)
        b = np.zeros_like(lb_tk)
        for k, color in enumerate(color_map.values()):
            r[lb_tk == k] = color[0]
            g[lb_tk == k] = color[1]
            b[lb_tk == k] = color[2]
        image_tk = np.concatenate((b, g, r), axis=2).astype(np.uint8())
        image_tk = cv2.resize(image_tk, (hsize, vsize))
        cv2.imwrite(os.path.join(results, 'U_twice_seed_1.4_multiclass_colortk_%04d.png' % frame_cnt), image_tk)
        """

        # The following is still under construction as its efficiency is really low
        # It is supposed to label a object to a car if its 1st OR 2nd likelihood is a car
        # Car (label tensor id:8)
        # !!!do not forget to change the saved png name everytime we try a different configuration

        # This for loop is slow
        """
        for i in range(label_fix[0].size()[0]):
            for j in range(label_fix[0].size()[1]):
                if label_second_fix[0][i][j] == 8:
                    #print("fix")
                    label_fix[0][i][j] = 8
        """

        """
        label_fix = label_fix.permute(1, 2, 0).cpu().numpy()
        image_fix = (label_fix * scale).astype(np.uint8())
        image = cv2.resize(image, (W, H))

        r = np.zeros_like(label_fix)
        g = np.zeros_like(label_fix)
        b = np.zeros_like(label_fix)
        for k, color in enumerate(color_map.values()):
            r[label_fix == k] = color[0]
            g[label_fix == k] = color[1]
            b[label_fix == k] = color[2]
        image = np.concatenate((b, g, r), axis=2).astype(np.uint8())
        image = cv2.resize(image, (hsize, vsize))
        cv2.imwrite(os.path.join(results, 'U_twice_seed_1.4_multiclass_color_fix_%04d.png' % frame_cnt), image)
        """

if __name__=='__main__':

    valid()