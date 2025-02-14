import os
import time
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from config import *
# from config_valid import *
from collections import OrderedDict

results = './results_image_2'
os.makedirs(results, exist_ok=True)



def valid(image_path, num):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return

    N_CLASSES = len(CLASSES)

    # Load segmentation model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=N_CLASSES,
        activation=ACTIVATION,
    )

    if LOAD_BEST_MODEL:
        param = torch.load(BEST_MODEL_NM, map_location=DEVICE)
        model.load_state_dict(param)
    else:
        param = torch.load(LATEST_MODE_NM, map_location=DEVICE)
        model.load_state_dict(param['model_param'])

    model.to(DEVICE)
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image {image_path}")
        return

    H, W, C = image.shape
    image_resized = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
    din = preprocessing_fn(image_resized, input_space='BGR')
    din = torch.from_numpy(din).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        start_time = time.time()
        dout = model(din)
        end_time = time.time()

    print(f"Inference time: {end_time - start_time:.4f} seconds")

    # Process segmentation result
    if N_CLASSES > 1:
        _, label = torch.max(dout, 1)
        scale = 255 // (N_CLASSES - 1)
    else:
        label = (dout > 0.9).int()[0]
        scale = 200

    label = label.permute(1, 2, 0).cpu().numpy()
    result_image = (label * scale).astype(np.uint8)
    result_image = cv2.resize(result_image, (W, H))

    # Save the segmented image
    output_path = os.path.join(results, f"segmented_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, result_image)

    print(f"Segmentation result saved at {output_path}")

    # color map for 12 classes
        # The order of class should be the same as config.py
        # As it is orderedDict
    color_map = OrderedDict([
        ('sky', (190, 255, 255)),  # sky blue ( a little bit of white)
        ('tree', (0, 255, 100)),  # green
        ('signsymbol', (255, 0, 0)),  # red
        ('car', (0, 0, 255)),  # blue
        ('pedestrian', (255, 255, 255)),  # white
        ('bicycle', (119, 11, 32)),  # dark red
        ('unlabeled', (0, 0, 0))  # black
    ])
    # print colored label
    # !!!do not forget to change the saved png name everytime we try a different configuration
    r = np.zeros_like(label)
    g = np.zeros_like(label)
    b = np.zeros_like(label)
    for k, color in enumerate(color_map.values()):
        r[label==k] = color[0]
        g[label==k] = color[1]
        b[label==k] = color[2]
    image = np.concatenate((b,g,r), axis=2).astype(np.uint8())
    image = cv2.resize(image, (W, H))
    
    cv2.imwrite(os.path.join(results, 'bike_car%01d.png'%num), image)




if __name__ == '__main__':
    image_dir = "./image"
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    for num, i in enumerate(image_files):
        valid(i, num)
