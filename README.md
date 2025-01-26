This is project of using pypi libarary segmentation_models_pytorch, briefly called SMP, to segment a target object such as a car or bycical.
SMP includes multiple well-known models. The default model used in this project is resent34 and related weights is imagenet
We only trained single class in this project as an example reference.

train.py: used to train SMP model under configuration config.py where training dataset called CamVid
valid.py: used to validate the trained model with a video cliip called racing_car.sd.mp4