"""
CAR CONFIG

This file is read by your car application's manage.py script to change the car
performance.

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#VEHICLE
DRIVE_LOOP_HZ = 20
MAX_LOOPS = 100000

#CAMERA
CAMERA_RESOLUTION = (120, 160) #(height, width)
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

#STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 420
STEERING_RIGHT_PWM = 360

#THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 400
THROTTLE_STOPPED_PWM = 360
THROTTLE_REVERSE_PWM = 310

#TRAINING
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8
# Training Weights (must total 1.0)
ANGLE_WEIGHT = 0.5
THROTTLE_WEIGHT = (1.0 - ANGLE_WEIGHT)
# Region of Interest cropping (pixels to crop from images)
ROI_CROP_TOP = 0
ROI_CROP_BOTTOM = 0

#JOYSTICK
USE_JOYSTICK_AS_DEFAULT = False
JOYSTICK_TYPE = "PS3"
JOYSTICK_MAX_THROTTLE = 0.5
JOYSTICK_STEERING_SCALE = 1.0
AUTO_RECORD_ON_THROTTLE = False

# TUB Control
USE_MULTI_TUBS = False
TUB_PATH = os.path.join(CAR_PATH, 'tub') # if using a single tub

#ROPE.DONKEYCAR.COM
ROPE_TOKEN="GET A TOKEN AT ROPE.DONKEYCAR.COM"
