from evaluation import collectGroundTruth, collectPredictions, computeR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


#collectGroundTruth('/home/techtrans2/RAT_DATASETS/LAB_RAT_DATASET/labels', (1060, 548))

kpts = np.load('ground_truth.npy')

#collectPredictions('/home/techtrans2/RAT_DATASETS/LAB_RAT_DATASET/images', 'yolo/yolov8m-rat.engine')

predictions = np.load('predictions.npy')

R = computeR('ground_truth.npy', 'predictions.npy', 'R.npy')

print(R.shape)

# dt = 1 / 13.17
# x_nose = kpts[:900, 0, 0]
# x_nose_detected = predictions[:900, 0, 0]

# x_nose_vel = (np.diff(x_nose, 1) / dt).var()

# print(x_nose_vel)

# x_nose_acc = (np.diff(x_nose, 2) / dt * 2).var()

# print(x_nose_acc)


# t = np.linspace(0, x_nose.shape[0] * dt, x_nose.shape[0])


# plt.plot(t, x_nose)

# plt.show()




