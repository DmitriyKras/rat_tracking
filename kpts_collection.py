from evaluation import collectGroundTruth, collectPredictions, computeR
import numpy as np


#collectGroundTruth('/home/techtrans2/RAT_DATASETS/LAB_RAT_DATASET/labels', (1060, 548))

#kpts = np.load('ground_truth.npy')

#collectPredictions('/home/techtrans2/RAT_DATASETS/LAB_RAT_DATASET/images', 'yolo/yolov8m-rat.engine')

#predictions = np.load('predictions.npy')

R = computeR('ground_truth.npy', 'predictions.npy')

print(R.shape)

