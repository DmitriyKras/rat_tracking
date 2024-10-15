import numpy as np
import os
from tqdm import tqdm
from yolo import YOLOPoseTRT
import cv2


def collectGroundTruth(labels_dir: str, imgsz: tuple, out_array='ground_truth.npy') -> None:
    labels = sorted(os.listdir(labels_dir))
    print(labels)
    kpts = []
    for label in tqdm(labels, total=len(labels)):
        pts = []
        with open(f"{labels_dir}/{label}", 'r') as f:
            label = f.readline().strip('\n').split(' ')
        label = np.array(label).astype(float)[5:]
        for i in range(13):
            pts.append([label[i * 3], label[i * 3 + 1]])
        kpts.append(pts)
    kpts = np.array(kpts) * np.array(imgsz)
    np.save(out_array, kpts)


def collectPredictions(images_dir: str, model_engine: str, out_array='predictions.npy') -> None:
    images = sorted(os.listdir(images_dir))
    print(images)
    yolo = YOLOPoseTRT(model_engine, (640, 384), n_kpts=13)
    kpts = []
    for img in tqdm(images, total=len(images)):
        img = cv2.imread(f"{images_dir}/{img}")
        label = yolo(img)[0]
        kpts.append(label.reshape(13, 2))
    kpts = np.array(kpts)
    np.save(out_array, kpts)


def computeR(gt: str, preds: str, out=None) -> np.ndarray:
    gt = np.load(gt)
    preds = np.load(preds)
    error = np.sqrt(((preds - gt) ** 2).sum(axis=2))
    R = np.diag(error.var(axis=0))
    if type(out) == str:
        np.save(out, R)
    return R


def computeRMSE(gt: str, preds: str) -> float:
    gt = np.load(gt)
    preds = np.load(preds)
    error = np.sqrt(((preds - gt) ** 2).sum(axis=2)).mean()
    return error
