import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm


def collectGroundTruth(labels_dir: str, imgsz: tuple, out_array='ground_truth.npy') -> None:
    labels = natsorted(os.listdir(labels_dir))
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


