import numpy as np
from yolo import YOLOPoseTRT
import cv2
import time
import torch
from kalmantorch import LinearKalmanFilter
from dynamic_models import *


dt = 1 / 13
n_kpts = 1

A, H, Q, R = constantAccelerationModel(n_kpts, dt, 'xy')
print(A)

#R = torch.from_numpy(np.load('R.npy')).float()

yolo = YOLOPoseTRT('yolo/yolov8m-rat.engine', (640, 384), n_kpts=13)

cap = cv2.VideoCapture('rec_2024_07_09_13_42_01_up.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#recorder = cv2.VideoWriter('out.mp4', fourcc, 13, (1060, 548))

flag = False


j = 0

while cap.isOpened():
    ret, frame = cap.read()


    if not ret or cv2.waitKey(0) == ord('q'):
        break

    tic = time.time()
    boxes = yolo(frame)

    if len(boxes):
        z = boxes[0][:2]

        if not flag:
            flag = True
            kalman = LinearKalmanFilter(A, H, Q * 10, R * 10, torch.from_numpy(np.concatenate((z, np.zeros_like(z), np.zeros_like(z)))))
            kalman.to('cpu')
        else:
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kalman.predict()
            j += 1
            #x = kalman.update(torch.from_numpy(np.concatenate((z, vel.flatten())))).numpy()
            if j == 4:
                x = kalman.update(torch.from_numpy(z)).numpy()
                j = 0
            else:
                x = kalman.x.numpy()
            frame = cv2.putText(frame, f"Vx: {float(x[2]):.2f} Vy: {float(x[3]):.2f} ax: {float(x[4]):.2f} ay: {float(x[5]):.2f}", 
                                (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            x = np.clip(x[: 2], 0, frame.shape[1])
            
            for i in range(n_kpts):
                frame = cv2.circle(frame, (int(z[2*i]), int(z[2*i + 1])), 3, (255, 0, 0), -1)
                frame = cv2.circle(frame, (int(x[2*i]), int(x[2*i + 1])), 3, (0, 0, 255), -1)
            
    toc = time.time()
    #time.sleep(max(0, dt - (toc - tic)))
    print('FPS:', 1 / (toc - tic))
    cv2.imshow('FRAME', frame)
    #recorder.write(frame)

#recorder.release()