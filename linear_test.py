import numpy as np
from yolo import YOLOPoseTRT
import cv2
import time
import torch
from kalmantorch import LinearKalmanFilter
from dynamic_models import *


dt = 1 / 20
n_kpts = 17

A, H, Q, R = constantAccelerationModel(n_kpts, dt, 'xyVxVy')

yolo = YOLOPoseTRT('yolo/yolov8n-pose.engine', (640, 384))

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#recorder = cv2.VideoWriter('out.mp4', fourcc, 5, (640, 480))

flag = False

lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              10, 0.03))

old_gray = None

while cap.isOpened():
    ret, frame = cap.read()


    if not ret or cv2.waitKey(1) == ord('q'):
        break

    tic = time.time()
    boxes = yolo(frame)

    if len(boxes):
        z = boxes[0]

        if old_gray is None:
            old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old_z = z.reshape(-1, 1, 2).astype(np.float32)

        if not flag:
            flag = True
            kalman = LinearKalmanFilter(A, H, Q, R, torch.from_numpy(np.concatenate((z, np.zeros_like(z), np.zeros_like(z)))))
            kalman.to('cpu')
        else:
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, old_z, None, **lk_params)

            vel = (p1 - old_z) / dt
            kalman.predict()
            x = kalman.update(torch.from_numpy(np.concatenate((z, vel.flatten())))).numpy()
            x = np.clip(x[: 34], 0, frame.shape[1])
            for i in range(17):
                frame = cv2.circle(frame, (int(z[2*i]), int(z[2*i + 1])), 2, (255, 0, 0), -1)
                frame = cv2.circle(frame, (int(x[2*i]), int(x[2*i + 1])), 2, (0, 0, 255), -1)
            old_frame = cur_frame
            old_z = z.reshape(-1, 1, 2).astype(np.float32)
            
        
    toc = time.time()
    time.sleep(max(0, dt - (toc - tic)))
    print('FPS:', 1 / (toc - tic))
    cv2.imshow('FRAME', frame)
    #recorder.write(frame)

#recorder.release()