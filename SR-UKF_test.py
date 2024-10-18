import numpy as np
from yolo import YOLOPoseTRT
import cv2
import time
from kalmantorch import SquareRootUnscentedKalmanFilter
import torch
from dynamic_models import CTRVmeasurementFunctionXY, CTRVstateTransitionFunctionV2


dt = 1 / 13.2
n_kpts = 1



R = np.load('R.npy')[:2, :2]
Q = torch.eye(n_kpts*5)
Q[0, 0] = 2
Q[1, 1] = 2
Q[2, 2] = 4
Q[3, 3] = 4
Q[4, 4] = 2


### CTRV  MODEL ###

kalman = SquareRootUnscentedKalmanFilter(dim_x=n_kpts*5, dim_z=n_kpts*2, dt=dt, hx=CTRVmeasurementFunctionXY, fx=CTRVstateTransitionFunctionV2)

kalman.Q = torch.linalg.cholesky(Q)
kalman.R = torch.linalg.cholesky(torch.from_numpy(R).float())
P = torch.eye(n_kpts * 5) * 30
kalman.P = torch.linalg.cholesky(P)



yolo = YOLOPoseTRT('yolo/yolov8m-rat.engine', (640, 384), n_kpts=13)

cap = cv2.VideoCapture('rec_2024_07_09_13_42_01_up.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#recorder = cv2.VideoWriter('out.mp4', fourcc, 14, (1060, 548))

flag = False


def state_to_kpts(x):
    z = np.zeros(n_kpts * 2)
    for i in range(n_kpts):
        z[i * 2 : (i + 1) * 2] = x[i * 5 : i * 5 + 2]
    return z

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
            kalman.x = torch.from_numpy(np.concatenate((z, np.ones((3, ))))).float()
        else:
            kalman.predict()
            #kalman.Q = CTRVcomputeQ(kalman.x, dt, 2, 1)
            x = kalman.update(torch.from_numpy(z).float()).numpy()
            x = state_to_kpts(x)
            x = np.clip(x, 0, frame.shape[1])

            for i in range(n_kpts):
                frame = cv2.circle(frame, (int(z[2*i]), int(z[2*i + 1])), 3, (255, 0, 0), -1)
                try:
                    frame = cv2.circle(frame, (int(x[2*i]), int(x[2*i + 1])), 3, (0, 0, 255), -1)
                except:
                    pass
            
        
    toc = time.time()
    #time.sleep(max(0, dt - (toc - tic)))
    print('FPS:', 1 / (toc - tic))
    cv2.imshow('FRAME', frame)
    #recorder.write(frame)

#recorder.release()