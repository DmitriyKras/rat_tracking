import numpy as np
from yolo import YOLOPoseTRT
import cv2
import time
from kalmantorch import UnscentedKalmanFilter
import torch
from dynamic_models import CTRVcomputeQ, CTRVmeasurementFunctionXY, CTRVstateTransitionFunction, CTRVstateTransitionFunctionV2


dt = 1 / 13.2
n_kpts = 13



R = np.load('R.npy')
Q = torch.eye(n_kpts*5) * 0.5


### CTRV  MODEL ###

kalman = UnscentedKalmanFilter(dim_x=n_kpts*5, dim_z=n_kpts*2, dt=dt, hx=CTRVmeasurementFunctionXY, fx=CTRVstateTransitionFunctionV2)

kalman.Q = Q
kalman.R = torch.from_numpy(R).float()
kalman.P = torch.eye(n_kpts*5) * 0.5


yolo = YOLOPoseTRT('yolo/yolov8m-rat.engine', (640, 384), n_kpts=13)

cap = cv2.VideoCapture('rec_2024_07_09_13_42_01_up.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#recorder = cv2.VideoWriter('out.mp4', fourcc, 14, (1060, 548))

flag = False

lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              10, 0.03))

old_gray = None

def kpts_vel_to_measurement(z):
    meas = np.zeros(n_kpts * 2)
    for i in range(n_kpts):
        meas[i * 2 : i * 2 + 2] = z[i * 2 : (i + 1) * 2]
        #meas[i * 3 + 2] = vel[i]
    return torch.from_numpy(meas).float()

def state_to_kpts(x):
    z = np.zeros(n_kpts * 2)
    for i in range(n_kpts):
        z[i * 2 : (i + 1) * 2] = x[i * 5 : i * 5 + 2]
    return z

while cap.isOpened():
    ret, frame = cap.read()


    if not ret or cv2.waitKey(1) == ord('q'):
        break

    tic = time.time()
    boxes = yolo(frame)
    if len(boxes):
        z = boxes[0]


        if not flag:
            flag = True
            kalman.x = torch.ones((n_kpts * 5,))
        else:
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kalman.predict()
            #kalman.Q = CTRVcomputeQ(kalman.x, dt, 2, 1)
            x = kalman.update(torch.from_numpy(z).float()).numpy()
            x = state_to_kpts(x)
            x = np.clip(x, 0, frame.shape[1])

            for i in range(13):
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