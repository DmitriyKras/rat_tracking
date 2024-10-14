import numpy as np
from yolo import YOLOPoseTRT
import cv2
import time
from kalmantorch import UnscentedKalmanFilter
import torch
from dynamic_models import CTRVcomputeQ, CTRVmeasurementFunctionXY, CTRVmeasurementFunctionXYV, CTRVstateTransitionFunction


dt = 1 / 13.2
n_kpts = 17



R = np.eye(n_kpts * 2)
Q = torch.eye(n_kpts*5) * 2


### CTRV  MODEL ###

kalman = UnscentedKalmanFilter(dim_x=n_kpts*5, dim_z=n_kpts*2, dt=dt, hx=CTRVmeasurementFunctionXY, fx=CTRVstateTransitionFunction)

kalman.Q = Q
kalman.R = torch.from_numpy(R)
kalman.P = torch.eye(n_kpts*5) * 20


yolo = YOLOPoseTRT('yolo/yolov8n-pose.engine', (640, 384))

cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#recorder = cv2.VideoWriter('out.mp4', fourcc, 14, (1060, 548))

flag = False

lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              10, 0.03))

old_gray = None

def kpts_vel_to_measurement(z, vel):
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

        if old_gray is None:
            old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old_z = z.reshape(-1, 1, 2).astype(np.float32)

        if not flag:
            flag = True
            kalman.x = torch.ones((n_kpts * 5,))
        else:
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, old_z, None, **lk_params)

            vel = (p1 - old_z) / dt
            vel = vel.squeeze() ** 2
            vel = np.sqrt(vel.sum(axis=1))
            kalman.predict()
            kalman.Q = CTRVcomputeQ(kalman.x, dt, 2, 1)
            x = kalman.update(kpts_vel_to_measurement(z, vel)).numpy()
            x = state_to_kpts(x)
            x = np.clip(x, 0, frame.shape[1])

            for i in range(13):
                frame = cv2.circle(frame, (int(z[2*i]), int(z[2*i + 1])), 3, (255, 0, 0), -1)
                try:
                    frame = cv2.circle(frame, (int(x[2*i]), int(x[2*i + 1])), 3, (0, 0, 255), -1)
                except:
                    pass
            old_frame = cur_frame
            old_z = z.reshape(-1, 1, 2).astype(np.float32)
            
        
    toc = time.time()
    #time.sleep(max(0, dt - (toc - tic)))
    print('FPS:', 1 / (toc - tic))
    cv2.imshow('FRAME', frame)
    #recorder.write(frame)

#recorder.release()