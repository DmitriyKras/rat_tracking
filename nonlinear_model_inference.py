import torch
import numpy as np
from dynamic_models import CTRVcomputeQ, CTRVmeasurementFunctionXY, CTRVstateTransitionFunction, stateAddCTRV, stateSubtractCTRV
from kalmantorch import UnscentedKalmanFilter
from tqdm import tqdm
import matplotlib.pyplot as plt



kpts = np.load('ground_truth.npy')[:900]
predictions = np.load('predictions.npy')[:900]
R = torch.from_numpy(np.load('R.npy')).float()

dt = 1 / 13.17
n_kpts = 13


x0 = predictions[0].flatten()

Q = torch.eye(n_kpts*5) * 5

### CTRV  MODEL ###

kalman = UnscentedKalmanFilter(dim_x=n_kpts*5, dim_z=n_kpts*2, dt=dt, hx=CTRVmeasurementFunctionXY, fx=CTRVstateTransitionFunction)

kalman.Q = Q
kalman.R = R
kalman.P = torch.eye(n_kpts*5) * 30
kalman.x = torch.from_numpy(np.concatenate((x0.reshape(-1, 2), np.ones((n_kpts, 3))), axis=1).flatten()).float()


def kpts_vel_to_measurement(z):
    meas = np.zeros(n_kpts * 2)
    for i in range(n_kpts):
        meas[i * 2 : i * 2 + 2] = z[i * 2 : (i + 1) * 2]
    return torch.from_numpy(meas).float()

def state_to_kpts(x):
    z = np.zeros(n_kpts * 2)
    for i in range(n_kpts):
        z[i * 2 : (i + 1) * 2] = x[i * 5 : i * 5 + 2]
    return z


filtered = [x0, ]
for z in tqdm(predictions[1:], total=predictions.shape[0]):
    z = z.flatten()
    kalman.predict()
    kalman.Q = CTRVcomputeQ(kalman.x, dt, 20, 10)
    x = kalman.update(kpts_vel_to_measurement(z)).numpy()
    x = state_to_kpts(x)
    filtered.append(x)

filtered = np.array(filtered).reshape(predictions.shape)

x_nose_f = filtered[:, 0, 0]
x_nose = kpts[:, 0, 0]
x_nose_d = predictions[:, 0, 0]

t = np.linspace(0, x_nose.shape[0] * dt, x_nose.shape[0])

plt.plot(t, x_nose_f - x_nose, 'b-')
plt.plot(t, x_nose_d - x_nose, 'r-')
plt.xlabel('time, s')
plt.ylabel('error, pix')


pred_rmse = np.sqrt(((kpts - predictions) ** 2).sum(axis=2)).mean()
print('Predicted RMSE:', pred_rmse)

filtered_rmse = np.sqrt(((kpts - filtered) ** 2).sum(axis=2)).mean()
print('Filtered RMSE:', filtered_rmse)

plt.legend((f"Prediction error. RMSE: {pred_rmse}", f"Filter error. RMSE: {filtered_rmse}"))
plt.show()