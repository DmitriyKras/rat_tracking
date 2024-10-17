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

Q = torch.eye(n_kpts*5) * 10

### CTRV  MODEL ###

kalman = UnscentedKalmanFilter(dim_x=n_kpts*5, dim_z=n_kpts*2, dt=dt, hx=CTRVmeasurementFunctionXY, fx=CTRVstateTransitionFunction,
                               residual_x=stateSubtractCTRV, state_add=stateAddCTRV)

kalman.Q = Q
kalman.R = R
kalman.P = torch.eye(n_kpts*5) * 50
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

i = 0

z1 = x0

n = 4


# for j in range(int(predictions.shape[0] / n) - 1):
#     dif = (predictions[(j + 1) * n] - predictions[j * n]) / n
#     for k in range(n):
#         predictions[j * n + k] = predictions[j * n] + dif * k


for j, z in tqdm(enumerate(predictions[1:]), total=predictions.shape[0]):
    z = z.flatten()
    kalman.predict()
    i += 1
    #kalman.Q = CTRVcomputeQ(kalman.x, dt, 20, 10)
    if i == n:
        x = kalman.update(kpts_vel_to_measurement(z)).numpy()
        i = 0
    else:
        x = kalman.x.numpy()
    #x = kalman.update(kpts_vel_to_measurement(z)).numpy()
    x = state_to_kpts(x)
    filtered.append(x)

filtered = np.array(filtered).reshape(predictions.shape)

x_nose_f = filtered[:, 0, 0]
#x_nose = kpts[:, 0, 0]
x_nose_d = predictions[:, 0, 0]

y_nose_f = filtered[:, 0, 1]
y_nose_d = predictions[:, 0, 1]

t = np.linspace(0, x_nose_d.shape[0] * dt, x_nose_d.shape[0])

#plt.plot(t, x_nose_f - x_nose, 'b-')
#plt.plot(t, x_nose_d - x_nose, 'r-')
plt.plot(x_nose_f, y_nose_f, 'r-', linewidth=2)
plt.plot(kpts[:, 0, 0], kpts[:, 0, 1], 'b-', linewidth=1)
plt.scatter(x_nose_d[::n], y_nose_d[::n], linewidth=1)
plt.xlabel('time, s')
plt.ylabel('error, pix')

#plt.plot(t, x_nose_d)

pred_rmse = np.sqrt(((kpts - predictions) ** 2).sum(axis=2)).mean()
print('Predicted RMSE:', pred_rmse)

filtered_rmse = np.sqrt(((kpts - filtered) ** 2).sum(axis=2)).mean()
print('Filtered RMSE:', filtered_rmse)

plt.legend(("UKF", "Real", 'Real'))

plt.show()