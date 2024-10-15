import torch
import numpy as np
from dynamic_models import constantAccelerationModel
from kalmantorch import LinearKalmanFilter
from tqdm import tqdm
import matplotlib.pyplot as plt



kpts = np.load('ground_truth.npy')[:900]
predictions = np.load('predictions.npy')[:900]
R = torch.from_numpy(np.load('R.npy'))

dt = 1 / 13.17
n_kpts = 13

A, H, Q, _ = constantAccelerationModel(n_kpts, dt, 'xy')

x0 = predictions[0].flatten()
kalman = LinearKalmanFilter(A, H, Q * 32000 , R, torch.from_numpy(np.concatenate((x0, np.zeros_like(x0), np.zeros_like(x0)))))


filtered = [x0, ]
for z in tqdm(predictions[1:], total=predictions.shape[0]):
    z = z.flatten()
    kalman.predict()
    x = kalman.update(torch.from_numpy(z)).numpy()[: 2*n_kpts]
    filtered.append(x)

filtered = np.array(filtered)
kpts = kpts.reshape(filtered.shape)
predictions = predictions.reshape(filtered.shape)

x_nose_f = filtered[:, 0]
x_nose = kpts[:, 0]
x_nose_d = predictions[:, 0]

t = np.linspace(0, x_nose.shape[0] * dt, x_nose.shape[0])

plt.plot(t, x_nose_f - x_nose, t, x_nose_d - x_nose)
plt.show()

pred_rmse = np.sqrt(((x_nose_d - x_nose) ** 2).sum())
print('Predicted RMSE:', pred_rmse)

filtered_rmse = np.sqrt(((x_nose_f - x_nose) ** 2).sum())
print('Predicted RMSE:', filtered_rmse)