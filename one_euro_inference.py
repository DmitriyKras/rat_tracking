import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, dt, x=None):
        """Compute the filtered signal."""
        t_e = dt
        if x is None:
            x = self.x_prev + self.dx_prev * dt
        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return self.x_prev


kpts = np.load('ground_truth.npy')[:900, 0]
predictions = np.load('predictions.npy')[:900, 0]

dt = 1 / 13.17
n_kpts = 13

x0, y0 = predictions[0].flatten()[:2]
x_filter = OneEuroFilter(0, x0, min_cutoff=1, beta=0.5)
y_filter = OneEuroFilter(0, y0, min_cutoff=1, beta=0.5)

n = 2
i = 1

filtered = [[x0, y0]]
for z in tqdm(predictions[1:], total=predictions.shape[0]):
    x, y = z.flatten()[:2]
    i += 1
    if i == n:
        x_f = x_filter(dt, x)
        y_f = y_filter(dt, y)
        i = 0
    else:
        x_f = x_filter(dt, None)
        y_f = y_filter(dt, None)
    filtered.append([x_f, y_f])

filtered = np.array(filtered)

x_nose_f = filtered[:, 0]
x_nose = kpts[:, 0]
x_nose_d = predictions[:, 0]

t = np.linspace(0, x_nose.shape[0] * dt, x_nose.shape[0])

plt.plot(t, x_nose_f - x_nose, t, x_nose_d - x_nose)
plt.show()

pred_rmse = np.sqrt(((x_nose_d - x_nose) ** 2).sum())
print('Predicted RMSE:', pred_rmse)

filtered_rmse = np.sqrt(((x_nose_f - x_nose) ** 2).sum())
print('Filtered RMSE:', filtered_rmse)

plt.plot(filtered[:, 0], filtered[:, 1], 'r-', linewidth=2)
plt.scatter(predictions[::n, 0], predictions[::n, 1], linewidth=0.5)
plt.plot(kpts[:, 0], kpts[:, 1], 'b-', linewidth=1)
plt.legend(('One euro', 'Predicted', 'Real'))
plt.show()