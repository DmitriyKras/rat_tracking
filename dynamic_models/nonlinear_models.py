import torch


### CTRV MODEL ###

def CTRVstateTransitionFunction(x, dt):
    # state x must be in shape [-1, n_kpts * 5] where state of each point is [x, y, v, w, dw/dt]
    _, dim_x = x.shape
    x = x.reshape(-1, 5)
    coef = x[:, 2] / x[:, 3]
    vel = x[:, 4] * dt
    x[:, 0] += coef * (torch.sin(x[:, 3] + vel) - torch.sin(x[:, 3]))
    x[:, 1] += coef * (-torch.cos(x[:, 3] + vel) + torch.cos(x[:, 3]))
    x[:, 3] += vel
    return x.reshape(-1, dim_x)


def CTRVcomputeQ(x, dt, sigma_a, sigma_w):
    # state x must be in shape [n_kpts * 5, ] where state of each point is [x, y, v, w, dw/dt]
    x = x.reshape(-1, 5)
    n_kpts = x.shape[0]
    G = torch.zeros((n_kpts, 5, 2))
    G[:, 0, 0] = 0.5 * dt ** 2 * torch.cos(x[:, 3])
    G[:, 1, 0] = 0.5 * dt ** 2 * torch.sin(x[:, 3])
    G[:, 2, 0] = dt
    G[:, 3, 1] = 0.5 * dt ** 2
    G[:, 4, 1] = dt

    E = torch.eye(2)
    E[0, 0] = sigma_a
    E[1, 1] = sigma_w

    Q = torch.bmm(G @ E, torch.transpose(G, 1, 2))
    Q_out = torch.zeros((n_kpts * 5, n_kpts * 5))
    for i in range(13):
        Q_out[i * 5 : (i + 1) * 5, i * 5 : (i + 1) * 5] = Q[i]
    return Q_out



### CTRA MODEL ###

# TODO: implement CTRA

def CTRVmeasurementFunctionXY(x):
    x = x.reshape(-1, 5)
    return x[:, :2]


def CTRVmeasurementFunctionXYV(x):
    x = x.reshape(-1, 5)
    return x[:, :3]
