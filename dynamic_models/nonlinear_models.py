import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2


### CTRV MODEL ###

def CTRVstateTransitionFunction(x, dt):
    # state x must be in shape [-1, n_kpts * 5] where state of each point is [x, y, v, w, dw/dt]
    _, dim_x = x.shape
    x = x.reshape(-1, 5)
    dw = x[:, 4]
    w = x[:, 3]
    mask = torch.abs(dw) < 1e-3
    coef = torch.where(mask, x[:, 2], x[:, 2] / dw)
    vel = dw * dt
    x[:, 0] += coef * torch.where(mask, torch.cos(w), torch.sin(w + vel) - torch.sin(w))
    x[:, 1] += coef * torch.where(mask, torch.sin(w), -torch.cos(w + vel) + torch.cos(w))
    x[:, 3] += vel
    return x.reshape(-1, dim_x)


def CTRVcomputeQ(x, dt, sigma_a, sigma_w):
    # state x must be in shape [n_kpts * 5, ] where state of each point is [x, y, v, w, dw/dt]
    x = x.reshape(-1, 5)
    n_kpts = x.shape[0]
    # G = torch.zeros((n_kpts, 5, 2))
    # G[:, 0, 0] = 0.5 * dt ** 2 * torch.cos(x[:, 3])
    # G[:, 1, 0] = 0.5 * dt ** 2 * torch.sin(x[:, 3])
    # G[:, 2, 0] = dt
    # G[:, 3, 1] = 0.5 * dt ** 2
    # G[:, 4, 1] = dt

    # E = torch.eye(2)
    # E[0, 0] = sigma_a
    # E[1, 1] = sigma_w

    # Q = torch.bmm(G @ E, torch.transpose(G, 1, 2))

    # custom Q

    Q = torch.zeros((n_kpts, 5, 5))
    I = torch.diag(torch.Tensor((10, 10, 20, 3.14 / 10, 2)))
    Q += I

    Q_out = torch.zeros((n_kpts * 5, n_kpts * 5))
    for i in range(13):
        Q_out[i * 5 : (i + 1) * 5, i * 5 : (i + 1) * 5] = Q[i]
    return Q_out


def stateMeanCTRV(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pass


def stateSubtractCTRV(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dif = x.reshape(-1, 5) - y.reshape(-1, 5)
    #w = dif[:, 3] % (2 * torch.pi)
    #w = torch.where(w > torch.pi, w - 2 * torch.pi, w)
    dif[:, 3] = (dif[:, 3] + torch.pi) % (torch.pi * 2) - torch.pi
    return dif.flatten()


def stateAddCTRV(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dif = x.reshape(-1, 5) + y.reshape(-1, 5)
    w = dif[:, 3] % (2 * torch.pi)
    #w = torch.where(w > torch.pi, w - 2 * torch.pi, w)
    dif[:, 3] = w
    return dif.flatten()


### CTRA MODEL ###

# TODO: implement CTRA

def CTRVmeasurementFunctionXY(x):
    x = x.reshape(-1, 5)
    return x[:, :2]


def CTRVmeasurementFunctionXYV(x):
    x = x.reshape(-1, 5)
    return x[:, :3]
