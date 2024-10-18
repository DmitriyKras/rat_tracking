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
    I = torch.diag(torch.Tensor((1, 1, 2, 2, 0.25)))
    Q += I

    Q_out = torch.zeros((n_kpts * 5, n_kpts * 5))
    for i in range(n_kpts):
        Q_out[i * 5 : (i + 1) * 5, i * 5 : (i + 1) * 5] = Q[i]
    return Q_out


def stateSubtractCTRV(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dif = x.reshape(-1, 5) - y.reshape(-1, 5)
    w = dif[:, 3]
    w = torch.where(w > torch.pi, w - 2 * torch.pi, w)
    w = torch.where(w < -torch.pi, w + 2 * torch.pi, w)
    dif[:, 3] = w
    return dif.flatten()


def stateAddCTRV(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dif = x.reshape(-1, 5) + y.reshape(-1, 5)
    w = dif[:, 3] % (2 * torch.pi)
    #w = torch.where(w > torch.pi, w - 2 * torch.pi, w)
    dif[:, 3] = w
    return dif.flatten()


def stateMeanCTRV(sigmas: torch.Tensor, Wm: torch.Tensor) -> torch.Tensor:
    #print(sigmas.shape)
    #print(Wm.shape)
    n_sigmas, x_dim = sigmas.shape
    mean = (sigmas.T @ Wm[:, None]).reshape(-1, 5)
    sigmas = sigmas.T.reshape(-1, 5, n_sigmas)
    phi = sigmas[:, 3, :]
    sin = (torch.sin(phi) @ Wm[:, None]).squeeze()
    cos = (torch.cos(phi) @ Wm[:, None]).squeeze()
    mean[:, 3] = torch.atan2(sin, cos)
    return mean.flatten()


### CTRV MODEL V2.0###

def CTRVstateTransitionFunctionV2(x, dt):
    # state x must be in shape [-1, n_kpts * 5] where state of each point is [x, y, v, w, dw/dt]
    _, dim_x = x.shape
    x = x.reshape(-1, 5)
    dw = x[:, 4]
    phi = dw * dt
    mask = torch.abs(dw) < 1e-6
    cos = torch.cos(phi)
    sin = torch.sin(phi)
    x[:, 0] += torch.where(mask, x[:, 2] * dt, (sin * x[:, 2] - (1 - cos) * x[:, 3]) / dw)
    x[:, 1] += torch.where(mask, x[:, 3] * dt, (sin * x[:, 3] + (1 - cos) * x[:, 2]) / dw)
    x[:, 2] += cos * x[:, 2] - sin * x[:, 3]
    x[:, 3] += sin * x[:, 2] + cos * x[:, 3]
    return x.reshape(-1, dim_x)


### CTRA MODEL ###

# TODO: implement CTRA

def CTRVmeasurementFunctionXY(x):
    x = x.reshape(-1, 5)
    return x[:, :2]


def CTRVmeasurementFunctionXYV(x):
    x = x.reshape(-1, 5)
    return x[:, :3]
