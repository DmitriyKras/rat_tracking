import numpy as np
import torch
from typing import Tuple


def constantVelocityModel(n_kpts: int, dt: float, measurement_type='xy') -> Tuple[torch.Tensor]:
    A = np.concatenate((np.concatenate((np.eye(n_kpts * 2), np.eye(n_kpts * 2) * dt), axis=1),
    np.concatenate((np.zeros((n_kpts * 2, n_kpts * 2)), np.eye(n_kpts * 2)), axis=1)), axis=0)
    A = torch.from_numpy(A)
    # init process noise covariance matrix based on acceleration
    Q = np.concatenate((np.concatenate((np.eye(n_kpts * 2) * 0.25 * dt ** 4, np.eye(n_kpts * 2) * 0.5 * dt ** 3), axis=1),
    np.concatenate((np.eye(n_kpts * 2) * 0.5 * dt ** 3, np.eye(n_kpts * 2) * dt ** 2), axis=1)), axis=0)
    Q = torch.from_numpy(Q)
    if measurement_type == 'xy':
        # init measurement error covariance matrix
        R = np.eye(n_kpts * 2)
        # init observation transition matrix
        H = np.concatenate((np.eye(n_kpts * 2), np.zeros((n_kpts * 2, n_kpts * 2))), axis=1)
    else:
        # init measurement error covariance matrix (random for now)
        R = np.eye(n_kpts * 4)
        # init observation transition matrix
        H = np.eye(n_kpts * 4)
    R = torch.from_numpy(R)
    H = torch.from_numpy(H)
    return A, H, Q, R


def constantAccelerationModel(n_kpts: int, dt: float, measurement_type='xy') -> Tuple[torch.Tensor]:
    A = np.eye(n_kpts * 6)
    A[: 2*n_kpts, 2*n_kpts : 4*n_kpts] = np.eye(2*n_kpts) * dt
    A[2*n_kpts : 4*n_kpts, 4*n_kpts :] = np.eye(2*n_kpts) * dt
    A[: 2*n_kpts, 4*n_kpts :] = np.eye(2*n_kpts) * 0.5 * dt ** 2
    A = torch.from_numpy(A)
    # init process noise covariance matrix based on acceleration
    Q11 = np.eye(n_kpts * 2) * 0.25 * dt ** 4
    Q12_21 = np.eye(n_kpts * 2) * 0.5 * dt ** 3
    Q13_31 = np.eye(n_kpts * 2) * 0.5 * dt ** 2
    Q22 = Q13_31 * 2
    Q33 = np.eye(n_kpts * 2)
    Q23_32 = np.eye(n_kpts * 2) * dt
    Q = np.concatenate((np.concatenate((Q11, Q12_21, Q13_31), axis=1),
    np.concatenate((Q12_21, Q22, Q23_32), axis=1),
    np.concatenate((Q13_31, Q23_32, Q33), axis=1)), axis=0)
    Q = torch.from_numpy(Q)
    if measurement_type == 'xy':
        # init measurement error covariance matrix
        R = np.eye(n_kpts * 2)
        # init observation transition matrix
        H = np.concatenate((np.eye(n_kpts * 2), np.zeros((n_kpts * 2, n_kpts * 4))), axis=1)
    else:
        # init measurement error covariance matrix (random for now)
        R = np.eye(n_kpts * 4)
        # init observation transition matrix
        H = H = np.concatenate((np.eye(n_kpts * 4), np.zeros((n_kpts * 4, n_kpts * 2))), axis=1)
    R = torch.from_numpy(R)
    H = torch.from_numpy(H)
    return A, H, Q, R
