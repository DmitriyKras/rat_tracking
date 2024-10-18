import torch
import numpy as np
from scipy.linalg import cholesky
from fannypack.utils import cholupdate


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3



class MerweScaledSigmaPoints:
    def __init__(self, n, alpha, beta, kappa, subtract=None):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        if subtract is None:
            self.subtract = torch.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self):
        return 2 * self.n + 1
    
    def sigma_points(self, x: torch.Tensor, P: torch.Tensor):
        if self.n != x.size()[0]:
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, x.size()))
        
        n = self.n

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        #print(P.max(), P.min())
        #P = P.numpy()
        #P = nearestPD(P)
        #U = torch.from_numpy(cholesky(nearestPD((lambda_ + n) * P))).float()
        U = torch.linalg.cholesky((lambda_ + n) * P).mH

        sigmas = torch.zeros((2*n+1, n)).to(x.device)
        sigmas[0] = x

        for k in range(n):
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas
    
    def sigma_points_square_root(self, x: torch.Tensor, S: torch.Tensor):
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = np.sqrt(n + lambda_) * S
        sigmas = torch.zeros((2*n+1, n)).to(x.device)
        sigmas[0] = x

        for k in range(n):
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas

    def _compute_weights(self):
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = torch.full((2*n + 1, ), c)
        self.Wm = torch.full((2*n + 1, ), c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)


def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                        mean_fn=None, residual_fn=None):
    
    kmax, n = sigmas.size()
    
    if mean_fn is None:
        x = torch.mm(Wm[None, :], sigmas).squeeze()
    else:
        x = mean_fn(sigmas, Wm)

    if residual_fn is None:
        y = sigmas - x[None, :]
        P = torch.mm(torch.transpose(y, 0, 1), torch.mm(torch.diag(Wc), y))
    else:
        P = torch.zeros((n, n)).to(Wm.device)
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * torch.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return x, P


def unscented_transform_square_root(sigmas: torch.Tensor, Wm: torch.Tensor, 
                                    Wc: torch.Tensor, noise_cov: torch.Tensor,
                                    mean_fn=None, residual_fn=None):
    kmax, n = sigmas.size()
    if mean_fn is None:
        x = torch.sum(Wm[:, None] * sigmas.float(), dim=0)
    else:
        x = mean_fn(sigmas, Wm)

    if residual_fn is None:
        y = sigmas - x[None, :]
    else:
        y = residual_fn(sigmas, x[None, :])
    
    concatenated = torch.cat(
            [
                y[1:, :] * torch.sqrt(Wc[1]),
                noise_cov.transpose(-1, -2),
            ],
            dim=0,
        )
    
    _, R = torch.linalg.qr(concatenated, mode="complete")

    L = R[: n, :].transpose(-1, -2)

    P = cholupdate(
            L=L,
            x=y[0, :],
            weight=Wc[0],
        )

    return x, P