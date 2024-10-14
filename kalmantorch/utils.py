import torch


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
    
    def sigma_points(self, x: torch.Tensor, P):
        if self.n != x.size()[0]:
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, x.size()))
        
        n = self.n

        lambda_ = self.alpha**2 * (n + self.kappa) - n

        U = torch.linalg.cholesky((lambda_ + n) * P).mH

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
