import torch
from kalmantorch import unscented_transform_square_root, MerweScaledSigmaPoints
from copy import deepcopy
from fannypack.utils import cholupdate


class SquareRootUnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, dt, hx, fx,
                 x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None,
                 state_add=None):
        
        self.x = torch.zeros((dim_x, ))
        self.P = torch.eye(dim_x)
        self.x_prior = torch.clone(self.x)
        self.P_prior = torch.clone(self.P)
        self.Q = torch.eye(dim_x)
        self.R = torch.eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = MerweScaledSigmaPoints(dim_x, 0.5, 2, dim_x - 3, subtract=residual_x)
        self._dt = dt
        self._num_sigmas = self.points_fn.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = self.points_fn.Wm, self.points_fn.Wc

        if residual_x is None:
            self.residual_x = torch.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = torch.subtract
        else:
            self.residual_z = residual_z

        if state_add is None:
            self.state_add = torch.add
        else:
            self.state_add = state_add

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = torch.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = torch.zeros((self._num_sigmas, self._dim_z))

        self.K = torch.zeros((dim_x, dim_z))    # Kalman gain
        self.y = torch.zeros((dim_z, ))           # residual
        self.z = torch.zeros((dim_z, 1))  # measurement
        self.S = torch.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = torch.zeros((dim_z, dim_z))   # inverse system uncertainty

        # these will always be a copy of x,P after predict() is called
        self.x_prior = torch.clone(self.x)
        self.P_prior = torch.clone(self.P)

        # these will always be a copy of x,P after update() is called
        self.x_post = torch.clone(self.x)
        self.P_post = torch.clone(self.P)

    def to(self, device='cuda'):
        self.K = self.K.to(device)
        self.y = self.y.to(device)
        self.z = self.z.to(device)
        self.S = self.S.to(device)
        self.SI = self.SI.to(device)
        self.x = self.x.to(device)
        self.P = self.P.to(device)
        self.Q = self.Q.to(device)
        self.R = self.R.to(device)
        self.sigmas_f = self.sigmas_f.to(device)
        self.sigmas_h = self.sigmas_h.to(device)
        self.Wm = self.Wm.to(device)
        self.Wc = self.Wc.to(device)

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points_square_root(self.x, self.P)

        self.sigmas_f = fx(sigmas.reshape(-1, self._dim_x), dt).reshape(sigmas.shape)

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        Pxz = torch.zeros((sigmas_f.size()[1], sigmas_h.size()[1])).to(x.device)
        N = sigmas_f.size()[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * torch.outer(dx, dz)
        return Pxz
    
    def predict(self, dt=None, fx=None, **fx_args):
        if dt is None:
            dt = self._dt


        self.compute_process_sigmas(dt, fx, **fx_args)
        self.x, self.P = unscented_transform_square_root(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)
        self.sigmas_f = self.points_fn.sigma_points_square_root(self.x, self.P)

        self.x_prior = torch.clone(self.x)
        self.P_prior = torch.clone(self.P)

    def update(self, z, R=None, hx=None, **hx_args):
        if z is None:
            self.z = torch.zeros((self._dim_z, 1))
            self.x_post = torch.clone(self.x)
            self.P_post = torch.clone(self.P)
            return
        
        if hx is None:
            hx = self.hx

        if R is None:
            R = self.R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        n_sigmas, dim_x = self.sigmas_f.shape
        self.sigmas_h = hx(self.sigmas_f.reshape(-1, self._dim_x)).reshape(n_sigmas, self._dim_z)
        # mean and covariance of prediction passed through unscented transform
        zp, self.S = unscented_transform_square_root(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)

        # compute cross variance of the state and the measurements
        #Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        X_k_pred_centered = self.sigmas_f - self.x[None, :]
        Y_k_pred_centered = self.sigmas_h - zp[None, :]

        Pxz = torch.sum(
            self.Wc[:, None, None]
            * (X_k_pred_centered[:, :, None] @ Y_k_pred_centered[:, None, :]),
            dim=0,
        )

        self.K = torch.linalg.solve(self.S.transpose(-1, -2), torch.linalg.solve(self.S, Pxz.transpose(-1, -2))).transpose(-1, -2)
        
        self.y = self.residual_z(z.float(), zp)   # residual
        # print('K: ', self.K.dtype)
        # print('y: ', self.y.dtype)
        # update Gaussian state estimate (x, P)
        #print(self.K)
        #print(torch.mm(self.K, self.y[:, None]))
        self.x = self.state_add(self.x, torch.mm(self.K, self.y[:, None]).squeeze())
        U = self.K @ self.S
        P = self.P
        for i in range(U.shape[1]):
            P = cholupdate(P, U[:, i], weight=torch.tensor(-1))
        self.P = P
        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = torch.clone(self.x)
        self.P_post = torch.clone(self.P)

        return self.x.cpu()

