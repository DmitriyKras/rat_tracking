import torch


class LinearKalmanFilter:
    def __init__(self, A: torch.Tensor, H: torch.Tensor, Q: torch.Tensor, 
                 R: torch.Tensor, x: torch.Tensor):
        self.A = A.float()
        self.H = H.float()
        self.Q = Q.float()
        self.R = R.float()
        self.P = torch.eye(A.size()[0]) * 10
        self.x = x[:, None].float()
        self.A_t = A.T.float()
        self.H_t = H.T.float()
        self.I = torch.eye(A.size()[0])
        self.device = 'cpu'

    def to(self, device='cuda'):
        self.A = self.A.to(device)
        self.H = self.H.to(device)
        self.Q = self.Q.to(device)
        self.R = self.R.to(device)
        self.P = self.P.to(device)
        self.x = self.x.to(device)
        self.A_t = self.A_t.to(device)
        self.H_t = self.H_t.to(device)
        self.I = self.I.to(device)
        self.device = device

    def predict(self):
        self.x = torch.mm(self.A, self.x)
        self.P = torch.mm(torch.mm(self.A, self.P), self.A_t) + self.Q

    def update(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device).float()
        P_inv = torch.inverse(self.P)
        S = torch.mm(torch.mm(self.H, P_inv), self.H_t) + self.R
        S_inv = torch.inverse(S)
        K = torch.mm(torch.mm(P_inv, self.H_t), S_inv)
        self.x += torch.mm(K, z[:, None] - torch.mm(self.H, self.x))
        self.P = torch.mm(self.I - torch.mm(K, self.H), P_inv)
        return self.x.cpu().flatten()
    