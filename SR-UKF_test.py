from kalmantorch import MerweScaledSigmaPoints, unscented_transform_square_root
from dynamic_models import CTRVcomputeQ
import torch


dim_x = 5
points = MerweScaledSigmaPoints(dim_x, 0.5, 2, dim_x - 3)

x = torch.ones((dim_x, ))
P = CTRVcomputeQ(x, 9, 9, 9)
print(x)
print(P)

sigmas = points.sigma_points_square_root(x, torch.linalg.cholesky(P))
print(sigmas)


x, P = unscented_transform_square_root(sigmas, points.Wm, points.Wc, torch.ones((dim_x, dim_x)))

print(x)
print(P)