import torch
from pytorch_utils import compute_distance_between_points

def chamfer_distance(S1, S2):
    def __cdist(S1, S2):
        dist = compute_distance_between_points(S1, S2)
        minimum, _ = torch.min(dist, dim=-1)
        return torch.sum(minimum, dim=-1)

    return torch.sum(__cdist(S1, S2) + __cdist(S2, S1))
