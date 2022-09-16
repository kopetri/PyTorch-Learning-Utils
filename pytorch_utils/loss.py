import torch

def chamfer_distance(S1, S2):
    def __cdist(S1, S2):
        N = S1.shape[1]
        M = S2.shape[1]
        C = S1.shape[-1]
        X = torch.repeat_interleave(S1, M, dim=1)
        Y = S2.repeat(1, N, 1)

        diff = (X-Y).reshape(-1, N, M, C)
        dist = torch.norm(diff, dim=-1)
        minimum, _ = torch.min(dist, dim=-1)
        return torch.sum(minimum, dim=-1)

    return torch.sum(__cdist(S1, S2) + __cdist(S2, S1))
