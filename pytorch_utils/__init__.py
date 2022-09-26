from pathlib import Path
import torch

def parse_ckpt(path, return_first=True):
    ckpts = [p.as_posix() for p in Path(path).glob("**/*") if p.suffix == ".ckpt"]
    if return_first:
        ckpt = ckpts[0]
        print("Loading checkpoint: ", ckpt)
        return ckpt
    else:
        print("Found {} checkpoits.".format(len(ckpts)))
        return ckpts

def compute_distance_between_points(A, B):
    # A.shape (B, N, C)
    # B.shape (B, M, C)
    # dist.shape (B, N, M)
    N = A.shape[1]
    M = B.shape[1]
    X = A.unsqueeze(2).repeat(1, 1, M, 1)
    Y = B.unsqueeze(1).repeat(1, N, 1, 1)

    diff = X - Y
    diff = torch.pow(diff, 2)
    diff = torch.sum(diff, dim=-1)
    dist = torch.sqrt(diff)
    return dist