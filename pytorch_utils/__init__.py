from pathlib import Path
import torch
import numpy as np

def generate_splits(directory, suffix, train=0.8, valid=0.05, test=0.15, shuffle=True):
    if isinstance(suffix, str):
        suffix = [suffix]
    assert train + valid + test == 1.0, train + valid + test
    directory = Path(directory)
    print("Generating splits...")
    def __write_files__(files, out):
        if isinstance(out, Path):
            out = out.as_posix()
        with open(out, "w") as txtfile:
            for f in sorted(files):
                txtfile.write("{}\n".format(f.relative_to(directory).as_posix()))

    files = [f for f in directory.rglob("*") if f.suffix in suffix]

    N = len(files)
    n_train = int(N * train)
    n_valid = int(N * valid)
    n_test  = int(N * test)

    rest = N - (n_train + n_valid + n_test)
    n_test += rest

    assert n_train + n_valid + n_test == N, "{} !+ {}".format(n_train + n_valid + n_test, N)

    if shuffle:
        np.random.shuffle(files)

    train_split = files[0:n_train]
    valid_split = files[n_train:n_train+n_valid]
    test_split  = files[n_train+n_valid:n_train+n_valid+n_test]

    __write_files__(train_split, Path(directory, "train.txt"))
    __write_files__(valid_split, Path(directory, "valid.txt"))
    __write_files__(test_split,  Path(directory,  "test.txt"))


def parse_ckpt(path, return_first=True):
    if Path(path).is_file():
        print("Loading checkpoint: ", path)
        return path
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