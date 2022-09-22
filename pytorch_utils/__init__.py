from pathlib import Path

def parse_ckpt(path, return_first=True):
    ckpts = [p.as_posix() for p in Path(path).glob("**/*") if p.suffix == ".ckpt"]
    if return_first:
        ckpt = ckpts[0]
        print("Loading checkpoint: ", ckpt)
        return ckpt
    else:
        print("Found {} checkpoits.".format(len(ckpts)))
        return ckpts