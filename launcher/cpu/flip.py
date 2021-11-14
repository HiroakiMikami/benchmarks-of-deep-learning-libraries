import argparse
import os
import torch
import random
import tempfile
import string
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)

    img = torch.randint(0, 255, size=(1, 3, 224, 224)).to(torch.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "input")
        torch.save(img, data_path)

        path = os.path.join(os.getcwd(), "cpu", args.target, "flip.py")
        out_dir = os.path.join(args.out_dir, args.target)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python",
            path,
            "--input-path",
            data_path,
            "--n-warmup",
            str(args.n_warmup),
            "--n-measure",
            str(args.n_measure),
            "--out", out_dir,
        ]
        p = subprocess.run(cmd)
    assert p.returncode == 0, f"returncode={p.returncode}"


if __name__ == "__main__":
    main()
