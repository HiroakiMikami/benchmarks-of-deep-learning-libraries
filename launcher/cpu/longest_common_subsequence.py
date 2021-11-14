import argparse
import os
import random
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

    str0 = "".join(random.choices(string.ascii_letters + string.digits, k=100))
    str1 = "".join(random.choices(string.ascii_letters + string.digits, k=200))

    path = os.path.join(os.getcwd(), "cpu", args.target, "longest_common_subsequence.py")
    out_dir = os.path.join(args.out_dir, args.target)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python",
        path,
        "--str0",
        str0,
        "--str1",
        str1,
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
