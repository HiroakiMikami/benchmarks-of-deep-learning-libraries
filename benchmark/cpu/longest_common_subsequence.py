import os
import argparse
import random
import string
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--target-dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)

    str0 = "".join(
        random.choices(string.ascii_letters + string.digits, k=100)
    )
    str1 = "".join(
        random.choices(string.ascii_letters + string.digits, k=200)
    )

    path = os.path.join(
        args.target_dir, "cpu", "longest_common_subsequence.sh"
    )

    cmd = [
        path,
        "--str0", str0,
        "--str1", str1,
        "--n-warmup", str(args.n_warmup),
        "--n-measure", str(args.n_measure),
    ]
    p = subprocess.run(cmd, capture_output=True)
    assert p.returncode == 0, f"returncode={p.returncode}"
    for line in p.stdout.decode("utf-8").split("\n"):
        if line.endswith(" #benchmark-time[sec]"):
            result = float(line.split(" ")[0])
            break
    cmd_str = " ".join(cmd)
    print(f"command: {cmd_str}")
    print(f"Time[sec] {result}")


if __name__ == "__main__":
    main()
