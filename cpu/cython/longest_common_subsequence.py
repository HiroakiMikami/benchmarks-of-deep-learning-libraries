import argparse
import json
import os
import time

from cpu.cython._longest_common_subsequence import  length_of_longest_common_subsequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--str0", type=str, required=True)
    parser.add_argument("--str1", type=str, required=True)
    parser.add_argument("--n-warmup", type=int, required=True)
    parser.add_argument("--n-measure", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    n = length_of_longest_common_subsequence(args.str0, args.str1)

    for _ in range(args.n_warmup):
        length_of_longest_common_subsequence(args.str0, args.str1)

    begin = time.time()
    for _ in range(args.n_measure):
        length_of_longest_common_subsequence(args.str0, args.str1)
    avg_sec = (time.time() - begin) / args.n_measure

    with open(os.path.join(args.out, "out.json"), "w") as file:
        json.dump(
            {
                "time_sec": avg_sec,
                "str0": args.str0,
                "str1": args.str1,
                "output": n,
            },
            file,
        )


if __name__ == "__main__":
    main()
