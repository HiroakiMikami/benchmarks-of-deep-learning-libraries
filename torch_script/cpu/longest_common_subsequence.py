import argparse
import time

import torch


@torch.jit.script
def length_of_longest_common_subsequence(str0: str, str1: str) -> int:
    N = len(str0)
    M = len(str1)
    table = torch.zeros(N + 1, M + 1, dtype=torch.long)

    for i in range(N):
        for j in range(M):
            if str0[i] == str1[j]:
                table[i + 1, j + 1] = table[i, j] + 1
            else:
                table[i + 1, j + 1] = torch.max(table[i + 1, j], table[i, j + 1])
    return int(table[-1, -1].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--str0", type=str, required=True)
    parser.add_argument("--str1", type=str, required=True)
    parser.add_argument("--n-warmup", type=int, required=True)
    parser.add_argument("--n-measure", type=int, required=True)
    args = parser.parse_args()

    for _ in range(args.n_warmup):
        length_of_longest_common_subsequence(args.str0, args.str1)

    begin = time.time()
    for _ in range(args.n_measure):
        length_of_longest_common_subsequence(args.str0, args.str1)
    avg_sec = (time.time() - begin) / args.n_measure

    print(f"{avg_sec} #benchmark-time[sec]")


if __name__ == "__main__":
    main()
