import argparse
import time
import os
import json

import torch
import torchvision


def _flip(value: torch.Tensor) -> torch.Tensor:
    return torchvision.transforms.RandomHorizontalFlip(p=1.0)(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--n-warmup", type=int, required=True)
    parser.add_argument("--n-measure", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.input_path)

    output = _flip(data)

    for _ in range(args.n_warmup):
        _flip(data)

    begin = time.time()
    for _ in range(args.n_measure):
        _flip(data)
    avg_sec = (time.time() - begin) / args.n_measure

    with open(os.path.join(args.out, "out.json"), "w") as file:
        json.dump({
            "time_sec": avg_sec,
        }, file)
    torch.save(output, os.path.join(args.out, "output"))
    torch.save(data, os.path.join(args.out, "input"))


if __name__ == "__main__":
    main()
