import argparse
import json
import os
import time
import numpy as np

import jax.numpy as jnp
from jax import jit
from jax.lax import cond, fori_loop


def _length_of_longest_common_subsequence(str0: jnp.array, str1: jnp.array) -> jnp.array:
    N = len(str0)
    M = len(str1)
    table = jnp.zeros(shape=(N + 1, M + 1), dtype=jnp.int64)

    def _body(i: int, table: jnp.array) -> jnp.array:
        def _body(j: int, table: jnp.array) -> jnp.array:
            def _update_true(_) -> jnp.array:
                return table[i, j] + 1

            def _update_false(_) -> jnp.array:
                v0 = table[i + 1, j]
                v1 = table[i, j + 1]
                return jnp.where(v0 > v1, v0, v1)

            v = cond(
                str0[i] == str1[j],
                _update_true,
                _update_false,
                (),
            )
            return table.at[i + 1, j + 1].set(v)
        return fori_loop(0, M, _body, table)
    table = fori_loop(0, N, _body, table)
    return table[-1, -1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--str0", type=str, required=True)
    parser.add_argument("--str1", type=str, required=True)
    parser.add_argument("--n-warmup", type=int, required=True)
    parser.add_argument("--n-measure", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    str0 = jnp.array(np.frombuffer(bytes(args.str0, "utf-8"), dtype=np.uint8))
    str1 = jnp.array(np.frombuffer(bytes(args.str1, "utf-8"), dtype=np.uint8))

    # jit
    length_of_longest_common_subsequence = jit(_length_of_longest_common_subsequence)

    n = int(length_of_longest_common_subsequence(str0, str1))

    for _ in range(args.n_warmup):
        length_of_longest_common_subsequence(str0, str1)

    begin = time.time()
    for _ in range(args.n_measure):
        length_of_longest_common_subsequence(str0, str1)
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
