import json
import os
import subprocess
import tempfile

import pytest


@pytest.mark.parametrize(
    "env",
    [
        "pytorch",
        "torchscript",
        "cython",
        "jax",
        # pytest.param("jax", marks=pytest.mark.xfail(reason="jax does not support if")),
    ]
)
def test_lcs(env: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "python",
            os.path.join("cpu", env, "longest_common_subsequence.py"),
            "--str0",
            "agcat",
            "--str1",
            "gac",
            "--n-warmup",
            "0",
            "--n-measure",
            "1",
            "--out",
            tmpdir,
        ]
        subprocess.run(cmd)
        with open(os.path.join(tmpdir, "out.json")) as f:
            out = json.load(f)
            assert out["output"] == 2
            assert "time_sec" in out
