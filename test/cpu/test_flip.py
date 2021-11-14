import json
import os
import pytest
import tempfile
import subprocess
import torch


@pytest.mark.parametrize("env", ["pytorch", "torchscript"])
def test_lcs(env: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        img = torch.arange(10).reshape(1, 1, 1, 10).expand(-1, 3, 10, -1).to(torch.uint8)
        data_path = os.path.join(tmpdir, "input")
        torch.save(img, data_path)

        cmd = [
            "python",
            os.path.join("cpu", env, "flip.py"),
            "--input-path", data_path,
            "--n-warmup", "0", "--n-measure", "1",
            "--out", tmpdir,
        ]
        subprocess.run(cmd)
        output = torch.load(os.path.join(tmpdir, "output"))
        expected = img.clone()
        for i in range(img.shape[2]):
            expected[:, :, :, i] = img[:, :, :, img.shape[2] - i - 1]
        assert torch.all(output == expected)
        with open(os.path.join(tmpdir, "out.json")) as f:
            out = json.load(f)
            assert "time_sec" in out
