"""The LLM agent's real per-world sandbox: no network, the host's user, cleaned up on close."""

from __future__ import annotations

import shutil
import subprocess

import pytest
from onc_agi.adapters.agents.llm import DockerSandbox, HarnessConfig

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0


@pytest.mark.skipif(not _docker_available(), reason="needs a Docker daemon")
def test_the_sandbox_runs_python_and_bash_offline_and_cleans_up() -> None:
    box = DockerSandbox.start(HarnessConfig(tool_timeout=60))
    try:
        box.write_data("patient_id,stratum,outcome,f00\nr0,all,1,0.5\nr1,all,0,-0.5\n")
        out, failed = box.run(
            ["python", "-"], stdin="import pandas as pd\nprint(pd.read_csv('/data/revealed.csv').shape)"
        )
        assert (out, failed) == ("(2, 4)", False)
        out, failed = box.run(["bash", "-lc", "touch /data/scratch && ls /data"])
        assert not failed and "scratch" in out
        out, failed = box.run(
            ["python", "-"],
            stdin="import urllib.request\nurllib.request.urlopen('http://1.1.1.1', timeout=5)",
        )
        assert failed  # no network
        out, failed = box.run(["python", "-"], stdin="raise SystemExit(3)")
        assert failed
    finally:
        box.close()
    assert not box.dir.exists()  # data and scratch files (owned by the host user) are removed
    gone = subprocess.run(["docker", "inspect", box.container], capture_output=True)
    assert gone.returncode != 0
