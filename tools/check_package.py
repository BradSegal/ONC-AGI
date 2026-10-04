"""Build and exercise both distributions in fresh environments outside the checkout.

Usage: uv run python tools/check_package.py [--dist existing-dist-directory]
Without --dist, build into a temporary directory. CI passes its release artifacts.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="onc-package-") as directory:
        outside = Path(directory)
        dist = args.dist.resolve() if args.dist else outside / "dist"
        if not args.dist:
            run("uv", "build", "--out-dir", str(dist), cwd=ROOT)
        artifacts = [*dist.glob("*.whl"), *dist.glob("*.tar.gz")]
        if len(artifacts) != 2 or {p.suffix for p in artifacts} != {".whl", ".gz"}:
            raise SystemExit("expected exactly one wheel and one sdist")
        run("uv", "run", "--locked", "twine", "check", "--strict", *(str(p) for p in artifacts), cwd=ROOT)
        wheel = next(p for p in artifacts if p.suffix == ".whl")
        run("uv", "run", "--locked", "check-wheel-contents", str(wheel), cwd=ROOT)
        for index, artifact in enumerate(artifacts):
            consumer = outside / f"consumer-{index}"
            run("uv", "venv", "--python", "3.12", str(consumer), cwd=outside)
            python = str(consumer / "bin" / "python")
            run("uv", "pip", "install", "--python", python, str(artifact), cwd=outside)
            run(
                python,
                "-I",
                "-c",
                "import onc_agi; import onc_agi.adapters.http; import onc_agi.adapters.client",
                cwd=outside,
            )
            run(str(consumer / "bin" / "onc-agi"), "smoke", cwd=outside)
            for relative in ("docs/agents-kit.md", "tools/FIXTURES.md"):
                text = (ROOT / relative).read_text()
                blocks = re.findall(r"```python(?: run)?\n(.*?)```", text, re.S)
                if not blocks:
                    raise SystemExit(f"no offline examples in {relative}")
                run(python, "-I", "-c", "\n\n".join(blocks), cwd=outside)
        print("wheel and sdist: clean-consumer checks passed")


if __name__ == "__main__":
    main()
