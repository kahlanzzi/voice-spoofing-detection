"""Export descriptive D52 val-B scores; never an external-run entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

from src.evaluation.live_warmup_report import report_json
from src.evaluation.live_warmup_scoring import score_call_plan


def export_report(output: Path, **scoring_args) -> str:
    """Reserve a new output before inference and preserve failures as empty files.

    The exclusive file prevents accidental overwrites and concurrent writers.
    A failed run leaves an empty reservation, not a completed JSON report.
    Runtime versions describe the environment; they do not certify its approval.
    """
    runtime = {name: version(name) for name in
               ("numpy", "scipy", "librosa", "soundfile", "torch", "safetensors")}
    runtime.update(python=platform.python_version(), platform=platform.platform(), device="cpu")
    with Path(output).open("xb") as stream:
        started = datetime.now(timezone.utc).isoformat()
        result = score_call_plan(**scoring_args)
        result["runtime"] = runtime
        result["started_at_utc"] = started
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload = report_json(result).encode("utf-8")
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-path", "manifest-path", "audio-root", "bundle-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("expected-plan-sha256", "expected-bundle-sha256", "code-commit"):
        parser.add_argument(f"--{name}", required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    digest = export_report(output, **args)
    print(f"report: {output}")
    print(f"report_sha256: {digest}")


if __name__ == "__main__":
    main()
