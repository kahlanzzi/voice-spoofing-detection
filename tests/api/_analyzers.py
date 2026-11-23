"""Test analyzers that run inside spawned pool workers (api/pool.py).

A spawned worker unpickles its factory by reference, so these live in an importable module, not in a test
file. They talk to the test only through files in a control folder:
- build_<n>: one file per analyzer build (n is the build's ordinal, taken with an exclusive create);
- mode: when present, its text sets what analyze() does for every worker (app-level tests);
- a job file whose bytes start with "cmd:" sets what analyze() does for that job (pool-level tests);
- started_<pid>: written when a job starts; release: lets a "wait" job finish.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from api.analyzer import AnalysisOutcome, AnalyzerInfo, AnalyzerUnavailable, MockAnalyzer

WAIT_LIMIT_S = 60


def take_build_ordinal(control: Path) -> int:
    n = 1
    while True:
        try:
            fd = os.open(control / f"build_{n}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return n


def builds(control: Path) -> int:
    return len(list(Path(control).glob("build_*")))


def started(control: Path) -> list[Path]:
    return sorted(Path(control).glob("started_*"))


class ScriptedAnalyzer(MockAnalyzer):
    """MockAnalyzer outcomes, unless the job file or the control folder's mode says otherwise.

    Commands: hang (sleep past any timeout), wait (until control/release exists), exit (os._exit: the
    worker dies mid-job), error (RuntimeError), unavailable (AnalyzerUnavailable), modules (write which
    database modules this process has imported to control/modules_<pid>.json).
    fail_from_build: builds with this ordinal or later raise (start-up hook failure).
    start_delay_s: sleep this long while building (start timeout).
    pid_versions: report the pid as model_version, so workers disagree on info()."""

    def __init__(self, control: str, fail_from_build: int | None = None, start_delay_s: float = 0.0,
                 pid_versions: bool = False, ready: bool = True, deferral_on_fit: float = 0.0993):
        super().__init__(deferral_on_fit)
        self.control = Path(control)
        self.build = take_build_ordinal(self.control)
        self.pid_versions = pid_versions
        self.ready = ready
        if start_delay_s:
            time.sleep(start_delay_s)
        if fail_from_build is not None and self.build >= fail_from_build:
            raise RuntimeError(f"start-up hook failed on build {self.build}")

    def command(self, path: Path) -> str | None:
        head = Path(path).read_bytes()[:64]
        if head.startswith(b"cmd:"):
            return head[4:].split(b"\n", 1)[0].decode()
        mode = self.control / "mode"
        return mode.read_text().strip() if mode.exists() else None

    def analyze(self, path: Path) -> AnalysisOutcome:
        command = self.command(path)
        (self.control / f"started_{os.getpid()}_{time.monotonic_ns()}").touch()
        if command == "hang":
            time.sleep(3600)
        elif command == "wait":
            deadline = time.monotonic() + WAIT_LIMIT_S
            while not (self.control / "release").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
        elif command == "exit":
            os._exit(3)
        elif command == "error":
            raise RuntimeError("analyzer error for the test")
        elif command == "unavailable":
            raise AnalyzerUnavailable()
        elif command == "modules":
            loaded = {name: name in sys.modules for name in ("api.db", "api.repository", "psycopg", "psycopg_pool")}
            (self.control / f"modules_{os.getpid()}.json").write_text(json.dumps(loaded))
        return super().analyze(path)

    def info(self) -> AnalyzerInfo:
        info = super().info()
        if self.pid_versions:
            info = AnalyzerInfo(**{**info.__dict__, "model_version": f"pid-{os.getpid()}"})
        return AnalyzerInfo(**{**info.__dict__, "ready": self.ready})
