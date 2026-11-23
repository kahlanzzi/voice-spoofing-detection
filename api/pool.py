"""The bounded analysis process pool (D53 (2); docs/api_design.md §4.3 step 6, §8).

K worker processes, started with the "spawn" method, set explicitly: the API process holds a psycopg pool
and threads, and the default start method differs between Python 3.13 (fork) and 3.14 (forkserver).
Each worker builds its analyzer once, at start, with the factory it is given, then serves one job at a time
over its own pipe. Workers share no queue, lock or semaphore, so one of them can be killed (on timeout)
without touching the others; it is then replaced by a new worker built with the same factory.

There is no queue: slot() raises PoolBusy when every worker is busy (the route answers 503 with
Retry-After). Workers never touch the database: the parent stores the record after the worker answers.

The factory is pickled into each worker, so it must be a module-level callable (a class or a function, or
a functools.partial of one). P1 passes a factory that loads and verifies the frozen ModelBundle; raising in
it refuses the start (or, for a replacement, marks the pool broken). The pool itself does not change.

Linux and macOS only: the parent waits on each worker's pipe with the event loop's add_reader, which
Windows' default loop does not have; start() refuses there (run the API in Docker or WSL2).
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import multiprocessing
import os
import signal
import sys
import traceback
from collections import deque
from contextlib import asynccontextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import AsyncIterator, Callable

from api.analyzer import AnalysisOutcome, Analyzer, AnalyzerInfo, AnalyzerUnavailable

log = logging.getLogger("api")

AnalyzerFactory = Callable[[], Analyzer]

START_METHOD = "spawn"
STOP_JOIN_S = 5  # a worker asked to stop at shutdown gets this long before it is killed
KILL_JOIN_S = 1  # reaping after SIGKILL takes milliseconds; the bound only guards against the unexpected
NEEDS_SELECTOR_LOOP = ("the analysis pool needs a selector event loop (Linux or macOS); "
                       "on Windows run the API in Docker or WSL2")


class PoolStartError(Exception):
    """A worker's start-up hook failed, timed out or disagreed with the others: the API must not start."""


class PoolBusy(Exception):
    """Every worker is busy: 503 analysis_unavailable with Retry-After."""


class PoolUnavailable(Exception):
    """The pool is not started, is shutting down, is broken, or its analyzer reports not ready: 503."""


class AnalyzerFailed(Exception):
    """The analyzer raised in the worker; the worker is still usable. Holds the worker's traceback text."""

    def __init__(self, worker_traceback: str):
        super().__init__("the analyzer raised in its worker")
        self.worker_traceback = worker_traceback


class WorkerLost(Exception):
    """The worker died, or its pipe failed, during a job."""


def _worker_main(conn: Connection, factory: AnalyzerFactory) -> None:
    """The worker process: build the analyzer once, report its info, then answer one job at a time."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C stops the parent, which then stops the workers
    try:
        analyzer = factory()
        conn.send(("ready", analyzer.info()))
    except BaseException:
        conn.send(("failed", traceback.format_exc()))
        return
    while True:
        try:
            path = conn.recv()
        except EOFError:  # the parent is gone: exit, never linger as an orphan
            return
        if path is None:  # clean stop
            return
        try:
            reply = ("ok", analyzer.analyze(Path(path)))
        except AnalyzerUnavailable:
            reply = ("unavailable", None)
        except Exception:
            reply = ("error", traceback.format_exc())
        conn.send(reply)


def _check_event_loop() -> None:
    """Refuse to start unless the running loop can watch a pipe (add_reader): a selector event loop on
    Linux or macOS. Windows (its pipes and its default Proactor loop) is not supported."""
    if sys.platform == "win32":
        raise PoolStartError(NEEDS_SELECTOR_LOOP)
    loop = asyncio.get_running_loop()
    read_fd, write_fd = os.pipe()
    try:
        loop.add_reader(read_fd, lambda: None)
    except NotImplementedError:
        raise PoolStartError(NEEDS_SELECTOR_LOOP) from None
    else:
        loop.remove_reader(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


async def _readable(conn: Connection, timeout_s: float) -> None:
    """Wait until the pipe has data (or EOF) without a thread; TimeoutError after timeout_s."""
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    fd = conn.fileno()
    loop.add_reader(fd, lambda: ready.done() or ready.set_result(None))
    try:
        await asyncio.wait_for(ready, timeout_s)
    finally:
        loop.remove_reader(fd)


class _Worker:
    def __init__(self, process: multiprocessing.process.BaseProcess, conn: Connection):
        self.process = process
        self.conn = conn
        self.pid = process.pid
        self.closed = False
        self.exitcode: int | None = None  # set when the pool has killed and reaped it


class JobState(enum.Enum):
    IDLE = "idle"        # slot taken, nothing sent yet
    SENT = "sent"        # job sent, no answer yet
    DONE = "done"        # answer received; the worker is free again
    RETIRED = "retired"  # the worker was killed during this job


class Job:
    """One slot: a worker held by one request."""

    def __init__(self, worker: _Worker):
        self.worker = worker
        self.state = JobState.IDLE


class AnalysisPool:
    def __init__(self, factory: AnalyzerFactory, size: int, start_timeout_s: float):
        if size < 1:
            raise ValueError(f"the pool needs at least one worker, got {size!r}")
        self.factory = factory
        self.size = size
        self.start_timeout_s = start_timeout_s
        self.context = multiprocessing.get_context(START_METHOD)
        self._idle: deque[_Worker] = deque()
        self._held: set[_Worker] = set()
        self._restarting = 0
        self._tasks: set[asyncio.Task] = set()
        self._info: AnalyzerInfo | None = None
        self._started = False
        self._closing = False
        self._broken = False

    # --- start and stop ---------------------------------------------------------------------------

    def _spawn(self) -> _Worker:
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(target=_worker_main, args=(child, self.factory), daemon=True,
                                       name="analysis-worker")
        process.start()
        child.close()  # the worker holds its own copy; the parent then sees EOF if the worker dies
        return _Worker(process, parent)

    async def _handshake(self, worker: _Worker) -> AnalyzerInfo:
        try:
            await _readable(worker.conn, self.start_timeout_s)
        except TimeoutError:
            raise PoolStartError(f"a worker did not start within {self.start_timeout_s} s") from None
        try:
            kind, payload = worker.conn.recv()
        except (EOFError, OSError):
            raise PoolStartError(f"a worker exited during start (exit code {worker.process.exitcode})") from None
        if kind != "ready":
            raise PoolStartError(f"a worker's start-up hook failed:\n{payload}")
        if not isinstance(payload, AnalyzerInfo):
            raise PoolStartError("a worker's analyzer info() is not an AnalyzerInfo")
        return payload

    async def start(self) -> None:
        """Start K workers and wait until each has built its analyzer. Raise PoolStartError if any fails,
        times out, or reports an info() different from the others, or if the event loop cannot watch a pipe
        (Windows); no worker is left running then."""
        if self._started or self._closing:
            raise RuntimeError("the pool can be started once")
        _check_event_loop()  # before any worker is spawned
        workers: list[_Worker] = []
        try:
            for _ in range(self.size):
                workers.append(self._spawn())
            results = await asyncio.gather(*(self._handshake(w) for w in workers), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if any(info != results[0] for info in results):
                raise PoolStartError("workers report different analyzer info")
        except BaseException:
            for worker in workers:
                self._kill(worker)
            raise
        self._info = results[0]
        self._idle.extend(workers)
        self._started = True

    async def close(self) -> None:
        """Stop every worker: idle ones are asked to stop, then anything still alive is killed. Pending
        replacements are cancelled. Safe to call more than once and after a failed start."""
        self._closing = True
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        workers = [*self._idle, *self._held]
        self._idle.clear()
        for worker in workers:
            if not worker.closed:
                try:
                    worker.conn.send(None)
                except OSError:
                    pass
        for worker in workers:
            if not worker.closed:
                await asyncio.to_thread(worker.process.join, STOP_JOIN_S)
                self._kill(worker)

    # --- state ------------------------------------------------------------------------------------

    def info(self) -> AnalyzerInfo:
        """The analyzer's info from the start handshake; ready only while the pool can serve. A full pool,
        or a worker being replaced, is still ready (busy is not unavailable)."""
        if self._info is None:
            raise PoolUnavailable("the pool is not started")
        ready = self._info.ready and self._started and not self._closing and not self._broken
        return dataclasses.replace(self._info, ready=ready)

    def counts(self) -> dict[str, int]:
        """idle + held + restarting == size while the pool is open and not broken."""
        return {"idle": len(self._idle), "held": len(self._held), "restarting": self._restarting}

    def worker_pids(self) -> set[int]:
        return {w.pid for w in [*self._idle, *self._held]}

    # --- slots ------------------------------------------------------------------------------------

    def _pop_idle(self) -> _Worker | None:
        while self._idle:
            worker = self._idle.popleft()
            if worker.process.is_alive():
                return worker
            log.warning("analysis worker found dead while idle (exit code %s); replacing it",
                        worker.process.exitcode)
            self._retire(worker)
        return None

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[Job]:
        """Take one idle worker for one request, without waiting. Raise PoolUnavailable or PoolBusy.
        On exit the worker goes back to idle if no job is in flight on it, otherwise it is killed and
        replaced. The check and the pop run with no await between them, on the event-loop thread."""
        if self._info is None or not self.info().ready:
            raise PoolUnavailable("the analysis pool cannot serve")
        worker = self._pop_idle()
        if worker is None:
            raise PoolBusy()
        job = Job(worker)
        self._held.add(worker)
        try:
            yield job
        finally:
            self._held.discard(worker)
            if job.state in (JobState.IDLE, JobState.DONE):
                self._return(worker)
            else:
                self._retire(worker)

    def _return(self, worker: _Worker) -> None:
        if worker in self._idle:
            raise RuntimeError("an analysis worker was released twice")
        if self._closing or worker.closed:
            self._kill(worker)
            return
        self._idle.append(worker)

    async def run(self, job: Job, path: Path, timeout_s: float) -> AnalysisOutcome:
        """Run one analysis on the job's worker. On timeout, cancellation or a lost worker, the worker is
        killed and reaped here, before the exception leaves (so the caller's temp file is not held open
        by it). Raise TimeoutError, AnalyzerUnavailable, AnalyzerFailed or WorkerLost."""
        if job.state is not JobState.IDLE:
            raise RuntimeError("one job per slot")
        worker = job.worker
        job.state = JobState.SENT
        try:
            worker.conn.send(str(path))
            await _readable(worker.conn, timeout_s)
            kind, payload = worker.conn.recv()
        except TimeoutError:  # before OSError: TimeoutError is an OSError subclass
            self._kill(worker)
            job.state = JobState.RETIRED
            raise
        except (EOFError, OSError) as exc:
            self._kill(worker)
            job.state = JobState.RETIRED
            raise WorkerLost() from exc
        except BaseException:  # CancelledError (client gone, shutdown), anything else
            self._kill(worker)
            job.state = JobState.RETIRED
            raise
        job.state = JobState.DONE
        if kind == "ok" and isinstance(payload, AnalysisOutcome):
            return payload
        if kind == "unavailable":
            raise AnalyzerUnavailable()
        if kind == "error":
            raise AnalyzerFailed(str(payload))
        raise AnalyzerFailed(f"unexpected worker reply {kind!r}")

    # --- kill and replace -------------------------------------------------------------------------

    def _kill(self, worker: _Worker) -> None:
        """SIGKILL and reap one worker, close its pipe. Idempotent."""
        if worker.closed:
            return
        worker.closed = True
        worker.process.kill()
        worker.process.join(KILL_JOIN_S)
        worker.conn.close()
        worker.exitcode = worker.process.exitcode
        if worker.exitcode is not None:
            worker.process.close()
        else:  # not reaped within KILL_JOIN_S: leave it to the daemon backstop, never hide it
            log.error("analysis worker %s did not exit after SIGKILL", worker.pid)

    def _retire(self, worker: _Worker) -> None:
        """Kill a worker that cannot be reused and start its replacement in the background."""
        self._kill(worker)
        if self._closing:
            return
        self._restarting += 1
        task = asyncio.get_running_loop().create_task(self._replace())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _replace(self) -> None:
        """Start one new worker with the same factory. If its start-up hook fails, the pool is broken: it
        stops taking new work (503 analysis_unavailable, /status unavailable) until the API is restarted;
        jobs already running on other workers still finish. No automatic retry (D53 (2) plan, P2b)."""
        worker = None
        try:
            worker = self._spawn()
            info = await self._handshake(worker)
            if info != self._info:
                raise PoolStartError("a replacement worker reports different analyzer info")
        except asyncio.CancelledError:
            if worker is not None:
                self._kill(worker)
            raise
        except Exception as exc:
            if worker is not None:
                self._kill(worker)
            self._broken = True
            log.error("analysis pool broken: a replacement worker did not start; restart the API (%s)", exc)
        else:
            if self._closing:
                self._kill(worker)
            else:
                self._idle.append(worker)
        finally:
            self._restarting -= 1
