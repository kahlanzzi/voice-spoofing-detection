"""api/pool.py with real spawned worker processes (D53 (2)).

Every test starts real workers with the spawn method; the analyzers are in tests/api/_analyzers.py and are
steered through files in a control folder. No database, no app: the pool alone.
"""
from __future__ import annotations

import asyncio
import functools
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

from api.analyzer import AnalysisOutcome, AnalyzerUnavailable
from api.pool import (NEEDS_SELECTOR_LOOP, START_METHOD, AnalysisPool, AnalyzerFailed, JobState, PoolBusy,
                      PoolStartError, PoolUnavailable, WorkerLost)
from tests.api._analyzers import ScriptedAnalyzer, builds, started

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason=NEEDS_SELECTOR_LOOP)

TIMEOUT_S = 10


LINUX = sys.platform == "linux"


def alive(pid: int) -> bool:
    """Whether a process still exists, from /proc (Linux only: the container and CI; macOS has no /proc)."""
    return Path(f"/proc/{pid}").exists()


def assert_reaped(worker) -> None:
    """The pool killed and reaped this worker; on Linux, /proc confirms that the pid is gone."""
    assert worker.closed and worker.exitcode is not None
    if LINUX:
        assert not alive(worker.pid)


def job_file(folder: Path, name: str, command: str | None = None) -> Path:
    path = folder / name
    path.write_bytes((f"cmd:{command}\n" if command else "") .encode() + os.urandom(256))
    return path


def factory(control: Path, **kwargs):
    return functools.partial(ScriptedAnalyzer, str(control), **kwargs)


@pytest.fixture
def control(tmp_path) -> Path:
    folder = tmp_path / "control"
    folder.mkdir()
    return folder


def run(coro):
    return asyncio.run(coro)


async def started_pool(control: Path, size: int = 2, **kwargs) -> AnalysisPool:
    pool = AnalysisPool(factory(control, **kwargs), size=size, start_timeout_s=30)
    await pool.start()
    return pool


async def until(predicate, limit_s: float = 30) -> None:
    deadline = time.monotonic() + limit_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.02)


def assert_invariant(pool: AnalysisPool) -> None:
    counts = pool.counts()
    assert counts["idle"] + counts["held"] + counts["restarting"] == pool.size, counts


async def analyze(pool: AnalysisPool, path: Path, timeout_s: float = TIMEOUT_S) -> AnalysisOutcome:
    async with pool.slot() as job:
        return await pool.run(job, path, timeout_s)


# --- start ---------------------------------------------------------------------------------------

def test_pool_uses_spawn_start_method(control):
    async def main():
        pool = await started_pool(control, size=1)
        try:
            assert START_METHOD == "spawn"
            assert pool.context.get_start_method() == "spawn"
            (worker,) = pool._idle
            assert type(worker.process).__name__ == "SpawnProcess"
        finally:
            await pool.close()
    run(main())


def test_analyzer_built_once_per_worker_across_jobs(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=2)
        try:
            assert builds(control) == 2
            for i in range(6):
                outcome = await analyze(pool, job_file(tmp_path, f"j{i}"))
                assert outcome.model_version == "mock"
            assert builds(control) == 2
            assert len({p.name.split("_")[1] for p in started(control)}) == 2  # both workers served jobs
        finally:
            await pool.close()
    run(main())


def test_workers_never_import_the_database_layer(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=1)
        try:
            await analyze(pool, job_file(tmp_path, "j", "modules"))
        finally:
            await pool.close()
    run(main())
    (report,) = control.glob("modules_*.json")
    assert json.loads(report.read_text()) == {"api.db": False, "api.repository": False, "psycopg": False,
                                              "psycopg_pool": False}


def test_startup_failure_of_one_worker_refuses_to_start_and_leaves_no_child(control):
    pool = AnalysisPool(factory(control, fail_from_build=2), size=2, start_timeout_s=30)
    with pytest.raises(PoolStartError, match="start-up hook failed on build 2"):
        run(pool.start())
    assert builds(control) == 2
    if LINUX:
        pids = [int(p.read_text()) for p in control.glob("build_*")]
        assert pids and not [pid for pid in pids if alive(pid)]
    with pytest.raises(PoolUnavailable):
        pool.info()


def test_startup_timeout_refuses_to_start(control):
    pool = AnalysisPool(factory(control, start_delay_s=30), size=1, start_timeout_s=1)
    started_at = time.monotonic()
    with pytest.raises(PoolStartError, match="did not start within 1 s"):
        run(pool.start())
    assert time.monotonic() - started_at < 10


def test_workers_with_different_info_refuse_to_start(control):
    pool = AnalysisPool(factory(control, pid_versions=True), size=2, start_timeout_s=30)
    with pytest.raises(PoolStartError, match="different analyzer info"):
        run(pool.start())


def test_pool_refuses_zero_workers(control):
    with pytest.raises(ValueError):
        AnalysisPool(factory(control), size=0, start_timeout_s=30)


# --- jobs ----------------------------------------------------------------------------------------

def test_full_pool_raises_busy_and_a_freed_slot_serves_again(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=2)
        try:
            waiting = [asyncio.create_task(analyze(pool, job_file(tmp_path, f"w{i}", "wait"))) for i in range(2)]
            await until(lambda: len(started(control)) == 2)
            with pytest.raises(PoolBusy):
                async with pool.slot():
                    pass
            assert pool.info().ready  # busy is not unavailable
            (control / "release").touch()
            await asyncio.gather(*waiting)
            assert (await analyze(pool, job_file(tmp_path, "next"))).model_version == "mock"
            assert pool.counts() == {"idle": 2, "held": 0, "restarting": 0}
        finally:
            await pool.close()
    run(main())


def test_timeout_kills_and_replaces_exactly_one_worker(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=2)
        try:
            before = pool.worker_pids()
            async with pool.slot() as job:
                hung = job.worker.pid
                with pytest.raises(TimeoutError):
                    await pool.run(job, job_file(tmp_path, "hang", "hang"), 1)
                assert job.state is JobState.RETIRED
                assert_reaped(job.worker)  # killed and reaped before the exception left run()
            assert_invariant(pool)
            await until(lambda: pool.counts()["idle"] == 2)
            after = pool.worker_pids()
            assert before - after == {hung}  # exactly one worker replaced
            assert len(after - before) == 1
            assert builds(control) == 3
            for i in range(4):
                await analyze(pool, job_file(tmp_path, f"after{i}"))
        finally:
            await pool.close()
    run(main())


def test_analyzer_error_keeps_the_worker(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=1)
        try:
            before = pool.worker_pids()
            with pytest.raises(AnalyzerFailed) as caught:
                await analyze(pool, job_file(tmp_path, "e", "error"))
            assert "analyzer error for the test" in caught.value.worker_traceback
            with pytest.raises(AnalyzerUnavailable):
                await analyze(pool, job_file(tmp_path, "u", "unavailable"))
            assert pool.worker_pids() == before and pool.counts()["idle"] == 1
            await analyze(pool, job_file(tmp_path, "ok"))
        finally:
            await pool.close()
    run(main())


def test_worker_death_mid_job_is_worker_lost_and_replaced(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=1)
        try:
            (before,) = pool.worker_pids()
            with pytest.raises(WorkerLost):
                await analyze(pool, job_file(tmp_path, "x", "exit"))
            assert_invariant(pool)
            await until(lambda: pool.counts()["idle"] == 1)
            assert before not in pool.worker_pids()
            await analyze(pool, job_file(tmp_path, "ok"))
        finally:
            await pool.close()
    run(main())


def test_replacement_failure_marks_pool_broken(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=2, fail_from_build=3)  # builds 1 and 2 start, 3 fails
        try:
            with pytest.raises(TimeoutError):
                await analyze(pool, job_file(tmp_path, "hang", "hang"), 1)
            await until(lambda: pool.counts()["restarting"] == 0)
            assert not pool.info().ready
            with pytest.raises(PoolUnavailable):
                async with pool.slot():
                    pass
        finally:
            await pool.close()
    run(main())


def test_cancelled_job_releases_and_replaces_its_worker(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=1)
        try:
            (worker,) = pool._idle
            task = asyncio.create_task(analyze(pool, job_file(tmp_path, "w", "hang")))
            await until(lambda: len(started(control)) == 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert_reaped(worker)
            assert_invariant(pool)
            await until(lambda: pool.counts()["idle"] == 1)
            await analyze(pool, job_file(tmp_path, "ok"))
        finally:
            await pool.close()
    run(main())


def test_send_failure_after_taking_slot_releases_it(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=1)
        try:
            async with pool.slot() as job:
                job.worker.conn.close()  # the parent's pipe fails before the job is sent
                with pytest.raises(WorkerLost):
                    await pool.run(job, job_file(tmp_path, "j"), TIMEOUT_S)
            assert_invariant(pool)
            await until(lambda: pool.counts()["idle"] == 1)
            await analyze(pool, job_file(tmp_path, "ok"))
        finally:
            await pool.close()
    run(main())


def test_exception_in_the_slot_before_submitting_returns_the_worker(control):
    async def main():
        pool = await started_pool(control, size=1)
        try:
            before = pool.worker_pids()
            with pytest.raises(ZeroDivisionError):
                async with pool.slot():
                    1 / 0
            assert pool.counts() == {"idle": 1, "held": 0, "restarting": 0}
            assert pool.worker_pids() == before
        finally:
            await pool.close()
    run(main())


def test_dead_idle_worker_is_replaced_on_pop(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=2)
        try:
            victim = pool._idle[0]
            victim.process.kill()
            victim.process.join(5)
            outcome = await analyze(pool, job_file(tmp_path, "j"))  # served by the other worker
            assert outcome.model_version == "mock"
            await until(lambda: pool.counts()["idle"] == 2)
            assert victim.pid not in pool.worker_pids()
        finally:
            await pool.close()
    run(main())


def test_slot_never_released_twice(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=2)
        try:
            for command in (None, "error", "unavailable", "exit"):
                try:
                    await analyze(pool, job_file(tmp_path, f"j{command}", command))
                except (AnalyzerFailed, AnalyzerUnavailable, WorkerLost):
                    pass
                assert_invariant(pool)
                await until(lambda: pool.counts()["idle"] == 2)
                assert len(pool._idle) == len({id(w) for w in pool._idle})
            async with pool.slot() as job:
                pass
            with pytest.raises(RuntimeError, match="released twice"):
                pool._return(job.worker)
        finally:
            await pool.close()
    run(main())


# --- shutdown ------------------------------------------------------------------------------------

def test_close_leaves_no_worker_process(control, tmp_path):
    async def main():
        pool = await started_pool(control, size=2)
        busy = asyncio.create_task(analyze(pool, job_file(tmp_path, "h", "hang")))
        await until(lambda: len(started(control)) == 1)
        workers = [*pool._idle, *pool._held]
        await pool.close()
        with pytest.raises(asyncio.CancelledError):
            busy.cancel()
            await busy
        return workers
    workers = run(main())
    assert len(workers) == 2
    # Every worker is reaped. The resource tracker that spawn starts on POSIX is not a pool worker and
    # stays until this (test) process exits; it is not checked here.
    for worker in workers:
        assert_reaped(worker)


def assert_refused_without_a_child(control, pool) -> None:
    before = set(multiprocessing.active_children())
    with pytest.raises(PoolStartError) as refused:
        run(pool.start())
    assert str(refused.value) == NEEDS_SELECTOR_LOOP
    assert builds(control) == 0  # no worker was spawned
    assert set(multiprocessing.active_children()) <= before
    assert not pool.worker_pids() and pool.counts() == {"idle": 0, "held": 0, "restarting": 0}
    with pytest.raises(PoolUnavailable):
        pool.info()


def test_a_loop_without_add_reader_refuses_to_start_and_leaves_no_child(control, monkeypatch):
    """Windows' default Proactor loop has no add_reader: the pool refuses to start (the API does not start)."""
    def no_add_reader(self, fd, callback, *args):
        raise NotImplementedError

    monkeypatch.setattr(asyncio.SelectorEventLoop, "add_reader", no_add_reader)
    assert_refused_without_a_child(control, AnalysisPool(factory(control), size=2, start_timeout_s=30))


def test_windows_refuses_to_start_and_leaves_no_child(control, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert_refused_without_a_child(control, AnalysisPool(factory(control), size=2, start_timeout_s=30))
