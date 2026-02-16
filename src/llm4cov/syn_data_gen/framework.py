import asyncio
import concurrent.futures
import logging
import math
import time
import traceback
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Generic, TypeVar

from tqdm import tqdm

TItem = TypeVar("TItem")
TResult = TypeVar("TResult")

Workflow = Callable[["RunContext", TItem], Awaitable[TResult]]

LOG = logging.getLogger("pipeline")


# -------------------------
# Metrics / counters
# -------------------------
@dataclass
class PoolStats:
    waiting: int = 0  # queued at semaphore boundary
    running: int = 0  # holding permit and executing
    done: int = 0
    failed: int = 0


@dataclass
class PipelineStats:
    total: int
    started: int = 0
    finished: int = 0
    failed: int = 0

    llm: PoolStats = field(default_factory=PoolStats)
    eda: PoolStats = field(default_factory=PoolStats)

    _t0: float = field(default_factory=time.time)
    _last_eta: float | None = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def elapsed(self) -> float:
        return time.time() - self._t0

    def pct(self) -> float:
        return 0.0 if self.total == 0 else (self.finished / self.total) * 100.0

    def eta_seconds(self) -> float | None:
        if self.finished <= 0:
            return None
        rem = self.total - self.finished
        if rem <= 0:
            return 0.0
        raw = self.elapsed() * (rem / self.finished)
        self._last_eta = raw if self._last_eta is None else (0.7 * self._last_eta + 0.3 * raw)
        return self._last_eta


def fmt_time_s(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or math.isnan(seconds):
        return "?"
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# -------------------------
# Result container per item
# -------------------------
@dataclass
class ItemResult(Generic[TResult]):
    ok: bool
    value: TResult | None = None
    error: str | None = None


# -------------------------
# Semaphore wrapper that tracks waiting/running
# -------------------------
class TrackedSemaphore:
    def __init__(self, sem: asyncio.Semaphore, pool_stats: PoolStats, lock: asyncio.Lock):
        self._sem = sem
        self._pool_stats = pool_stats
        self._lock = lock

    async def __aenter__(self) -> "TrackedSemaphore":
        async with self._lock:
            self._pool_stats.waiting += 1

        await self._sem.acquire()

        async with self._lock:
            self._pool_stats.waiting -= 1
            self._pool_stats.running += 1

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        async with self._lock:
            self._pool_stats.running -= 1
        self._sem.release()


# -------------------------
# Public: RunContext used by workflows
# -------------------------
@dataclass
class RunContext:
    """
    Shared, reusable context passed into workflow(ctx, item).

    Your workflows should call:
      - await ctx.run_llm(async_fn, *args, **kwargs)
      - await ctx.run_eda(sync_fn, *args, **kwargs)

    Put shared objects (AsyncOpenAI client, tokenizer, paths, configs) in ctx.shared.
    """

    stats: PipelineStats
    llm_gate: TrackedSemaphore
    eda_gate: TrackedSemaphore
    eda_executor: concurrent.futures.Executor

    llm_timeout_s: float = 120.0
    eda_timeout_s: float = 180.0

    shared: dict[str, Any] = field(default_factory=dict)
    log_stage_timing: bool = False

    async def run_llm(
        self,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        timeout_s: float | None = None,
        label: str = "LLM",
        **kwargs: Any,
    ) -> Any:
        t0 = time.time()
        try:
            async with self.llm_gate:
                out = await asyncio.wait_for(
                    fn(*args, **kwargs),
                    timeout=self.llm_timeout_s if timeout_s is None else timeout_s,
                )
            async with self.stats.lock:
                self.stats.llm.done += 1
            return out
        except Exception:
            async with self.stats.lock:
                self.stats.llm.failed += 1
            raise
        finally:
            if self.log_stage_timing:
                LOG.debug("%s stage took %.2fs", label, time.time() - t0)

    async def run_eda(
        self,
        fn_sync: Callable[..., Any],
        *args: Any,
        timeout_s: float | None = None,
        label: str = "EDA",
        **kwargs: Any,
    ) -> Any:
        t0 = time.time()
        try:
            async with self.eda_gate:
                loop = asyncio.get_running_loop()
                fut = loop.run_in_executor(self.eda_executor, lambda: fn_sync(*args, **kwargs))
                out = await asyncio.wait_for(
                    fut,
                    timeout=self.eda_timeout_s if timeout_s is None else timeout_s,
                )
            async with self.stats.lock:
                self.stats.eda.done += 1
            return out
        except Exception:
            async with self.stats.lock:
                self.stats.eda.failed += 1
            raise
        finally:
            if self.log_stage_timing:
                LOG.debug("%s stage took %.2fs", label, time.time() - t0)


# -------------------------
# Internal: progress monitor
# -------------------------
async def _progress_monitor(stats: PipelineStats, pbar: tqdm, interval_s: float = 1.0) -> None:
    while True:
        await asyncio.sleep(interval_s)
        async with stats.lock:
            done = stats.finished
            postfix = {
                "elapsed": fmt_time_s(stats.elapsed()),
                "eta": fmt_time_s(stats.eta_seconds()),
                "done%": f"{stats.pct():.1f}",
                "LLM(w/r)": f"{stats.llm.waiting}/{stats.llm.running}",
                "EDA(w/r)": f"{stats.eda.waiting}/{stats.eda.running}",
                "fail": f"{stats.failed}",
            }
        pbar.set_postfix(postfix)
        if done >= stats.total:
            return


# -------------------------
# Internal: worker loop
# -------------------------
async def _worker_loop(
    wid: int,
    queue: asyncio.Queue[tuple[int, TItem] | None],
    results: list[ItemResult[TResult]],
    ctx: RunContext,
    workflow: Workflow[TItem, TResult],
    pbar: tqdm,
    pbar_lock: asyncio.Lock,
    *,
    include_traceback: bool,
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            return

        idx, item = job
        try:
            # Workflow runs end-to-end. If it raises, control plane records and continues.
            out = await workflow(ctx, item)
            results[idx] = ItemResult(ok=True, value=out, error=None)

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if include_traceback:
                msg = msg + "\n" + traceback.format_exc()

            results[idx] = ItemResult(ok=False, value=None, error=msg)

            async with ctx.stats.lock:
                ctx.stats.failed += 1

            LOG.warning("worker=%d idx=%d | item failed: %s", wid, idx, msg.splitlines()[0])

        finally:
            async with ctx.stats.lock:
                ctx.stats.finished += 1

            async with pbar_lock:
                pbar.update(1)

            queue.task_done()


# -------------------------
# Public: pipeline runner
# -------------------------
async def run_pipeline_queue_workers(
    items: Sequence[TItem],
    workflow: Workflow[TItem, TResult],
    *,
    orchestrator_workers: int = 512,
    llm_concurrency: int = 128,
    eda_concurrency: int = 8,
    llm_timeout_s: float = 120.0,
    eda_timeout_s: float = 180.0,
    shared: dict[str, Any] | None = None,
    show_progress: bool = True,
    include_traceback: bool = False,
    log_stage_timing: bool = False,
) -> tuple[list[ItemResult[TResult]], PipelineStats]:
    """
    Main entrypoint.

    Usage:
      results, stats = await run_pipeline_queue_workers(items, workflow=run_item_specified, ...)

    - items: preprocessed dataset items
    - workflow(ctx, item): your per-item logic (vanilla/react/eval)
    - shared: dict injected into ctx.shared (clients, tokenizers, paths, configs)
    - Failure policy: any exception from workflow is recorded and the run continues.
    """
    n = len(items)
    stats = PipelineStats(total=n)

    # Build gates and executor once (control plane)
    llm_gate = TrackedSemaphore(asyncio.Semaphore(llm_concurrency), stats.llm, stats.lock)
    eda_gate = TrackedSemaphore(asyncio.Semaphore(eda_concurrency), stats.eda, stats.lock)

    results: list[ItemResult[TResult]] = [
        ItemResult(ok=False, error="not_started") for _ in range(n)
    ]

    queue: asyncio.Queue[tuple[int, TItem] | None] = asyncio.Queue()
    for idx, item in enumerate(items):
        queue.put_nowait((idx, item))

    for _ in range(orchestrator_workers):
        queue.put_nowait(None)

    pbar = tqdm(total=n, desc="Pipeline", dynamic_ncols=True, disable=not show_progress)
    pbar_lock = asyncio.Lock()
    mon_task = (
        asyncio.create_task(_progress_monitor(stats, pbar, interval_s=1.0))
        if show_progress
        else None
    )

    async with stats.lock:
        stats.started = 0  # optional; you can increment started inside worker if you want

    # ThreadPoolExecutor is generally best here (EDA work is blocking subprocess waits).
    with concurrent.futures.ThreadPoolExecutor(max_workers=eda_concurrency) as eda_executor:
        ctx = RunContext(
            stats=stats,
            llm_gate=llm_gate,
            eda_gate=eda_gate,
            eda_executor=eda_executor,
            llm_timeout_s=llm_timeout_s,
            eda_timeout_s=eda_timeout_s,
            shared={} if shared is None else dict(shared),
            log_stage_timing=log_stage_timing,
        )

        workers = [
            asyncio.create_task(
                _worker_loop(
                    wid=i,
                    queue=queue,
                    results=results,
                    ctx=ctx,
                    workflow=workflow,
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                    include_traceback=include_traceback,
                )
            )
            for i in range(orchestrator_workers)
        ]

        await queue.join()

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    if mon_task is not None:
        await mon_task

    pbar.close()

    LOG.info(
        (
            "Done. total=%d finished=%d failed=%d | "
            "LLM done=%d failed=%d | "
            "EDA done=%d failed=%d | "
            "elapsed=%s"
        ),
        stats.total,
        stats.finished,
        stats.failed,
        stats.llm.done,
        stats.llm.failed,
        stats.eda.done,
        stats.eda.failed,
        fmt_time_s(stats.elapsed()),
    )

    return results, stats
