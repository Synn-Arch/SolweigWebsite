#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
"""N8-14 region executor: ONE bounded multicore owner + per-worker scratch.

Implements dossier 03 ("Parallel owner choice", "Faults and determinism")
and ARCHITECTURE.md ("Threading and cancellation") for the region scale.
Three decisions are OWNED HERE and are normative for the integrator:

THREAD-BUDGET RULE (exact)
    N := RuntimeOptions.threads_per_worker at workflow binding
    (RuntimeOptions already enforces 1 <= N <= cpu_budget <= total CPUs).
    The region owner creates exactly N-1 persistent background worker
    threads and treats the SUBMITTING thread as the Nth worker, so
    observed block concurrency (max in-flight blocks) is <= N regardless
    of region count, block count, dispatch size, or timestep count --
    no W*H blowup. N = 1 creates ZERO background threads and executes
    blocks directly in the calling thread (the dossier's "For H=1 use the
    same correct fast implementation without task overhead").

    Two consumer parallelism classes share this ONE owner (no per-backend
    pools, no nested parallelism):

    * ExecutionMode.BLOCK_FANOUT -- the consumer is single-threaded per
      block (the ISPC/native leaf C: one C entry, one thread). The owner
      provides the parallelism at BLOCK level: blocks of a region run
      concurrently on the N budget slots, each leaf call using exactly
      one thread. Total executing threads <= N.
    * ExecutionMode.SELF_PARALLEL -- the consumer owns the whole budget
      internally (the Numba B: prange over rows inside the kernel). The
      owner GRANTS the budget exclusively: it dispatches nothing
      concurrently (in-flight blocks == 1, pool workers idle-blocked on
      the queue and consume no CPU) while the consumer's Numba threading
      layer -- sized to the SAME N, never N*num_workers -- executes the
      block. Numba's threading layer is thus "under" the owner: same
      budget, granted rather than added.

    The two modes never overlap on one pool: sessions are exclusive
    (a second concurrent submission raises PoolSessionConflict) and a
    consumer that re-enters the owner from inside a block raises the same
    error (structural nested-parallelism guard). No OpenMP is configured
    and no ISPC task runtime exists (the native entry is single-thread
    per call), so nothing nests beneath either mode.

FORK POLICY (the routed fork-while-locked decision, review note N4 of
    n8_30_review_n8_10_loader.md)
    The owner is THREAD-only: it never forks and never execs. If the
    process forks anyway (worker-pool schedulers, user code), an
    ``os.register_at_fork(after_in_child=...)`` hook installed by this
    module poisons every live pool LOCK-FREE (plain attribute writes and
    dict clears -- the child is single-threaded by definition and the
    handler must not, and does not, acquire any lock, so the classic
    fork-while-registry-locked deadlock cannot occur). Any child-side use
    of an inherited pool raises ForkedRegionPool; the child RE-PREPARES
    per the N8-10 taxonomy -- construct a fresh pool / call shared_pool()
    again, which in the child returns a fresh instance because the fork
    hook cleared the table. This mirrors NativeHandle's pid-pinned
    StaleNativeHandle rule: parent pointers/threads are never reused
    across a fork. Forking while a session is ACTIVE is tolerated the
    same way (the hook does not touch queues or slots), and the parent's
    in-flight session is unaffected because poisoning happens only in
    the child's copy of the state.

REGISTRY-GROWTH BOUND (the routed note)
    The module-level pool table is keyed by ``(pid, budget)`` and hard-
    capped at MAX_POOL_BUDGET_SLOTS = 8 distinct budgets; requesting a
    9th distinct budget raises PoolRegistryBound LOUDLY (thread-budget
    misconfiguration must not silently create owners). Growth is bounded
    by distinct thread budgets, never by regions, timesteps, blocks or
    in-pool workers. Live threads are bounded by sum(N_i - 1) over live
    pools. For the N8-10 handle registry: region workers are THREADS
    sharing one pid, and NativeHandle.execute performs no registry
    writes, so region execution contributes ZERO handle-registry growth
    -- that registry grows only with (process count x workflow
    generations), which is process topology, not region decomposition.

CANCELLATION OWNERSHIP
    An admitted error in any block cancels the REGION deterministically:
    the cancellation flag stops dispatch of not-yet-started blocks
    (already-running blocks finish or fail naturally -- bounded drain =
    at most one block per budget slot), all workers are joined, and the
    exception PROPAGATED is the first error in CANONICAL ORDER: the
    lowest block index, and within a block produce-before-consume --
    exactly the error a serial loop would have raised first. Wall-clock
    completion order cannot change which error the caller sees. The
    original exception object is re-raised with its type intact
    (UnsupportedInput stays UnsupportedInput so the real dispatcher's
    fallback decision is unchanged; NativeHandleError subclasses stay
    loud and are never swallowed or converted into a fallback). On
    failure no report is returned and nothing is published -- the caller
    sees the exception, never a half-computed frame presented as success.

COMMON CALL BOUNDARY (ARCHITECTURE: the planner/pool never special-cases
    a backend)
    A consumer is any object with::

        mode: ExecutionMode            # declared parallelism class only
        produce(ctx: BlockContext)     # per-block decode/classify stage
        consume(payload, ctx: BlockContext)  # kernel + output-span write

    The executor imports no backend module, knows no backend name, and
    distinguishes consumers ONLY by the declared execution mode (a
    parallelism contract, not a backend identity). A/B/C plug in as
    data: the frozen oracle, the N8-12 Numba B control and the N8-10
    native handle all run through the same two calls. Per-block
    admission lives entirely inside produce/consume -- the executor adds
    only ONCE-PER-CALL structural checks (plan type, output dtype/shape)
    that run BEFORE any block is dispatched.

SCRATCH (dossier "Memory model")
    Microblock scratch is INDEPENDENT of dispatch size: ScratchSlot
    arenas are leased PER BLOCK from a fixed pool of exactly N slots
    (per-worker-bounded), allocate each named buffer ONCE on first use
    and reuse it for every later block/region; arena count and shapes
    depend on the execution block b and the consumer's name set only --
    never on region_blocks, region count or timestep count. Lease
    exclusivity is by construction (a slot is owned by exactly one
    executing thread between acquire/release), so workers can never
    alias each other's scratch; slot contents are UNDEFINED across
    leases (consumers must fully write before read -- pinned by a poison
    canary in the test suite).
"""
from __future__ import annotations

import contextlib
import enum
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .region_plan import RegionPlan

__all__ = [
    'ExecutionMode', 'BlockContext', 'RegionReport', 'RegionPool',
    'ForkedRegionPool', 'ClosedRegionPool', 'PoolSessionConflict',
    'PoolRegistryBound', 'ScratchShapeError', 'ScratchSlot',
    'execute_regions', 'execute_serial', 'shared_pool', 'resolve_budget',
    'shutdown_all_pools', 'reset_pools_for_tests', 'MAX_POOL_BUDGET_SLOTS',
]

#: Hard cap on distinct thread budgets with a live shared pool (see the
#: registry-growth bound above). A 9th distinct budget is a configuration
#: error and fails loudly instead of spawning more owners.
MAX_POOL_BUDGET_SLOTS = 8


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RegionExecutorError(RuntimeError):
    """Base class of the region-owner machinery (never a fallback signal)."""


class ForkedRegionPool(RegionExecutorError):
    """A pool inherited across os.fork was used in the child.

    The parent's worker threads do not exist in the child; re-prepare --
    build a fresh RegionPool (or call shared_pool() again in the child) --
    exactly like NativeHandle's StaleNativeHandle re-prepare rule."""


class ClosedRegionPool(RegionExecutorError):
    """The pool was closed; build (or shared_pool()) a new one."""


class PoolSessionConflict(RegionExecutorError):
    """A second submission hit an owner with an active session.

    This is the structural nested-parallelism guard: one owner executes
    ONE region submission at a time, so block-fanout never stacks on a
    self-parallel consumer and never nests inside itself."""


class PoolRegistryBound(RegionExecutorError):
    """More than MAX_POOL_BUDGET_SLOTS distinct thread budgets requested."""


class ScratchShapeError(ValueError):
    """A scratch buffer name was requested with a different shape/dtype.

    Slots are reuse arenas, not growable caches: re-binding a name to a
    new shape would silently reallocate per block (malloc churn) or alias
    a stale layout, so it is rejected loudly."""


# ---------------------------------------------------------------------------
# Consumer contract
# ---------------------------------------------------------------------------

class ExecutionMode(enum.Enum):
    """Declared parallelism class of a consumer (not a backend identity)."""

    BLOCK_FANOUT = 'block_fanout'      # single-thread leaf; owner fans blocks out
    SELF_PARALLEL = 'self_parallel'    # consumer owns the whole budget internally


@dataclass(frozen=True)
class BlockContext:
    """Everything one block execution sees.

    ``output`` is the shared float32 [7, total_rows] frame; the consumer
    writes ONLY its own span ``output[:, start:stop]`` (blocks are
    independent, which is what makes parallel composition bitwise-exact).
    ``slot`` is this block's exclusive scratch lease. ``state`` is the
    caller's immutable shared payload (geometry lease / coefficients),
    passed through untouched. ``cancelled()`` lets long consumers exit
    cooperatively; it never changes results, only saves work.

    Consumers lease scratch at FULL block capacity (``block_capacity`` =
    plan.block_pixels) and use ``[:rows]`` views for the tail block,
    keeping every arena fixed-shape (see ScratchShapeError) regardless
    of dispatch size or tail position.
    """

    index: int                 # global block index (ascending serial order)
    start: int                 # first pixel of the block
    stop: int                  # one past the last pixel
    region_index: int
    block_capacity: int        # plan.block_pixels: the full-block row count
    slot: 'ScratchSlot'
    state: Mapping[str, Any]
    output: np.ndarray
    cancelled: Callable[[], bool]


def _check_consumer(consumer: Any) -> tuple[ExecutionMode, Callable, Callable]:
    """Validate the common call boundary (names only -- never a backend)."""
    mode = getattr(consumer, 'mode', None)
    if not isinstance(mode, ExecutionMode):
        raise TypeError(
            'consumer must expose mode: ExecutionMode (got '
            f'{mode!r}); the region owner only accepts the documented '
            'produce/consume contract')
    produce = getattr(consumer, 'produce', None)
    consume = getattr(consumer, 'consume', None)
    if not callable(produce) or not callable(consume):
        raise TypeError(
            'consumer must expose callable produce(ctx) and '
            'consume(payload, ctx)')
    return mode, produce, consume


# ---------------------------------------------------------------------------
# Scratch: per-block lease of per-worker-bounded arenas
# ---------------------------------------------------------------------------

class ScratchSlot:
    """One exclusively-leased scratch arena (reuse-friendly, bounded).

    ``buffer(name, shape, dtype)`` allocates each named buffer once and
    returns the SAME memory for every later request with the same name,
    shape and dtype. Slot contents are undefined across leases.
    """

    __slots__ = ('id', '_buffers', '_allocations')

    def __init__(self, slot_id: int) -> None:
        self.id = slot_id
        self._buffers: dict[str, np.ndarray] = {}
        self._allocations = 0

    def buffer(self, name: str, shape, dtype) -> np.ndarray:
        buf = self._buffers.get(name)
        if buf is None:
            buf = np.empty(shape, dtype=dtype)
            self._buffers[name] = buf
            self._allocations += 1
            return buf
        if buf.shape != tuple(shape) or buf.dtype != np.dtype(dtype):
            raise ScratchShapeError(
                f'scratch buffer {name!r} re-requested as '
                f'{tuple(shape)}/{np.dtype(dtype)} but bound to '
                f'{buf.shape}/{buf.dtype}; slots are fixed-shape reuse '
                f'arenas -- use a distinct name per shape')
        return buf

    @property
    def allocation_count(self) -> int:
        return self._allocations

    def buffer_names(self) -> tuple[str, ...]:
        return tuple(self._buffers)

    def release_buffers(self) -> None:
        """Drop arena references (pool close / child poison path)."""
        self._buffers.clear()


class _SlotPool:
    """Exactly ``count`` slots; acquire blocks until one is free.

    With count == the thread budget there is always a slot for every
    executing thread, so acquire never actually waits in practice; the
    condition keeps that guarantee true by construction rather than hope.
    """

    def __init__(self, count: int) -> None:
        self._lock = threading.Lock()
        self._free = threading.Condition(self._lock)
        self._all: tuple[ScratchSlot, ...] = tuple(
            ScratchSlot(i) for i in range(count))
        self._free_list: list[ScratchSlot] = list(self._all)
        self._leased: set[int] = set()

    def acquire(self) -> ScratchSlot:
        with self._free:
            while not self._free_list:
                self._free.wait()
            slot = self._free_list.pop()
            self._leased.add(slot.id)
            return slot

    def release(self, slot: ScratchSlot) -> None:
        with self._free:
            self._leased.discard(slot.id)
            self._free_list.append(slot)
            self._free.notify()

    @property
    def leased_ids(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._leased)

    def buffer_allocations(self) -> int:
        """Total one-time buffer allocations across all slots.

        Each slot's counter is a plain int read (diagnostics only); the
        authoritative per-slot numbers are exact because only the
        leasing thread mutates a slot.
        """
        return sum(slot.allocation_count for slot in self._all)

    def stats(self) -> dict:
        with self._lock:
            leased = frozenset(self._leased)
            free = len(self._free_list)
        return {
            'slot_count': len(self._all),
            'slots_free': free,
            'slots_leased': sorted(leased),
            'buffer_allocations': self.buffer_allocations(),
        }


# ---------------------------------------------------------------------------
# The pool owner
# ---------------------------------------------------------------------------

class _FanoutSession:
    """One region's block fanout: shared counters under the pool lock."""

    def __init__(self, pool: 'RegionPool', region_index: int,
                 blocks: list[tuple[int, int, int]],
                 produce: Callable, consume: Callable,
                 output: np.ndarray, state: Mapping[str, Any],
                 block_capacity: int):
        self.pool = pool
        self.region_index = region_index
        self.block_capacity = block_capacity
        self.blocks = blocks
        self.produce = produce
        self.consume = consume
        self.output = output
        self.state = state
        self.lock = pool._lock
        self.errors: dict[int, BaseException] = {}
        self.started: set[int] = set()
        self.cancelled = False
        self.outstanding = len(blocks)
        self.dequeued_after_cancel = 0
        self.in_flight = 0
        self.max_in_flight = 0

    def is_cancelled(self) -> bool:
        with self.lock:
            return self.cancelled

    def handle(self, item: tuple[int, int, int]) -> None:
        """Run or skip one dequeued block (worker or submitting thread)."""
        index, start, stop = item
        run = False
        with self.lock:
            if self.cancelled:
                self.dequeued_after_cancel += 1
            else:
                run = True
                self.started.add(index)
                self.in_flight += 1
                if self.in_flight > self.max_in_flight:
                    self.max_in_flight = self.in_flight
        if not run:
            self._retire()
            return
        slot = self.pool._slots.acquire()
        try:
            ctx = BlockContext(index=index, start=start, stop=stop,
                               region_index=self.region_index,
                               block_capacity=self.block_capacity,
                               slot=slot, state=self.state,
                               output=self.output,
                               cancelled=self.is_cancelled)
            payload = self.produce(ctx)
            self.consume(payload, ctx)
        except BaseException as exc:  # record everything; first-index wins
            with self.lock:
                self.errors.setdefault(index, exc)
                self.cancelled = True
        finally:
            self.pool._slots.release(slot)
            with self.lock:
                self.in_flight -= 1
            self._retire()

    def _retire(self) -> None:
        with self.lock:
            self.outstanding -= 1
            done = self.outstanding == 0
        if done:
            with self.pool._idle:
                self.pool._idle.notify_all()

    def run(self) -> None:
        pool = self.pool
        pool._ensure_workers()
        for item in self.blocks:
            pool._queue.put(item)
        # The submitting thread is the Nth worker (thread-budget rule):
        # it works the same queue until every block is retired.
        while True:
            with self.lock:
                if self.outstanding == 0:
                    break
            try:
                item = pool._queue.get(timeout=0.02)
            except queue.Empty:
                continue
            self.handle(item)


class RegionPool:
    """The ONE bounded multicore owner for region execution.

    Persistent background workers (N-1) are created lazily on the first
    BLOCK_FANOUT session and live until close(); SELF_PARALLEL sessions
    use none. The pool owns the N scratch slots (see SCRATCH above).
    """

    def __init__(self, budget: int, *, name: str | None = None):
        self._budget = resolve_budget(budget)
        self._name = name or f'solweig-region-{self._budget}'
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._workers: list[threading.Thread] = []
        self._session: _FanoutSession | None = None
        self._active_sessions = 0
        self._closing = False
        self._closed = False
        self._poisoned = False
        self._fork_generation = _FORK_GENERATION
        self._slots = _SlotPool(self._budget)
        self._executions = 0
        _track_pool(self)   # fork poisoning covers explicit pools too

    # -- identity / diagnostics -------------------------------------------

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def worker_count(self) -> int:
        with self._lock:
            return len(self._workers)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def fork_generation(self) -> int:
        return self._fork_generation

    def scratch_stats(self) -> dict:
        """Slot/buffer accounting (bounded by budget, never by blocks)."""
        return self._slots.stats()

    def describe(self) -> dict:
        return {
            'name': self._name,
            'budget': self._budget,
            'workers': self.worker_count,
            'closed': self.closed,
            'poisoned': self._poisoned,
            'fork_generation': self._fork_generation,
            'executions': self._executions,
            'scratch': self._slots.stats(),
        }

    # -- lifecycle ---------------------------------------------------------

    def _ensure_workers(self) -> None:
        """Spawn the N-1 persistent workers exactly once (budget rule)."""
        with self._lock:
            if self._closed:
                raise ClosedRegionPool('pool is closed')
            if self._poisoned:
                raise ForkedRegionPool(
                    'pool inherited across fork; re-prepare in the child '
                    '(build a fresh RegionPool / shared_pool())')
            while len(self._workers) < self._budget - 1:
                thread = threading.Thread(
                    target=self._worker_main,
                    name=f'{self._name}-w{len(self._workers)}',
                    daemon=True)
                self._workers.append(thread)
                thread.start()

    def _worker_main(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            session = self._session
            if session is not None:
                session.handle(item)

    def close(self) -> None:
        """Join all workers and drop scratch arenas (idempotent).

        Must not race an active session (the join could never complete);
        that misuse raises instead of hanging.
        """
        with self._lock:
            if self._closed:
                return
            if self._session is not None or self._active_sessions:
                raise RegionExecutorError(
                    'cannot close a pool with an active region session')
            self._closing = True
            workers = list(self._workers)
        for _ in workers:
            self._queue.put(None)
        for thread in workers:
            thread.join()
        with self._lock:
            self._closed = True
            self._workers.clear()
        _forget_pool(self)

    # -- execution ---------------------------------------------------------

    @contextlib.contextmanager
    def _session_guard(self, mode: ExecutionMode):
        with self._lock:
            if self._poisoned:
                raise ForkedRegionPool(
                    'pool inherited across fork; re-prepare in the child '
                    '(build a fresh RegionPool / shared_pool())')
            if self._closed or self._closing:
                raise ClosedRegionPool('pool is closed; build a new one')
            if self._session is not None or self._active_sessions:
                raise PoolSessionConflict(
                    'this owner already has an active region session; one '
                    'owner executes one submission at a time (no nested '
                    'parallelism, no second owner)')
            self._active_sessions += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_sessions -= 1

    def execute_regions(self, plan: RegionPlan, consumer: Any,
                        output: np.ndarray, *,
                        state: Mapping[str, Any] | None = None
                        ) -> 'RegionReport':
        mode, produce, consume = _check_consumer(consumer)
        _validate_output(plan, output)
        state = state if state is not None else {}
        totals = {'regions': 0, 'blocks': 0, 'started': 0, 'skipped': 0,
                  'max_in_flight': 0}
        blocks = plan.block_spans()
        with self._session_guard(mode):
            for region_index, rng in enumerate(plan.region_block_ranges()):
                region_blocks = [(i, blocks[i][0], blocks[i][1])
                                 for i in rng]
                totals['regions'] += 1
                totals['blocks'] += len(region_blocks)
                if mode is ExecutionMode.BLOCK_FANOUT:
                    session = _FanoutSession(
                        self, region_index, region_blocks, produce, consume,
                        output, state, plan.block_pixels)
                    with self._lock:
                        if self._session is not None:
                            raise PoolSessionConflict(
                                'concurrent fanout session on one owner')
                        self._session = session
                    try:
                        session.run()
                    finally:
                        with self._lock:
                            self._session = None
                            self._idle.notify_all()
                    totals['started'] += len(session.started)
                    totals['skipped'] += session.dequeued_after_cancel
                    totals['max_in_flight'] = max(
                        totals['max_in_flight'], session.max_in_flight)
                    if session.errors:
                        _raise_first(session.errors, region_index, blocks,
                                     started=tuple(sorted(session.started)))
                else:
                    started_i, skipped, errors = self._run_self_parallel(
                        region_index, region_blocks, produce, consume,
                        output, state, plan.block_pixels)
                    totals['started'] += len(started_i)
                    totals['skipped'] += skipped
                    totals['max_in_flight'] = max(totals['max_in_flight'], 1)
                    if errors:
                        _raise_first(errors, region_index, blocks,
                                     started=tuple(started_i))
        with self._lock:
            self._executions += 1
        return RegionReport(
            mode=mode.value, budget=self._budget,
            pool_workers=self.worker_count, pool=self._name,
            regions=totals['regions'], blocks=totals['blocks'],
            blocks_started=totals['started'],
            blocks_skipped_after_cancel=totals['skipped'],
            max_in_flight=totals['max_in_flight'],
            slot_buffer_allocations=self._slots.buffer_allocations(),
        )

    def _run_self_parallel(self, region_index: int,
                           region_blocks: list[tuple[int, int, int]],
                           produce: Callable, consume: Callable,
                           output: np.ndarray, state: Mapping[str, Any],
                           block_capacity: int
                           ) -> tuple[int, int, dict[int, BaseException]]:
        """SELF_PARALLEL: blocks run sequentially in the calling thread.

        The budget is granted to the consumer's own threading layer; the
        owner dispatches nothing concurrently (in-flight <= 1). Error
        semantics match serial: stop at the first failing block.
        """
        errors: dict[int, BaseException] = {}
        cancelled = False
        started: list[int] = []
        skipped = 0
        for index, start, stop in region_blocks:
            if cancelled:
                skipped += 1
                continue
            started.append(index)
            slot = self._slots.acquire()
            try:
                ctx = BlockContext(
                    index=index, start=start, stop=stop,
                    region_index=region_index,
                    block_capacity=block_capacity, slot=slot, state=state,
                    output=output, cancelled=lambda: cancelled)
                payload = produce(ctx)
                consume(payload, ctx)
            except BaseException as exc:
                errors.setdefault(index, exc)
                cancelled = True
            finally:
                self._slots.release(slot)
        return started, skipped, errors


@dataclass(frozen=True)
class RegionReport:
    """What one execute_regions call did (functional counts, no timing)."""

    mode: str
    budget: int
    pool_workers: int
    pool: str
    regions: int
    blocks: int
    blocks_started: int
    blocks_skipped_after_cancel: int
    max_in_flight: int
    slot_buffer_allocations: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _raise_first(errors: Mapping[int, BaseException], region_index: int,
                 blocks: tuple, *, started_after_cancel: int = 0,
                 started: tuple[int, ...] = ()) -> None:
    """Re-raise the FIRST error in canonical order (lowest block index).

    The original exception object is re-raised with its type intact; a
    note records the region/block for diagnosis without changing the
    class (the real dispatcher's UnsupportedInput fallback decision and
    NativeHandleError loudness depend on the type passing through).
    A ``_solweig_region_failure`` attribute carries the mechanical facts
    (failing indices, blocks started, starts after cancellation) for
    tests and evidence; it is best-effort because a few exception types
    forbid attribute writes.
    """
    index = min(errors)
    exc = errors[index]
    start, stop = blocks[index]
    note = (f'solweig region: block {index} [{start}:{stop}] in region '
            f'{region_index} failed first in canonical order '
            f'({len(errors)} failing block(s)); region cancelled')
    add_note = getattr(exc, 'add_note', None)
    if callable(add_note):
        add_note(note)
    try:
        exc._solweig_region_failure = {
            'first_failing_block': index,
            'failing_blocks': tuple(sorted(errors)),
            'region_index': region_index,
            'blocks_started': tuple(started),
            'started_after_cancel': started_after_cancel,
        }
    except (AttributeError, TypeError):  # pragma: no cover
        pass
    raise exc


def _validate_output(plan: RegionPlan, output: Any) -> None:
    """Once-per-call structural checks; run BEFORE any block dispatch."""
    if not isinstance(plan, RegionPlan):
        raise TypeError(f'plan must be a RegionPlan; got {type(plan)!r}')
    if not isinstance(output, np.ndarray):
        raise TypeError(
            'output must be a float32 ndarray [7, total_rows]; got '
            f'{type(output)!r}')
    if output.dtype != np.float32:
        raise ValueError(f'output dtype must be float32; got {output.dtype}')
    if output.shape != (7, plan.total_rows):
        raise ValueError(
            f'output shape {output.shape} does not match the plan '
            f'(7, {plan.total_rows})')


# ---------------------------------------------------------------------------
# Budget resolution + shared-pool registry (bounded)
# ---------------------------------------------------------------------------

def resolve_budget(budget: int | None = None) -> int:
    """Resolve the owner's thread budget N.

    Explicit values are validated like RuntimeOptions.threads_per_worker
    (positive, <= total CPUs); None reads the CURRENT runtime options
    snapshot's threads_per_worker -- the existing default thread budget,
    which RuntimeOptions already constrains to <= cpu_budget <= CPUs.
    """
    if budget is not None:
        if (isinstance(budget, bool) or not isinstance(budget, int)
                or budget < 1):
            raise ValueError(
                f'budget must be a positive integer; got {budget!r}')
        cpus = os.cpu_count() or 1
        if budget > cpus:
            raise ValueError(f'budget={budget} exceeds {cpus} available CPUs')
        return int(budget)
    from solweig_light.runtime import get_runtime_options
    return int(get_runtime_options().threads_per_worker)


#: RLock because shared_pool constructs RegionPool while holding it and
#: the constructor registers itself for fork poisoning. The fork hook
#: NEVER takes this lock (see _poison_child_pools).
_POOLS_LOCK = threading.RLock()
_POOLS: dict[tuple[int, int], RegionPool] = {}
_LIVE_POOLS: list[RegionPool] = []
_FORK_GENERATION = 0

#: Bound on pools tracked for fork poisoning. Closed/poisoned entries are
#: pruned first; beyond that, tracking new pools would itself be the
#: thread-budget misconfiguration the loud caps exist for.
_MAX_TRACKED_POOLS = 64


def _track_pool(pool: RegionPool) -> None:
    with _POOLS_LOCK:
        if any(p is pool for p in _LIVE_POOLS):
            return
        if len(_LIVE_POOLS) >= _MAX_TRACKED_POOLS:
            _LIVE_POOLS[:] = [p for p in _LIVE_POOLS
                              if not p.closed and not p.poisoned]
        if len(_LIVE_POOLS) < _MAX_TRACKED_POOLS:
            _LIVE_POOLS.append(pool)


def shared_pool(budget: int | None = None) -> RegionPool:
    """THE shared owner for a thread budget (one per (pid, budget)).

    Registry-growth bound: at most MAX_POOL_BUDGET_SLOTS distinct live
    budgets are tracked; a 9th raises PoolRegistryBound. Closed or
    poisoned entries are replaced in place (no growth).
    """
    resolved = resolve_budget(budget)
    key = (os.getpid(), resolved)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is not None and (pool.closed or pool.poisoned):
            _POOLS.pop(key, None)
            pool = None
        if pool is None:
            live = sum(1 for p in _POOLS.values()
                       if not p.closed and not p.poisoned)
            if live >= MAX_POOL_BUDGET_SLOTS:
                raise PoolRegistryBound(
                    f'more than {MAX_POOL_BUDGET_SLOTS} distinct live '
                    f'thread budgets requested; the region owner table '
                    f'is intentionally bounded -- reuse shared_pool(N)')
            pool = RegionPool(resolved)
            _POOLS[key] = pool
            # NOTE: no _LIVE_POOLS append here -- RegionPool.__init__
            # already registered the pool through _track_pool (the one
            # dedup-guarded registration point). A second append left a
            # duplicate that survived close()'s single remove.
        return pool


def _forget_pool(pool: RegionPool) -> None:
    with _POOLS_LOCK:
        for key, entry in list(_POOLS.items()):
            if entry is pool:
                _POOLS.pop(key, None)
        _LIVE_POOLS[:] = [p for p in _LIVE_POOLS if p is not pool]


def shutdown_all_pools() -> None:
    """Close every live tracked pool (test teardown / workflow end).

    close() untracks each pool; anything that failed to close (e.g. an
    active session refuses close and stays genuinely live) keeps its
    entry, and closed/poisoned leftovers are pruned so _LIVE_POOLS
    reflects LIVE pools only.
    """
    with _POOLS_LOCK:
        pools = list(_LIVE_POOLS)
    for pool in pools:
        with contextlib.suppress(Exception):
            pool.close()
    with _POOLS_LOCK:
        _LIVE_POOLS[:] = [p for p in _LIVE_POOLS
                          if not p.closed and not p.poisoned]


def reset_pools_for_tests() -> None:
    shutdown_all_pools()
    with _POOLS_LOCK:
        _POOLS.clear()
        _LIVE_POOLS.clear()


def _poison_child_pools() -> None:
    """os.register_at_fork(after_in_child=...) hook -- LOCK-FREE by design.

    The child is single-threaded; taking any lock here (pool lock, POOLS
    lock) could deadlock on a lock held by another parent thread at the
    fork instant. We only do plain attribute writes and container
    clears, so the fork-while-locked edge cannot hang the child.
    """
    global _FORK_GENERATION
    _FORK_GENERATION += 1
    _POOLS.clear()
    for pool in _LIVE_POOLS:
        pool._poisoned = True
    _LIVE_POOLS.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_poison_child_pools)


# ---------------------------------------------------------------------------
# Public entries
# ---------------------------------------------------------------------------

def execute_regions(plan: RegionPlan, consumer: Any, output: np.ndarray, *,
                    pool: RegionPool | None = None,
                    budget: int | None = None,
                    state: Mapping[str, Any] | None = None) -> RegionReport:
    """Execute a plan's regions through ONE bounded owner.

    ``pool`` selects the owner explicitly (recommended); otherwise the
    shared owner for ``budget`` (or the current runtime default budget)
    is used. See the module docstring for the budget/fork/registry
    policies and the cancellation semantics.
    """
    if pool is None:
        pool = shared_pool(budget)
    return pool.execute_regions(plan, consumer, output, state=state)


def execute_serial(plan: RegionPlan, consumer: Any, output: np.ndarray, *,
                   state: Mapping[str, Any] | None = None) -> RegionReport:
    """The serial composition reference (no pool, no threads).

    Runs the SAME produce/consume contract over the SAME block spans in
    ascending order with one scratch slot, raising the first error
    immediately -- this is the semantics parallel execution must match
    bitwise (composition equality) and in error identity.
    """
    mode, produce, consume = _check_consumer(consumer)
    _validate_output(plan, output)
    state = state if state is not None else {}
    slot = ScratchSlot(0)
    blocks = plan.block_spans()
    started = 0
    for region_index, rng in enumerate(plan.region_block_ranges()):
        for index in rng:
            start, stop = blocks[index]
            started += 1
            ctx = BlockContext(
                index=index, start=start, stop=stop,
                region_index=region_index, block_capacity=plan.block_pixels,
                slot=slot, state=state,
                output=output, cancelled=lambda: False)
            payload = produce(ctx)
            consume(payload, ctx)
    return RegionReport(
        mode=mode.value, budget=1, pool_workers=0, pool='serial',
        regions=plan.region_count, blocks=plan.total_blocks,
        blocks_started=started, blocks_skipped_after_cancel=0,
        max_in_flight=1 if started else 0,
        slot_buffer_allocations=slot.allocation_count,
    )
