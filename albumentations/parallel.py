"""Deterministic, order-independent parallel augmentation.

This module adds a *new entry point* :func:`parallel_augment` that runs an existing
:class:`~albumentations.core.composition.Compose` pipeline over many images in a pool
of worker processes, while guaranteeing that the result for each image depends only on
the image identifier and the global seed -- never on the processing order, the worker
the image happens to land on, or the configured parallelism.

The single-process call ``compose(**data)`` is intentionally left untouched. This
module does not modify any transform or any numerical implementation; it only controls
*when* and *with which random state* the existing pipeline is invoked.

Randomness model (background)
-----------------------------
Every stochastic decision inside albumentations is drawn from exactly two streams held
on each transform/composition instance:

* ``self.random_generator`` -- a :class:`numpy.random.Generator`
* ``self.py_random`` -- a :class:`random.Random`

All transforms of a pipeline share the same two stream objects (they are propagated by
:meth:`BaseCompose.set_random_state`). Local generators created inside some transforms
(e.g. ``CoarseDropout``, ``ISONoise``) are themselves seeded from these two streams, so
controlling the two streams controls every random decision.

Per-image context derivation (the documented rule)
--------------------------------------------------
Before processing image ``image_id`` under global seed ``seed`` both streams are
rebuilt from a 32-bit per-image seed derived as follows (no step depends on order,
worker identity or parallelism)::

    key     = "albumentations.parallel.v1" || byte 0x1F || utf8(str(seed))
              || byte 0x1F || utf8(str(image_id))
    digest  = blake2b(key, digest_size=8)          # 64-bit BLAKE2b hash
    img_seed = int.from_bytes(digest, "big")        # 0 .. 2**64 - 1
    rng     = numpy.random.default_rng(img_seed)
    pyr     = random.Random(img_seed)

``img_seed`` is therefore a pure deterministic function of ``(seed, image_id)``.
The same image always receives the identical pair of streams, in every process, in
every run, regardless of which worker handles it or how the input is shuffled. The
derivation never uses "one fixed seed per process".

The same formula is exposed as :func:`derive_image_seed` so callers can audit or
reuse it.
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing
import os
import random
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Union

import numpy as np

from .core.composition import BaseCompose

__all__ = [
    "AugmentationFailure",
    "ParallelAugmentationResult",
    "benchmark_parallel_augment",
    "derive_image_seed",
    "parallel_augment",
]

# Namespace prefix for the derivation hash. Bumped only if the derivation rule ever
# changes; keeping it fixed makes the contract stable and explicit.
_DERIVATION_DOMAIN = b"albumentations.parallel.v1"
_SEED_DOMAIN = 2**32  # derive_image_seed exposes a 32-bit value for convenience
# Byte separating domain/seed/id fields in the derivation hash key (unit separator).
_FIELD_SEPARATOR = b"\x1f"


# Sentinel raised inside workers for image-read failures, so they can be distinguished
# from transform failures in the failure records.
class _ReadError(Exception):
    pass


@dataclass
class AugmentationFailure:
    """A single item that could not be augmented.

    Attributes:
        index (int): Position of the item in the input sequence (always valid, even if
            the item had no explicit identifier).
        image_id (Any): Identifier of the item, or its positional index when none was
            given.
        stage (str): ``"read"`` when the ``reader`` call raised, ``"transform"`` when
            the augmentation pipeline raised.
        error_type (str): Class name of the underlying exception.
        message (str): String representation of the underlying exception.
        attempts (int): Number of attempts made (a crashed worker triggers one retry;
            ordinary item-level errors are not retried).

    """

    index: int
    image_id: Any
    stage: str
    error_type: str
    message: str
    attempts: int


@dataclass
class ParallelAugmentationResult:
    """Outcome of a :func:`parallel_augment` run.

    Attributes:
        outputs (list[dict[str, Any] | None]): Augmented data dictionaries, in the same
            order as the input. Entries for failed items are ``None``; see
            :attr:`failures`.
        failures (list[AugmentationFailure]): Every item that was skipped after its
            retries were exhausted.
        num_succeeded (int): Number of successfully augmented items.
        num_read_failures (int): Number of failures raised by the ``reader`` callable.
        num_transform_failures (int): Number of failures raised inside the pipeline.
        num_worker_retries (int): Number of chunks that were resubmitted because a
            worker process died (``BrokenProcessPool``).
        num_workers (int): Parallelism actually used.
        elapsed_seconds (float): Wall-clock duration of the run.
        throughput (float): ``num_succeeded / elapsed_seconds`` items per second.
        seed (int | None): Global seed used for per-image context derivation.

    """

    outputs: list[dict[str, Any] | None]
    failures: list[AugmentationFailure]
    num_succeeded: int
    num_read_failures: int
    num_transform_failures: int
    num_worker_retries: int
    num_workers: int
    elapsed_seconds: float
    throughput: float
    seed: int | None

    @property
    def num_failed(self) -> int:
        """Number of items that could not be augmented."""
        return len(self.failures)


def _coerce_image_id(image_id: Any) -> str:
    """Normalize an identifier to its canonical string form used in the hash key."""
    return str(image_id)


def derive_image_seed(seed: int, image_id: Any) -> int:
    """Derive the deterministic 32-bit per-image seed.

    Implements the rule documented in the module docstring::

        key     = "albumentations.parallel.v1" || 0x1F || str(seed) || 0x1F
                  || str(image_id)
        img_seed = int(blake2b(utf8(key), digest_size=8)) mod 2**32

    The result depends solely on ``seed`` and ``image_id``: it is identical in every
    process, every run and every ordering, and does not use worker identity.

    Args:
        seed (int): Global seed. Non-negative integers are accepted; larger values are
            reduced modulo ``2**32`` so the same logical seed always maps to the same
            key material.
        image_id (Any): Image identifier. Its ``str()`` representation participates in
            the hash, so stable identifiers (paths, dataset indices, UUIDs) are
            required for reproducibility.

    Returns:
        int: A seed in ``[0, 2**32 - 1]``.

    """
    seed_key = str(int(seed) % _SEED_DOMAIN)
    key = _FIELD_SEPARATOR.join(
        [
            _DERIVATION_DOMAIN,
            seed_key.encode("utf-8"),
            _coerce_image_id(image_id).encode("utf-8"),
        ],
    )
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_DOMAIN


def _build_streams(seed: int | None, image_id: Any) -> tuple[np.random.Generator, random.Random]:
    """Build the two RNG streams for one image.

    A ``None`` seed still derives deterministic streams from the literal ``"None"``;
    callers that want reproducible results should always pass an explicit seed.
    """
    seed_key = "None" if seed is None else str(int(seed) % _SEED_DOMAIN)
    key = _FIELD_SEPARATOR.join(
        [
            _DERIVATION_DOMAIN,
            seed_key.encode("utf-8"),
            _coerce_image_id(image_id).encode("utf-8"),
        ],
    )
    img_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return np.random.default_rng(img_seed), random.Random(img_seed)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

Item = Union[np.ndarray, tuple[Any, dict[str, Any]], dict[str, Any]]
NormalizedItem = tuple[Any, dict[str, Any] | None]


def _normalize_items(items: Sequence[Item] | Iterable[Item]) -> list[NormalizedItem]:
    """Normalize the accepted input shapes into ``(image_id, data_or_none)`` pairs.

    Accepted element shapes:
        * a data ``dict`` (e.g. ``{"image": arr, "mask": mask}``) -- the positional
          index becomes the identifier;
        * a ``(image_id, data_dict)`` tuple -- explicit identifier;
        * a raw :class:`numpy.ndarray` -- treated as ``{"image": array}`` and the
          positional index becomes the identifier.
    """
    normalized: list[NormalizedItem] = []
    for index, item in enumerate(items):
        if isinstance(item, tuple):
            if len(item) != 2 or not (isinstance(item[1], dict) or item[1] is None):
                raise ValueError(
                    "Tuple items must be (image_id, data_dict) or (image_id, None) "
                    f"when a reader is used; got a malformed tuple at index {index}.",
                )
            normalized.append((item[0], item[1]))
        elif isinstance(item, dict):
            if "image_id" in item:
                normalized.append((item["image_id"], {k: v for k, v in item.items() if k != "image_id"}))
            else:
                normalized.append((index, item))
        elif isinstance(item, np.ndarray):
            normalized.append((index, {"image": item}))
        else:
            raise TypeError(
                f"Unsupported item type at index {index}: {type(item).__name__}. "
                "Pass a data dict, an (image_id, data_dict) tuple, or a numpy array.",
            )
    return normalized


def _validate_num_workers(num_workers: Any) -> None:
    """Reject illegal parallelism at startup, before any pool is created."""
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError(f"num_workers must be an int, got {type(num_workers).__name__}.")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}.")
    if num_workers > os.cpu_count() * 4:
        raise ValueError(
            f"num_workers={num_workers} is more than 4x the detected CPU count ({os.cpu_count()}); refusing to start.",
        )


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

# Process-global state, populated once per worker by ``_init_worker``.
_WORKER_STATE: dict[str, Any] = {}


def _init_worker(transform: BaseCompose, seed: int | None, reader: Callable[..., Any] | None) -> None:
    """Initializer run once when each worker process starts.

    The pipeline is deep-copied inside the worker so that the caller's instance is
    never mutated and workers never share mutable transform state. The copy's
    ``_base_seed`` is cleared so Compose's torch-DataLoader worker-seed hook cannot
    override the per-image streams that this module installs before every call.

    OpenCV is also pinned to a single thread per worker. Without this, N worker
    processes each spawn the default number of OpenCV threads (often the full core
    count), oversubscribing the CPU and erasing parallel gains. This changes the
    thread scheduling of OpenCV calls only; the per-pixel numerics are identical.
    """
    from copy import deepcopy

    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:  # pragma: no cover - cv2 is a hard dependency in practice
        pass

    worker_transform = deepcopy(transform)
    # Neutralize the torch worker hook for the worker-local copy only.
    compose = worker_transform if isinstance(worker_transform, BaseCompose) else None
    if compose is not None and hasattr(compose, "_base_seed"):
        compose._base_seed = None  # noqa: SLF001 - intentional: disable torch worker reseed hook
    _WORKER_STATE["transform"] = worker_transform
    _WORKER_STATE["seed"] = seed
    _WORKER_STATE["reader"] = reader


def _reset_worker_streams(transform: BaseCompose, seed: int | None, image_id: Any) -> None:
    """Install fresh, per-image-derived streams on every transform of the pipeline.

    The whole pipeline shares the two stream objects after ``set_random_state``, which
    is the same propagation mechanism :meth:`Compose.set_random_seed` uses.
    """
    rng, py_rng = _build_streams(seed, image_id)
    transform.set_random_state(rng, py_rng)


def _process_one(
    transform: BaseCompose,
    seed: int | None,
    reader: Callable[..., Any] | None,
    image_id: Any,
    data: dict[str, Any] | None,
) -> dict[str, Any]:
    if reader is not None:
        try:
            loaded = reader(image_id)
        except Exception as exc:
            raise _ReadError(str(exc)) from exc
        if isinstance(loaded, np.ndarray):
            data = {"image": loaded}
        elif isinstance(loaded, dict):
            data = loaded
        else:
            raise TypeError(
                f"reader must return a numpy array or a data dict, got {type(loaded).__name__}",
            )
    if data is None:
        raise ValueError("No data provided and no reader configured.")
    _reset_worker_streams(transform, seed, image_id)
    return transform(**data)


def _worker(chunk: list[tuple[int, Any, dict[str, Any] | None]]) -> list[dict[str, Any]]:
    """Process a chunk; one item failing never aborts the rest of the chunk."""
    transform = _WORKER_STATE["transform"]
    seed = _WORKER_STATE["seed"]
    reader = _WORKER_STATE["reader"]
    results: list[dict[str, Any]] = []
    for index, image_id, data in chunk:
        try:
            output = _process_one(transform, seed, reader, image_id, data)
            results.append({"index": index, "ok": True, "output": output})
        except _ReadError as exc:  # noqa: PERF203 - per-item isolation inside the loop
            cause = exc.__cause__
            results.append(
                {
                    "index": index,
                    "ok": False,
                    "stage": "read",
                    "error_type": type(cause).__name__,
                    "message": str(cause),
                },
            )
        except Exception as exc:  # noqa: BLE001 - item-level isolation, recorded above
            results.append(
                {
                    "index": index,
                    "ok": False,
                    "stage": "transform",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
    return results


# ---------------------------------------------------------------------------
# Driver side
# ---------------------------------------------------------------------------


def _chunk(items: list[NormalizedItem], chunk_size: int) -> list[list[tuple[int, Any, dict[str, Any] | None]]]:
    return [
        [(index, items[index][0], items[index][1]) for index in range(start, min(start + chunk_size, len(items)))]
        for start in range(0, len(items), chunk_size)
    ]


def _default_chunk_size(num_items: int, num_workers: int) -> int:
    """Split into at least a few chunks per worker so load stays balanced."""
    if num_items == 0:
        return 1
    target_chunks = max(num_workers * 4, 1)
    return max(1, int(np.ceil(num_items / target_chunks)))


def _drain_executor(
    executor: ProcessPoolExecutor,
    chunks: list[list[tuple[int, Any, dict[str, Any] | None]]],
    snapshot: list[int],
    in_flight_limit: int,
    chunk_attempts: list[int],
    absorb: Callable[[list[dict[str, Any]]], None],
) -> tuple[Exception | None, set[int]]:
    """Submit every chunk in ``snapshot`` to one pool and collect their payloads.

    Returns ``(broken_exc, completed)``: ``broken_exc`` is the first exception that
    signals a dead pool (``None`` when every chunk finished), and ``completed`` is the
    set of chunk indices whose payloads were absorbed successfully.
    """
    pending: dict[Any, int] = {}
    ptr = 0
    completed: set[int] = set()
    while ptr < len(snapshot) or pending:
        while ptr < len(snapshot) and len(pending) < in_flight_limit:
            chunk_index = snapshot[ptr]
            ptr += 1
            chunk_attempts[chunk_index] += 1
            try:
                future = executor.submit(_worker, chunks[chunk_index])
            except Exception as exc:  # noqa: BLE001 - any pool break must trigger the retry policy
                return exc, completed
            pending[future] = chunk_index
        try:
            done = next(as_completed(list(pending)))
        except Exception as exc:  # noqa: BLE001 - any pool break must trigger the retry policy
            return exc, completed
        chunk_index = pending.pop(done)
        try:
            payload = done.result()
        except Exception as exc:  # noqa: BLE001 - BrokenProcessPool/worker death -> retry policy
            return exc, completed
        absorb(payload)
        completed.add(chunk_index)
    return None, completed


def _reconcile_after_pool_break(
    broken_exc: Exception,
    snapshot: list[int],
    todo: set[int],
    chunks: list[list[tuple[int, Any, dict[str, Any] | None]]],
    chunk_attempts: list[int],
    attempts_for_index: list[int],
    normalized: list[NormalizedItem],
    failures: list[AugmentationFailure],
) -> tuple[int, int]:
    """Reclassify unfinished chunks after a pool death: retry once or record failure.

    Chunks never submitted (``attempts == 0``) stay queued for the fresh pool and are
    not counted as retries. Submitted chunks get one retry; a chunk that already used
    its retry is recorded as a failure for each item it contained.

    Returns:
        tuple[int, int]: ``(num_retried_chunks, num_transform_failures_recorded)``.

    """
    num_retried = 0
    num_failures = 0
    for chunk_index in snapshot:
        if chunk_index not in todo or chunk_attempts[chunk_index] == 0:
            continue
        attempts = chunk_attempts[chunk_index]
        for index, _id, _data in chunks[chunk_index]:
            attempts_for_index[index] = attempts + 1
        if attempts >= 2:
            for index, _id, _data in chunks[chunk_index]:
                failures.append(
                    AugmentationFailure(
                        index=index,
                        image_id=normalized[index][0],
                        stage="transform",
                        error_type=type(broken_exc).__name__,
                        message=f"worker process failed after one retry: {broken_exc}",
                        attempts=attempts,
                    ),
                )
                num_failures += 1
            todo.discard(chunk_index)
        else:
            # This chunk will be resubmitted to the fresh pool on the next loop.
            num_retried += 1
    return num_retried, num_failures


def _build_executor_kwargs(max_tasks_per_child: int | None) -> dict[str, Any]:
    """Translate the public ``max_tasks_per_child`` option for the running Python."""
    if max_tasks_per_child is None:
        return {}
    if not isinstance(max_tasks_per_child, int) or max_tasks_per_child < 1:
        raise ValueError("max_tasks_per_child must be a positive int or None.")
    kwargs: dict[str, Any] = {}
    with contextlib.suppress(TypeError):  # max_tasks_per_child unsupported on Python < 3.11
        kwargs["max_tasks_per_child"] = max_tasks_per_child
    return kwargs


def _run_pool(
    transform: BaseCompose,
    normalized: list[NormalizedItem],
    chunks: list[list[tuple[int, Any, dict[str, Any] | None]]],
    *,
    num_workers: int,
    seed: int | None,
    reader: Callable[..., Any] | None,
    mp_context: str | multiprocessing.context.BaseContext | None,
    max_tasks_per_child: int | None,
) -> tuple[
    list[dict[str, Any] | None],
    list[AugmentationFailure],
    int,
    int,
    int,
]:
    """Drive the process pool, including the one-retry policy for dead workers."""
    context = _resolve_mp_context(mp_context)
    executor_kwargs = _build_executor_kwargs(max_tasks_per_child)
    n_items = len(normalized)
    outputs: list[dict[str, Any] | None] = [None] * n_items
    failures: list[AugmentationFailure] = []
    read_failures = transform_failures = worker_retries = 0
    attempts_for_index = [1] * n_items
    chunk_attempts = [0] * len(chunks)

    def _absorb(payload: list[dict[str, Any]]) -> None:
        nonlocal read_failures, transform_failures
        for entry in payload:
            index = entry["index"]
            if entry["ok"]:
                outputs[index] = entry["output"]
                continue
            failures.append(
                AugmentationFailure(
                    index=index,
                    image_id=normalized[index][0],
                    stage=entry["stage"],
                    error_type=entry["error_type"],
                    message=entry["message"],
                    attempts=attempts_for_index[index],
                ),
            )
            if entry["stage"] == "read":
                read_failures += 1
            else:
                transform_failures += 1

    def _make_executor() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(transform, seed, reader),
            **executor_kwargs,
        )

    # At most num_workers chunks are in flight at once, bounding resident memory to
    # roughly num_workers * chunk_size items regardless of dataset size. After a pool
    # death the retry round runs one chunk at a time so the responsible chunk can be
    # isolated instead of condemning healthy neighbours.
    todo = set(range(len(chunks)))
    in_flight_limit = num_workers
    while todo:
        snapshot = sorted(todo)
        executor = _make_executor()
        try:
            broken_exc, completed = _drain_executor(
                executor,
                chunks,
                snapshot,
                in_flight_limit,
                chunk_attempts,
                _absorb,
            )
        finally:
            # A broken pool cannot serve further futures; always tear it down and
            # rebuild on the next outer iteration.
            executor.shutdown(wait=True, cancel_futures=True)

        todo.difference_update(completed)
        if broken_exc is None:
            continue

        retried_chunks, recorded_failures = _reconcile_after_pool_break(
            broken_exc=broken_exc,
            snapshot=snapshot,
            todo=todo,
            chunks=chunks,
            chunk_attempts=chunk_attempts,
            attempts_for_index=attempts_for_index,
            normalized=normalized,
            failures=failures,
        )
        worker_retries += retried_chunks
        transform_failures += recorded_failures
        in_flight_limit = 1

    return outputs, failures, read_failures, transform_failures, worker_retries


def _prepare_inputs(
    items: Sequence[Item] | Iterable[Item],
    *,
    num_workers: int,
    reader: Callable[..., Any] | None,
    chunk_size: int | None,
) -> tuple[list[NormalizedItem], list[list[tuple[int, Any, dict[str, Any] | None]]], int]:
    """Validate arguments and build the normalized items plus their chunk partition."""
    _validate_num_workers(num_workers)
    normalized = _normalize_items(items)
    if reader is None:
        for index, (_, data) in enumerate(normalized):
            if data is None:
                raise ValueError(f"Item at index {index} has no data and no reader was provided.")

    n_items = len(normalized)
    if chunk_size is None:
        resolved_chunk_size = _default_chunk_size(n_items, num_workers)
    elif isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive int or None, got {chunk_size!r}.")
    else:
        resolved_chunk_size = chunk_size

    return normalized, _chunk(normalized, resolved_chunk_size), n_items


def parallel_augment(
    transform: BaseCompose,
    items: Sequence[Item] | Iterable[Item],
    *,
    num_workers: int = 1,
    seed: int | None = None,
    reader: Callable[..., Any] | None = None,
    chunk_size: int | None = None,
    mp_context: str | multiprocessing.context.BaseContext | None = None,
    max_tasks_per_child: int | None = None,
) -> ParallelAugmentationResult:
    """Augment many images in parallel with per-image deterministic randomness.

    The augmentation result of any item depends only on its identifier and ``seed`` --
    it is identical across worker counts, process assignments and input ordering. The
    existing single-process pipeline and transform numerics are not modified.

    Args:
        transform (BaseCompose): The existing pipeline (typically ``A.Compose([...])``).
            It is deep-copied inside workers; the caller's instance is never mutated.
        items: Input items. Each element is one of:

            * a data dict, e.g. ``{"image": img, "mask": m}`` -- its positional index
              is used as the identifier;
            * an ``(image_id, data_dict)`` tuple with an explicit identifier (path,
              dataset index, UUID, ...);
            * a raw :class:`numpy.ndarray`, treated as ``{"image": array}``.

            When ``reader`` is given, the data dict may be omitted by passing
            ``(image_id, None)`` pairs.
        num_workers (int): Number of worker processes. Must be a positive int; illegal
            values raise at startup. ``1`` runs an in-process loop with the same
            per-image derivation and is the reference for equivalence checks. Default
            is ``1``, so old callers that do not opt in keep the original behavior.
        seed (int | None): Global seed for per-image random-context derivation.
            Default ``None`` (still deterministic per ``(None, image_id)``, but passing
            an explicit int is recommended for reproducible experiments).
        reader (Callable[[Any], Any] | None): Optional callable mapping an identifier
            to an image array or data dict, executed *inside* the worker so image bytes
            never cross the process boundary twice. Exceptions raised by ``reader`` are
            counted as read failures and skipped without aborting the batch.
        chunk_size (int | None): Items per submitted task. Bounds the number of
            in-flight items (and therefore memory) to roughly
            ``num_workers * chunk_size``. Default: ``ceil(N / (4 * num_workers))``.
        mp_context: Multiprocessing start method: ``"fork"`` (default on Linux),
            ``"spawn"`` (default elsewhere and always available) or an already created
            context. Fork avoids re-importing the scientific stack per worker; spawn is
            platform-independent and safer with non-fork-safe native libraries.
        max_tasks_per_child (int | None): Forwarded to :class:`ProcessPoolExecutor`
            (Python >= 3.11); ignored on older versions.

    Returns:
        ParallelAugmentationResult: Aligned outputs (``None`` for failed items),
        per-failure records with identifiers, retry/throughput statistics.

    Raises:
        TypeError: If ``num_workers`` is not an int or an item has an unsupported type.
        ValueError: If ``num_workers`` is out of range or a tuple item is malformed.

    Examples:
        >>> import albumentations as A
        >>> transform = A.Compose([A.HorizontalFlip(p=0.5), A.GaussNoise(p=0.8)])
        >>> items = [{"image": img} for img in images]
        >>> result = parallel_augment(transform, items, num_workers=4, seed=137)
        >>> augmented = result.outputs  # aligned with `items`; None marks failures

    """
    normalized, chunks, n_items = _prepare_inputs(
        items,
        num_workers=num_workers,
        reader=reader,
        chunk_size=chunk_size,
    )
    start = time.perf_counter()

    if num_workers == 1:
        # In-process reference path: identical derivation, no process pool. The
        # caller's transform is still deep-copied so its RNG state is never mutated.
        outputs: list[dict[str, Any] | None] = [None] * n_items
        failures: list[AugmentationFailure] = []
        read_failures = transform_failures = worker_retries = 0
        attempts_for_index = [1] * n_items

        def _absorb(payload: list[dict[str, Any]]) -> None:
            nonlocal read_failures, transform_failures
            for entry in payload:
                index = entry["index"]
                if entry["ok"]:
                    outputs[index] = entry["output"]
                    continue
                failures.append(
                    AugmentationFailure(
                        index=index,
                        image_id=normalized[index][0],
                        stage=entry["stage"],
                        error_type=entry["error_type"],
                        message=entry["message"],
                        attempts=attempts_for_index[index],
                    ),
                )
                if entry["stage"] == "read":
                    read_failures += 1
                else:
                    transform_failures += 1

        _init_worker(transform, seed, reader)
        for chunk in chunks:
            _absorb(_worker(chunk))
    else:
        outputs, failures, read_failures, transform_failures, worker_retries = _run_pool(
            transform,
            normalized,
            chunks,
            num_workers=num_workers,
            seed=seed,
            reader=reader,
            mp_context=mp_context,
            max_tasks_per_child=max_tasks_per_child,
        )

    elapsed = time.perf_counter() - start
    num_succeeded = sum(out is not None for out in outputs)
    return ParallelAugmentationResult(
        outputs=outputs,
        failures=sorted(failures, key=lambda f: f.index),
        num_succeeded=num_succeeded,
        num_read_failures=read_failures,
        num_transform_failures=transform_failures,
        num_worker_retries=worker_retries,
        num_workers=num_workers,
        elapsed_seconds=elapsed,
        throughput=(num_succeeded / elapsed) if elapsed > 0 else float("inf"),
        seed=seed,
    )


def _resolve_mp_context(
    mp_context: str | multiprocessing.context.BaseContext | None,
) -> multiprocessing.context.BaseContext:
    if isinstance(mp_context, multiprocessing.context.BaseContext):
        return mp_context
    if isinstance(mp_context, str):
        return multiprocessing.get_context(mp_context)
    # Default: fork where the platform supports it (Linux), spawn elsewhere. Fork
    # avoids re-importing the whole scientific stack in every worker -- that import
    # alone can cost more than a small augmentation batch under spawn. Spawn remains
    # available explicitly for code that must not rely on an inherited interpreter.
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def benchmark_parallel_augment(
    transform: BaseCompose,
    items: Sequence[Item] | Iterable[Item],
    *,
    seed: int | None = None,
    worker_counts: Sequence[int] = (1, 2, 4),
    reader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run :func:`parallel_augment` at several parallelism levels and quantify speedup.

    Every run uses the same ``seed`` and the same per-image derivation, so the outputs
    can additionally be compared for equality across worker counts.

    Args:
        transform (BaseCompose): Pipeline to benchmark.
        items: Same input shapes as :func:`parallel_augment`.
        seed (int | None): Global seed. Default ``None``.
        worker_counts (Sequence[int]): Worker counts to benchmark. Must include ``1``
            as the single-process baseline; it is prepended automatically if missing.
        reader (Callable[[Any], Any] | None): Optional in-worker reader.

    Returns:
        dict: ``{"runs": [{"num_workers", "elapsed_seconds", "throughput", ...}],
        "speedup_vs_single_process": {...}, "outputs_equal": bool}``.

    """
    counts = list(dict.fromkeys(int(w) for w in worker_counts))
    if 1 not in counts:
        counts = [1, *counts]
    for w in counts:
        _validate_num_workers(w)

    # The same items are traversed once per worker count; materialize a generator so it
    # cannot be exhausted after the first run.
    materialized_items = list(items)

    runs: list[dict[str, Any]] = []
    reference_outputs: list[Any] | None = None
    outputs_equal = True
    for workers in counts:
        result = parallel_augment(
            transform,
            materialized_items,
            num_workers=workers,
            seed=seed,
            reader=reader,
        )
        runs.append(
            {
                "num_workers": workers,
                "elapsed_seconds": result.elapsed_seconds,
                "throughput": result.throughput,
                "num_succeeded": result.num_succeeded,
                "num_failed": result.num_failed,
            },
        )
        if reference_outputs is None:
            reference_outputs = result.outputs
        elif not _outputs_equal(reference_outputs, result.outputs):
            outputs_equal = False

    baseline = next(run for run in runs if run["num_workers"] == 1)
    speedup = {
        str(run["num_workers"]): (
            run["elapsed_seconds"] / baseline["elapsed_seconds"] if baseline["elapsed_seconds"] > 0 else float("inf")
        )
        for run in runs
    }
    return {
        "runs": runs,
        "speedup_vs_single_process": speedup,
        "outputs_equal": outputs_equal,
        "seed": seed,
    }


def _outputs_equal(a: list[Any], b: list[Any]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is None:
            continue
        if set(x) != set(y):
            return False
        for key in x:
            ax, by = x[key], y[key]
            if isinstance(ax, np.ndarray):
                if not isinstance(by, np.ndarray) or not np.array_equal(ax, by):
                    return False
            elif ax != by:
                return False
    return True
