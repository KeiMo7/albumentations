"""Tests for the deterministic parallel augmentation entry point.

Covers the acceptance contract:
* multi-process output equals the single-process reference under the same seed;
* changing parallelism does not change any result;
* the same image id yields identical output across workers and input ordering;
* single-item transform/read failures are isolated, located by id and counted;
* illegal parallelism is rejected before any pool starts;
* a crashed worker triggers exactly one chunk retry, then a recorded failure;
* throughput speedup is quantified;
* the per-image seed derivation rule is pinned to known answers.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pytest

import albumentations as A
from albumentations.parallel import (
    AugmentationFailure,
    ParallelAugmentationResult,
    derive_image_seed,
    parallel_augment,
)

pytestmark = pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="Multiprocessing tests rely on the spawn/fork semantics of Linux CI",
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.7),
            A.GaussNoise(p=0.8),
            A.Rotate(limit=25, p=0.6),
            A.Blur(blur_limit=5, p=0.4),
        ],
        seed=999,
    )


def _make_items(n: int = 24, size: int = 64, with_mask: bool = False) -> list[dict]:
    rng = np.random.default_rng(2024)
    items = []
    for i in range(n):
        data = {"image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8)}
        if with_mask:
            data["mask"] = rng.integers(0, 2, (size, size), dtype=np.uint8)
        items.append(data)
    return items


def _assert_results_equal(left: ParallelAugmentationResult, right: ParallelAugmentationResult) -> None:
    assert len(left.outputs) == len(right.outputs)
    for index, (a, b) in enumerate(zip(left.outputs, right.outputs)):
        assert a is not None and b is not None, index
        assert set(a) == set(b), index
        for key in a:
            if isinstance(a[key], np.ndarray):
                assert np.array_equal(a[key], b[key]), (index, key)
            else:
                assert a[key] == b[key], (index, key)


# ---------------------------------------------------------------------------
# Derivation rule (documented, pinned)
# ---------------------------------------------------------------------------


def test_derive_image_seed_known_answer():
    # Independent re-implementation of the documented rule; pins the exact bytes.
    def reference(seed: int, image_id: str) -> int:
        key = b"\x1f".join(
            [
                b"albumentations.parallel.v1",
                str(seed % 2**32).encode("utf-8"),
                str(image_id).encode("utf-8"),
            ],
        )
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % 2**32

    for seed, image_id in [(0, "a"), (137, "img_0"), (137, "img_1"), (2**33, 42)]:
        assert derive_image_seed(seed, image_id) == reference(seed, image_id)
        assert 0 <= derive_image_seed(seed, image_id) < 2**32


def test_derive_image_seed_is_order_independent_and_distinct():
    assert derive_image_seed(137, "x") == derive_image_seed(137, "x")
    assert derive_image_seed(137, "x") != derive_image_seed(137, "y")
    assert derive_image_seed(137, "x") != derive_image_seed(138, "x")
    # Large seeds are reduced modulo 2**32 before hashing.
    assert derive_image_seed(137 + 2**32, "x") == derive_image_seed(137, "x")


# ---------------------------------------------------------------------------
# 1. Multi-process == single process, and 2. parallelism invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_parallel_matches_single_process(num_workers):
    transform = _make_pipeline()
    items = _make_items()

    reference = parallel_augment(transform, items, num_workers=1, seed=137)
    result = parallel_augment(transform, items, num_workers=num_workers, seed=137)

    _assert_results_equal(reference, result)
    assert result.num_workers == num_workers
    assert result.num_failed == 0
    assert result.num_succeeded == len(items)


def test_results_invariant_to_worker_count_and_reruns():
    transform = _make_pipeline()
    items = _make_items()

    first = parallel_augment(transform, items, num_workers=2, seed=7)
    second = parallel_augment(transform, items, num_workers=4, seed=7)
    third = parallel_augment(transform, items, num_workers=4, seed=7)  # repeat run

    _assert_results_equal(first, second)
    _assert_results_equal(second, third)


def test_same_image_identical_across_workers_and_order():
    """The key assertion: one image id -> one output, any worker, any order."""
    transform = _make_pipeline()
    rng = np.random.default_rng(11)
    images = {f"img_{i:03d}": rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for i in range(16)}

    items_a = [(image_id, {"image": img}) for image_id, img in images.items()]
    items_b = list(reversed(items_a))  # different submission order

    run_a = parallel_augment(transform, items_a, num_workers=4, seed=137)
    run_b = parallel_augment(transform, items_b, num_workers=2, seed=137)

    by_id_a = {image_id: run_a.outputs[i] for i, (image_id, _) in enumerate(items_a)}
    by_id_b = {image_id: run_b.outputs[i] for i, (image_id, _) in enumerate(items_b)}

    assert set(by_id_a) == set(by_id_b)
    for image_id in by_id_a:
        assert np.array_equal(by_id_a[image_id]["image"], by_id_b[image_id]["image"])


def test_spawn_context_matches_fork_and_single_process():
    """The spawn start method must produce the same id-derived results as fork."""
    transform = _make_pipeline()
    items = _make_items(n=8)

    reference = parallel_augment(transform, items, num_workers=1, seed=137)
    spawned = parallel_augment(
        transform,
        items,
        num_workers=2,
        seed=137,
        mp_context="spawn",
    )
    _assert_results_equal(reference, spawned)


def test_masks_and_bboxes_transform_consistently():
    transform = A.Compose(
        [A.HorizontalFlip(p=0.8), A.Rotate(limit=20, p=0.7)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        seed=5,
    )
    rng = np.random.default_rng(3)
    items = []
    for _ in range(8):
        items.append(
            {
                "image": rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
                "mask": rng.integers(0, 2, (64, 64), dtype=np.uint8),
                "bboxes": np.array([[5, 6, 40, 45]], dtype=np.float32),
                "labels": [1],
            },
        )

    single = parallel_augment(transform, items, num_workers=1, seed=137)
    multi = parallel_augment(transform, items, num_workers=3, seed=137)
    _assert_results_equal(single, multi)


def test_caller_transform_is_not_mutated():
    transform = _make_pipeline()
    items = _make_items(n=4)
    before_rng = transform.random_generator.bit_generator.state
    before_py = transform.py_random.getstate()

    parallel_augment(transform, items, num_workers=2, seed=137)

    assert transform.random_generator.bit_generator.state == before_rng
    assert transform.py_random.getstate() == before_py


# ---------------------------------------------------------------------------
# 3. Failures: transform errors and read errors
# ---------------------------------------------------------------------------


def _transform_that_raises(img, **kwargs):
    raise RuntimeError("boom inside pipeline")


_FAIL_SENTINEL = 250


def _maybe_fail_on_content(img, **kwargs):
    """Fails only for images carrying the sentinel pixel value."""
    if np.all(img[0, 0] == _FAIL_SENTINEL):
        raise ValueError("single bad item")
    return img


def test_single_transform_failure_is_isolated_and_located():
    # Pipeline whose first transform always raises; every item will fail, proving the
    # batch continues and each failure is located by id.
    transform = A.Compose([A.Lambda(image=_transform_that_raises, p=1.0)], seed=1)
    items = [(f"img_{i:03d}", {"image": np.zeros((16, 16, 3), dtype=np.uint8)}) for i in range(6)]

    result = parallel_augment(transform, items, num_workers=2, seed=137)

    assert result.num_succeeded == 0
    assert result.num_transform_failures == 6
    assert result.num_read_failures == 0
    assert all(out is None for out in result.outputs)
    assert [f.image_id for f in result.failures] == [f"img_{i:03d}" for i in range(6)]
    assert all(f.stage == "transform" and f.error_type == "RuntimeError" for f in result.failures)
    assert all("boom inside pipeline" in f.message for f in result.failures)
    assert all(isinstance(f, AugmentationFailure) for f in result.failures)


def test_one_bad_item_does_not_stop_the_batch():
    transform = A.Compose([A.Lambda(image=_maybe_fail_on_content, p=1.0)], seed=1)
    # Item 3 carries the sentinel; the pipeline must raise only for that one image.
    items = []
    for i in range(8):
        fill = _FAIL_SENTINEL if i == 3 else i
        items.append({"image": np.full((8, 8, 3), fill, dtype=np.uint8)})

    result = parallel_augment(transform, items, num_workers=2, seed=137)

    assert result.num_succeeded == 7
    assert result.num_transform_failures == 1
    failure = result.failures[0]
    assert failure.index == 3
    assert failure.image_id == 3  # positional index used as id
    assert failure.error_type == "ValueError"
    assert result.outputs[3] is None
    assert [i for i, out in enumerate(result.outputs) if out is not None] == [0, 1, 2, 4, 5, 6, 7]


def _failing_reader(image_id):
    if image_id in {"bad_1", "bad_2"}:
        raise OSError(f"cannot read {image_id}")
    return np.full((12, 12, 3), abs(hash(image_id)) % 255, dtype=np.uint8)


def test_reader_failures_are_counted_and_skipped():
    transform = _make_pipeline()
    ids = [f"good_{i}" for i in range(4)] + ["bad_1", "bad_2"]
    items = [(image_id, None) for image_id in ids]

    result = parallel_augment(transform, items, num_workers=2, seed=137, reader=_failing_reader)

    assert result.num_succeeded == 4
    assert result.num_read_failures == 2
    assert result.num_transform_failures == 0
    assert {f.image_id for f in result.failures} == {"bad_1", "bad_2"}
    assert all(f.stage == "read" and f.error_type == "OSError" for f in result.failures)
    assert result.outputs[4] is None and result.outputs[5] is None
    assert all(out is not None for i, out in enumerate(result.outputs) if i < 4)


class _PayloadReader:
    """Picklable reader carrying its payload in instance state (spawn-safe)."""

    def __init__(self, payload: dict[str, np.ndarray]):
        self.payload = payload

    def __call__(self, image_id):
        return self.payload[image_id]


def test_reader_outputs_match_direct_dict_input():
    transform = _make_pipeline()
    rng = np.random.default_rng(0)
    payload = {f"img_{i}": rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for i in range(6)}

    via_reader = parallel_augment(
        transform,
        [(image_id, None) for image_id in payload],
        num_workers=2,
        seed=137,
        reader=_PayloadReader(payload),
    )
    via_dict = parallel_augment(
        transform,
        [(image_id, {"image": img}) for image_id, img in payload.items()],
        num_workers=1,
        seed=137,
    )
    for a, b in zip(via_reader.outputs, via_dict.outputs):
        assert np.array_equal(a["image"], b["image"])


# ---------------------------------------------------------------------------
# 4. Illegal parallelism rejected at startup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_workers", [0, -1, -4])
def test_illegal_worker_count_rejected(bad_workers):
    transform = _make_pipeline()
    items = _make_items(n=2)
    with pytest.raises(ValueError, match="num_workers"):
        parallel_augment(transform, items, num_workers=bad_workers, seed=1)


@pytest.mark.parametrize("bad_workers", [1.5, "2", None, True])
def test_non_integer_worker_count_rejected(bad_workers):
    transform = _make_pipeline()
    items = _make_items(n=2)
    with pytest.raises(TypeError, match="num_workers"):
        parallel_augment(transform, items, num_workers=bad_workers, seed=1)


def test_excessive_worker_count_rejected():
    transform = _make_pipeline()
    items = _make_items(n=2)
    with pytest.raises(ValueError, match="CPU count"):
        parallel_augment(transform, items, num_workers=os.cpu_count() * 4 + 1, seed=1)


def test_bad_chunk_size_rejected():
    transform = _make_pipeline()
    items = _make_items(n=2)
    with pytest.raises(ValueError, match="chunk_size"):
        parallel_augment(transform, items, num_workers=2, chunk_size=0, seed=1)


def test_malformed_items_rejected():
    transform = _make_pipeline()
    with pytest.raises(ValueError, match="image_id"):
        parallel_augment(transform, [("only_one_element",)], num_workers=1, seed=1)
    with pytest.raises(TypeError, match="Unsupported item type"):
        parallel_augment(transform, [object()], num_workers=1, seed=1)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Worker crash: one retry, then a recorded failure
# ---------------------------------------------------------------------------


def _crashing_reader(image_id):
    if image_id == "cursed":
        os._exit(17)  # hard-kill the worker process, no exception propagation
    return np.zeros((16, 16, 3), dtype=np.uint8)


def test_worker_crash_retries_once_then_records_failure():
    transform = _make_pipeline()
    # Nine items with chunk_size=2 => the last chunk contains only "cursed", so a
    # failed retry records precisely that one item rather than a chunk neighbour.
    ids = [f"img_{i}" for i in range(8)] + ["cursed"]
    items = [(image_id, None) for image_id in ids]

    result = parallel_augment(
        transform,
        items,
        num_workers=2,
        seed=137,
        reader=_crashing_reader,
        chunk_size=2,
    )

    # The cursed chunk is retried once and still fails -> recorded; everything else
    # is augmented successfully.
    assert result.num_worker_retries >= 1
    assert result.num_succeeded == 8
    cursed_failures = [f for f in result.failures if f.image_id == "cursed"]
    assert len(cursed_failures) == 1
    assert cursed_failures[0].attempts == 2
    assert cursed_failures[0].error_type == "BrokenProcessPool"
    assert result.num_succeeded + result.num_failed == len(items)


# ---------------------------------------------------------------------------
# Throughput quantification
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.cpu_count() < 2, reason="need at least two cores for a speedup claim")
def test_throughput_speedup_is_quantified():
    transform = A.Compose(
        [
            A.Rotate(limit=30, p=0.9),
            A.Blur(blur_limit=7, p=0.9),
            A.GaussNoise(p=0.9),
            A.RandomBrightnessContrast(p=0.9),
            A.ElasticTransform(p=0.5),
        ],
        seed=3,
    )
    # Enough heavy items to amortize process-pool startup; small batches are not the
    # regime parallel augmentation is meant for.
    items = _make_items(n=128, size=256)
    workers = min(4, os.cpu_count())

    single = parallel_augment(transform, items, num_workers=1, seed=137)
    multi = parallel_augment(transform, items, num_workers=workers, seed=137)

    _assert_results_equal(single, multi)
    assert single.throughput > 0 and multi.throughput > 0
    speedup = single.elapsed_seconds / multi.elapsed_seconds
    # A real, measured speedup; the threshold leaves headroom for noisy shared runners.
    assert speedup > 1.1, f"expected speedup > 1.1, got {speedup:.2f}"
    assert result_has_metrics(single) and result_has_metrics(multi)


def result_has_metrics(result: ParallelAugmentationResult) -> bool:
    return (
        isinstance(result.elapsed_seconds, float)
        and isinstance(result.throughput, float)
        and result.elapsed_seconds > 0
    )


def test_benchmark_helper_reports_speedup_and_equality():
    transform = _make_pipeline()
    items = _make_items(n=12, size=32)

    report = A.benchmark_parallel_augment(
        transform,
        items,
        seed=137,
        worker_counts=[1, 2],
    )

    assert report["outputs_equal"] is True
    assert "1" in report["speedup_vs_single_process"]
    assert report["speedup_vs_single_process"]["1"] == pytest.approx(1.0)
    assert {run["num_workers"] for run in report["runs"]} == {1, 2}
    for run in report["runs"]:
        assert run["throughput"] > 0


def test_benchmark_accepts_generator_and_reuses_it_for_every_worker_count():
    transform = _make_pipeline()
    # A one-shot iterator would be exhausted after the first run without materializing.
    report = A.benchmark_parallel_augment(
        transform,
        ({"image": np.zeros((8, 8, 3), dtype=np.uint8)} for _ in range(6)),
        seed=137,
        worker_counts=[1, 2],
    )
    assert report["outputs_equal"] is True
    assert all(run["num_succeeded"] == 6 for run in report["runs"])


# ---------------------------------------------------------------------------
# Misc API surface
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_result():
    transform = _make_pipeline()
    result = parallel_augment(transform, [], num_workers=2, seed=137)
    assert result.outputs == []
    assert result.num_succeeded == 0
    assert result.num_failed == 0


def test_ndarray_items_and_positional_ids():
    transform = _make_pipeline()
    arrays = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)]
    result = parallel_augment(transform, arrays, num_workers=2, seed=137)
    assert result.num_succeeded == 4
    assert [f.image_id for f in result.failures] == []


def test_missing_data_without_reader_rejected():
    transform = _make_pipeline()
    with pytest.raises(ValueError, match="no reader"):
        parallel_augment(transform, [("x", None)], num_workers=1, seed=1)
