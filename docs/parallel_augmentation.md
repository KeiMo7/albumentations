# Deterministic parallel augmentation

`albumentations.parallel_augment` runs an **existing** `Compose` pipeline over many
images in a pool of worker processes while keeping every per-image result
deterministic. It is a separate entry point: the single-process call
`transform(**data)` and every transform's numerical implementation are unchanged.

- New entry point: `A.parallel_augment(...)` (also `A.derive_image_seed`,
  `A.benchmark_parallel_augment`).
- Existing pipelines, existing call sites and existing tests are untouched. Not
  opting in means exactly the old behavior.

## When to use it

Use it when augmentation is the throughput bottleneck of a data pipeline and the
same dataset is processed repeatedly. Do **not** use it for a handful of small
images: process-pool startup dominates small batches, so `num_workers=1` (the
default) is the right choice there.

## Quick start

```python
import albumentations as A

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.GaussNoise(p=0.8),
        A.Rotate(limit=30, p=0.6),
    ],
    seed=999,
)

# Input shapes:
#   data dict                       -> positional index is the image id
#   (image_id, data_dict) tuple     -> explicit, stable identifier
#   raw numpy array                 -> {"image": array}, positional id
items = [{"image": img, "mask": mask} for img, mask in dataset]

result = A.parallel_augment(
    transform,
    items,
    num_workers=4,   # positive int; illegal values raise before any pool starts
    seed=137,        # global seed for per-image random-context derivation
)

augmented = result.outputs           # aligned with `items`; None marks a failed item
assert result.num_failed == 0
print(result.throughput)             # items/second, quantified against single process
```

### Loading images inside the workers

Passing image arrays through the pool means the parent process already holds every
decoded image. To decode inside workers instead, pass identifiers plus a `reader`:

```python
def read_image(image_id):
    img = cv2.imread(image_id)          # runs inside the worker process
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = A.parallel_augment(
    transform,
    [(path, None) for path in image_paths],
    num_workers=4,
    seed=137,
    reader=read_image,                  # must be picklable (module-level function/class)
)
```

A reader may return either a numpy array (treated as `{"image": array}`) or a full
data dict (for masks, bboxes, ...). Exceptions raised by the reader are counted as
**read failures** and skipped without aborting the batch.

## The randomness contract (deterministic derivation rule)

Every stochastic decision in albumentations is drawn from exactly two streams held
on each transform/composition instance and shared across the whole pipeline:

- `random_generator` — a `numpy.random.Generator`
- `py_random` — a `random.Random`

Local generators created inside some transforms (e.g. `CoarseDropout`, `ISONoise`)
are themselves seeded from these two streams, so controlling them controls every
random decision.

Before image `image_id` is processed under global seed `seed`, both streams are
rebuilt from a per-image seed derived by this fixed, documented rule:

```text
field separator = byte 0x1F
key  = b"albumentations.parallel.v1"
       + 0x1F + utf8(str(seed mod 2**32))
       + 0x1F + utf8(str(image_id))
img_seed = int.from_bytes(blake2b(key, digest_size=8), byteorder="big")
rng      = numpy.random.default_rng(img_seed)
pyr      = random.Random(img_seed)
```

`A.derive_image_seed(seed, image_id)` returns the 32-bit reduction of that value and
is independently testable. Consequences, all guaranteed by construction:

1. `img_seed` is a pure function of `(seed, image_id)`. It never depends on worker
   identity, processing order, time, or `num_workers`.
2. The same image id always produces the identical pair of streams and therefore the
   identical augmented output, in any process, any run, any worker count, and any
   input ordering.
3. There is deliberately **no "one fixed seed per process"**. Such a scheme would make
   an image's result depend on which worker happened to handle it, which is exactly
   the non-determinism this module exists to remove.
4. Image identifiers must be **stable**: use file paths, dataset indices, or UUIDs.
   Do not use object ids (`id(obj)`) or other values that change between runs.
5. If you pass bare data dicts/arrays without explicit ids, the **positional index**
   is the identifier. Reordering that input then legitimately changes which id each
   image carries (image at position 3 is id `3` in both runs). To get order-independent
   output under shuffling, pass explicit `(image_id, data)` tuples with stable ids.
6. A `None` seed still derives deterministic streams (from the literal string
   `"None"`), but passing an explicit integer is recommended for reproducible
   experiments.

### Parallelism and ordering do not change results; what does?

- Changing `num_workers`, shuffling the input, or rerunning: **no change** to any
  per-image output.
- Changing the global `seed` or an image id: changes that image's derived streams
  and therefore its output (by design).
- Extreme cases remain deterministic: a worker crash, a pool rebuild, and the
  one-retry path all reinstall the same per-image streams, so a successful retry
  yields the same output an uninterrupted run would have produced.

## Parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `transform` | required | Existing pipeline (`A.Compose([...])`). Deep-copied inside each worker; the caller's instance is never mutated. |
| `items` | required | Data dicts, `(image_id, data_dict)` tuples, raw arrays, or `(image_id, None)` when a `reader` is used. |
| `num_workers` | `1` | Worker processes. Must be a positive `int`; `0`, negatives, non-ints, and values above `4 × os.cpu_count()` raise at startup. `1` runs an in-process loop with the same derivation and is the equivalence reference. |
| `seed` | `None` | Global seed for the per-image derivation. |
| `reader` | `None` | Optional picklable callable `image_id -> array | dict`, executed inside workers. |
| `chunk_size` | `ceil(N / (4 × num_workers))` | Items per submitted task. Bounds in-flight memory. |
| `mp_context` | `"fork"` on Linux, `"spawn"` elsewhere | Start method, or a prebuilt context. Fork avoids re-importing the scientific stack per worker; spawn is safer with non-fork-safe native libraries. |
| `max_tasks_per_child` | `None` | Forwarded to `ProcessPoolExecutor` on Python ≥ 3.11; ignored on older versions. |

### New configuration and old callers

The only new configuration is the `parallel_augment(...)` entry point and its
keyword arguments. Nothing new is added to `Compose`, and no existing argument
changes meaning. Old callers that keep using `transform(**data)` are unaffected and
require no changes. If you do not call the new entry point, nothing in your program
changes.

## Result object and failure semantics

`parallel_augment` returns a `ParallelAugmentationResult`:

- `outputs`: list aligned with the input; `None` marks a failed item.
- `failures`: `AugmentationFailure(index, image_id, stage, error_type, message,
  attempts)` — every failed item is located by both position and identifier.
- `num_succeeded`, `num_read_failures`, `num_transform_failures`,
  `num_worker_retries`, `num_workers`, `elapsed_seconds`, `throughput`, `seed`.

Rules:

- **Image read failure** (`reader` raised): the item is skipped, counted under
  `num_read_failures`, and the rest of the batch continues.
- **Transform failure** (the pipeline raised for one image): isolated inside the
  worker, recorded with the image id and exception details; other items continue.
- **Worker process death** (`BrokenProcessPool`, hard kill, ...): the affected chunk
  is retried **exactly once** on a freshly built pool. If the retry also dies, every
  item in that chunk is recorded as a transform failure (`attempts == 2`). Ordinary
  item-level Python exceptions are not retried — they are deterministic and would
  simply fail again.
- **Illegal parallelism**: rejected at startup, before any process is created.

## Memory bound

At most `num_workers` chunks are in flight at once, each holding at most
`chunk_size` items. Resident memory is therefore bounded by approximately

```text
num_workers × chunk_size × (per-item input + per-item output peak)
```

plus one deep copy of the pipeline per worker. It does **not** grow with dataset
size. With the default `chunk_size`, in-flight items are at most
`num_workers × ceil(N / (4 × num_workers))`, i.e. roughly `N/4` regardless of the
worker count.

No GPU is used: this module orchestrates CPU processes only and introduces no CUDA
context, no GPU-side copies, and no GPU synchronization points. Each worker pins
OpenCV to a single thread (`cv2.setNumThreads(1)`) so that N processes do not each
spawn the default thread count and oversubscribe the CPU; this changes OpenCV thread
scheduling only, never the numerical result.

## Benchmarking

```python
report = A.benchmark_parallel_augment(
    transform, items, seed=137, worker_counts=(1, 2, 4, 8),
)
# report["runs"]                      -> elapsed/throughput per worker count
# report["speedup_vs_single_process"] -> {"1": 1.0, "2": ..., "4": ...}
# report["outputs_equal"]             -> True across all worker counts
```

Measured on this 16-core Linux machine (128–256 images, 256–512 px, heavy pipeline
with `Rotate`/`Blur`/`GaussNoise`/`ElasticTransform`, fork, single-threaded OpenCV
per worker), 4 workers give roughly **2.6–3.2×** and 8 workers **3.6–4.7×**
throughput over the single-process path, with byte-identical outputs. Small batches
under `spawn` can be slower than single-process because each worker re-imports the
scientific stack; use fork on Linux or batch enough images to amortize startup.

## Multi-machine environments

No cross-machine coordination is required and none is attempted. The derivation is
stateless: any machine that knows `(seed, image_id)` computes the same
`img_seed` independently, because BLAKE2b and the `numpy`/`random` seeding semantics
used here are stable across machines and Python processes. There is no shared
counter, coordinator, or quorum to fail.

Requirements for reproducible output across machines:

1. All machines use the same global `seed` and the same image-id convention.
2. The same albumentations/numpy/OpenCV major versions are installed (transform
   numerics, not the derivation, are version-dependent — this is the same
   compatibility caveat as single-process reproducibility).

Degradation when a machine cannot participate: its shard simply fails locally under
the normal failure policy (one retry, then recorded failures). Other machines are
unaffected because no machine depends on another's random state. If your
scheduler/coordinator itself fails to assign shards, fall back to processing the
unassigned shards with `num_workers=1` on any single node — outputs are identical
because they never depended on the worker topology.

## Compatibility with existing weights and serialized data

- **Model weights**: unaffected. Augmentation runs on CPU and changes nothing about
  training checkpoints. Old weights remain valid; no migration is needed.
- **Serialized pipelines** (`to_dict`/`from_dict`, `ReplayCompose`): the format is
  unchanged. This module stores nothing new on the pipeline and does not alter
  `Compose`'s `seed` argument. Replay dictionaries produced by older versions replay
  exactly as before.
- **Old augmented data / cached outputs**: nothing invalidates them, but note that
  the per-image derivation defines a *new* random-context assignment. An image
  augmented through `parallel_augment(seed=S)` is not expected to equal the same
  image augmented through a sequentially seeded single-process run — those are two
  different, each internally consistent, randomness conventions. Within the parallel
  convention, results are stable forever for a fixed `(seed, image_id)` and fixed
  library versions. If you have caches keyed on the old sequential convention, keep
  them; the two conventions coexist and never overwrite each other because the
  parallel path is a separate entry point.
- **Relationship to the existing torch-DataLoader worker seeding**: `Compose` already
  adjusts its seed inside torch `DataLoader` workers via `_get_effective_seed`. That
  mechanism is order/worker dependent and is intentionally bypassed inside this
  module's worker-local pipeline copies (their `_base_seed` is cleared), because the
  id-derived streams must be the single source of randomness here. Both mechanisms
  remain available; they are simply different entry points with different contracts.

## Known limitations

- `spawn` (the default on Windows/macOS) requires every `reader`, pipeline component
  and custom transform to be picklable and defined at module scope; local lambdas
  cannot be shipped to workers. On Linux, prefer the default `fork`.
- Results are reproducible across runs and topologies for fixed library versions.
  Upgrading numpy/OpenCV/albumentations can change transform numerics, exactly as in
  single-process mode; the derivation rule itself (`...parallel.v1`) is versioned and
  will only change behind a new domain string.
- Determinism covers randomness. A transform whose output depends on wall-clock time
  or other external state remains external-state dependent; no built-in transform
  does this.
- Very small workloads should stay on `num_workers=1`; pool startup is not free.
