"""Tests for whole-volume 3D geometric transforms (Affine3D, ElasticTransform3D).

Covers the four acceptance scenarios:
1. Whole-volume geometric transforms (rotation/scale/flip/elastic in 3D, not per-slice).
2. Volume/mask alignment under a shared transform with mask-safe interpolation.
3. Multi-modal volumes sharing one set of sampled random parameters.
4. Chunked (slab-wise) processing with seamless boundaries for large volumes.
Plus inverse-transform recovery, missing-modality degradation and parameter validation.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import ndimage

import albumentations as A
from albumentations.augmentations.transforms3d import functional as f3d


def make_blob_volume(shape=(16, 40, 40)):
    """Volume with an ellipsoid blob and a matching labeled mask."""
    depth, height, width = shape
    zz, yy, xx = np.mgrid[0:depth, 0:height, 0:width]
    cz, cy, cx = depth / 2, height / 2, width / 2
    blob = ((zz - cz) ** 2 / (depth / 5) ** 2 + (yy - cy) ** 2 / (height / 6) ** 2 + (xx - cx) ** 2 / (width / 6) ** 2) <= 1
    volume = np.zeros(shape, dtype=np.float32)
    volume[blob] = 1.0
    mask = np.zeros(shape, dtype=np.uint8)
    mask[blob] = 1
    mask[0:3, 5:10, 5:10] = 2  # second class far from the blob
    return volume, mask


# ---------------------------------------------------------------------------
# 1. Whole-volume geometric transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["rotate", "axes", "k"],
    [
        (((0, 0), (0, 0), (90, 90)), (1, 2), 3),   # 90 deg around z == rot90 in the yx plane
        (((0, 0), (90, 90), (0, 0)), (0, 2), 1),   # 90 deg around y == rot90 in the zx plane
        (((90, 90), (0, 0), (0, 0)), (0, 1), 3),   # 90 deg around x == rot90 in the zy plane
    ],
)
def test_affine3d_rotate90_matches_rot90(rotate, axes, k):
    """Exact 90-degree rotations around each axis must match np.rot90 in the matching plane."""
    volume = np.random.randint(0, 256, (20, 20, 20), dtype=np.uint8)
    transform = A.Affine3D(rotate=rotate, p=1.0)
    transformed = transform(volume=volume)["volume"]
    np.testing.assert_array_equal(transformed, np.rot90(volume, k=k, axes=axes))


def test_affine3d_rotation_is_true_3d():
    """A rotation around x mixes the z and y axes jointly (not slice-by-slice)."""
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[1, 4, 4] = 1.0  # off-center along z only
    transform = A.Affine3D(rotate=((90, 90), (0, 0), (0, 0)), p=1.0)
    transformed = transform(volume=volume)["volume"]
    # 90 deg around x maps (z=1, y=4) -> (z=4, y=7) for a 9x9x9 cube centered at 4
    assert transformed[4, 7, 4] > 0.5
    assert transformed[1, 4, 4] < 0.5


@pytest.mark.parametrize(
    ["flip_prob", "expected_axes"],
    [
        ((1.0, 0.0, 0.0), (2,)),      # flip x
        ((0.0, 1.0, 0.0), (1,)),      # flip y
        ((0.0, 0.0, 1.0), (0,)),      # flip z
        ((1.0, 1.0, 1.0), (0, 1, 2)),  # flip all
    ],
)
def test_affine3d_flip(flip_prob, expected_axes):
    volume = np.random.randint(0, 256, (5, 6, 7), dtype=np.uint8)
    transform = A.Affine3D(flip_prob=flip_prob, p=1.0)
    transformed = transform(volume=volume)["volume"]
    np.testing.assert_array_equal(transformed, np.flip(volume, axis=expected_axes))


def test_affine3d_identity_by_default():
    volume = np.random.randint(0, 256, (5, 6, 7), dtype=np.uint8)
    transformed = A.Affine3D(p=1.0)(volume=volume)["volume"]
    np.testing.assert_array_equal(transformed, volume)


def test_affine3d_scale_changes_content_not_shape():
    volume = np.zeros((9, 21, 21), dtype=np.float32)
    volume[4, 8:13, 8:13] = 1.0
    transform = A.Affine3D(scale=2.0, p=1.0)
    transformed = transform(volume=volume)["volume"]
    assert transformed.shape == volume.shape
    # The 5x5 square scaled by 2 around the center covers a ~10x10 region
    assert transformed[4].sum() > volume[4].sum() * 3


@pytest.mark.parametrize("plane,nonzero_axis", [("xy", 2), ("xz", 1), ("yz", 0)])
def test_affine3d_plane_restriction(plane, nonzero_axis):
    """With plane set, only the rotation angle of the corresponding axis is non-zero."""
    volume = np.random.randint(0, 256, (5, 10, 10), dtype=np.uint8)
    transform = A.Affine3D(rotate=(30, 30, 30), plane=plane, p=1.0)
    transform.set_random_seed(0)
    transform(volume=volume)
    angles = transform.get_applied_params()["angles_xyz"]
    for axis, angle in enumerate(angles):
        if axis == nonzero_axis:
            assert angle != 0.0
        else:
            assert angle == 0.0


def test_affine3d_plane_xy_leaves_depth_untouched():
    """Rotation restricted to the axial (xy) plane keeps z slices independent:
    a marker keeps its z index."""
    volume = np.zeros((9, 21, 21), dtype=np.float32)
    volume[2, 5, 10] = 1.0
    transform = A.Affine3D(rotate=(0, 0, 45), plane="xy", p=1.0)
    transformed = transform(volume=volume)["volume"]
    assert transformed[2].sum() > 0.5
    assert transformed[[1, 3]].sum() == 0.0


# ---------------------------------------------------------------------------
# 2. Volume/mask alignment under a shared transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        A.Affine3D(rotate=((-30, 30), (-30, 30), (-30, 30)), scale=(0.8, 1.2), flip_prob=0.5, p=1.0),
        A.ElasticTransform3D(alpha=4.0, sigma=8.0, p=1.0),
    ],
    ids=["affine", "elastic"],
)
def test_volume_mask_spatial_alignment(transform):
    """Image volume and mask volume stay spatially aligned under the same transform."""
    volume, mask = make_blob_volume()
    composed = A.Compose([transform], seed=7)
    out = composed(volume=volume, mask3d=mask)

    transformed_volume = out["volume"]
    transformed_mask = out["mask3d"]

    # Nearest-neighbor mask interpolation: no intermediate class values introduced
    assert set(np.unique(transformed_mask)) <= {0, 1, 2}
    assert transformed_mask.dtype == mask.dtype

    # The blob moved to the same place in both targets
    com_volume = np.array(ndimage.center_of_mass(transformed_volume > 0.5))
    com_mask = np.array(ndimage.center_of_mass(transformed_mask == 1))
    assert np.abs(com_volume - com_mask).max() < 1.0

    # The second class region is still present and disjoint from the blob class
    assert (transformed_mask == 2).sum() > 0
    assert not np.any((transformed_mask == 2) & (transformed_mask == 1))


def test_mask_uses_nearest_neighbor_by_default():
    """Default mask interpolation must not introduce values outside the label set."""
    mask = np.random.randint(0, 4, (8, 20, 20), dtype=np.uint8)
    transform = A.Affine3D(rotate=((-45, 45), (-45, 45), (-45, 45)), scale=(0.7, 1.3), p=1.0)
    transform.set_random_seed(3)
    transformed = transform(mask3d=mask)["mask3d"]
    assert set(np.unique(transformed)) <= set(np.unique(mask))


def test_keypoints_track_volume():
    """Keypoints are transformed consistently with the volume."""
    volume, _ = make_blob_volume()
    keypoints = np.array([[20.0, 20.0, 8.0]], dtype=np.float32)  # (x, y, z) at blob center
    composed = A.Compose(
        [A.Affine3D(rotate=((-30, 30), (-30, 30), (-30, 30)), scale=(0.9, 1.1), p=1.0)],
        seed=13,
        keypoint_params=A.KeypointParams(format="xyz"),
    )
    out = composed(volume=volume, keypoints=keypoints)
    com_volume = np.array(ndimage.center_of_mass(out["volume"] > 0.5))  # (z, y, x)
    kp = out["keypoints"][0]  # (x, y, z)
    assert np.abs(np.array([kp[2], kp[1], kp[0]]) - com_volume).max() < 1.0


# ---------------------------------------------------------------------------
# 3. Multi-modal volumes share one set of random parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        A.Affine3D(rotate=((-30, 30), (-30, 30), (-30, 30)), scale=(0.8, 1.2), flip_prob=0.5, p=1.0),
        A.ElasticTransform3D(alpha=4.0, sigma=8.0, p=1.0),
    ],
    ids=["affine", "elastic"],
)
def test_multimodal_volumes_share_random_params(transform):
    """All modalities in `volumes` are transformed with the same sampled parameters."""
    rng = np.random.default_rng(42)
    modality_a = rng.integers(0, 256, (6, 20, 20), dtype=np.uint8)
    modality_b = rng.integers(0, 256, (6, 20, 20), dtype=np.uint8)

    # Transform both modalities together, stacked
    composed_multi = A.Compose([transform], seed=123)
    out_multi = composed_multi(volumes=np.stack([modality_a, modality_b]))

    # Transform each modality separately with the same seed
    composed_single = A.Compose([transform.__class__(**transform.get_transform_init_args())], seed=123)
    out_a = composed_single(volume=modality_a)["volume"]
    composed_single = A.Compose([transform.__class__(**transform.get_transform_init_args())], seed=123)
    out_b = composed_single(volume=modality_b)["volume"]

    np.testing.assert_array_equal(out_multi["volumes"][0], out_a)
    np.testing.assert_array_equal(out_multi["volumes"][1], out_b)


def test_multimodal_params_sampled_once():
    """The params recorded for a multi-modal call equal those of a single-volume call."""
    volumes = np.random.randint(0, 256, (3, 6, 20, 20), dtype=np.uint8)
    composed_multi = A.Compose([A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)], seed=99)
    composed_multi(volumes=volumes)
    params_multi = composed_multi.transforms[0].get_applied_params()

    composed_single = A.Compose([A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)], seed=99)
    composed_single(volume=volumes[0])
    params_single = composed_single.transforms[0].get_applied_params()

    np.testing.assert_array_equal(params_multi["matrix"], params_single["matrix"])
    np.testing.assert_array_equal(params_multi["offset"], params_single["offset"])


def test_missing_modality_is_preserved():
    """A missing modality (None) passes through; the rest are augmented normally."""
    rng = np.random.default_rng(0)
    modality_a = rng.integers(0, 256, (6, 20, 20), dtype=np.uint8)
    modality_b = rng.integers(0, 256, (6, 20, 20), dtype=np.uint8)

    composed = A.Compose([A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)], seed=11)
    out = composed(volumes=[modality_a, None, modality_b])

    result = out["volumes"]
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[1] is None  # missing modality stays marked as None

    composed_single = A.Compose([A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)], seed=11)
    np.testing.assert_array_equal(result[0], composed_single(volume=modality_a)["volume"])
    composed_single = A.Compose([A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)], seed=11)
    np.testing.assert_array_equal(result[2], composed_single(volume=modality_b)["volume"])


# ---------------------------------------------------------------------------
# 4. Chunked processing is seamless
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        A.Affine3D(rotate=((-30, 30), (-30, 30), (-30, 30)), scale=(0.8, 1.2), p=1.0),
        A.ElasticTransform3D(alpha=4.0, sigma=8.0, p=1.0),
    ],
    ids=["affine", "elastic"],
)
def test_chunked_processing_matches_whole(transform):
    """Chunked processing produces results identical to whole-volume processing."""
    volume = np.random.randint(0, 256, (8, 24, 24), dtype=np.uint8)

    transform.set_random_seed(0)
    whole = transform(volume=volume)["volume"]

    init_args = transform.get_transform_init_args()
    init_args["max_chunk_voxels"] = 97
    chunked_transform = transform.__class__(**init_args)
    chunked_transform.set_random_seed(0)
    chunked = chunked_transform(volume=volume)["volume"]

    np.testing.assert_array_equal(whole, chunked)


def test_chunked_processing_mask_alignment():
    """Chunking keeps volume and mask aligned (no seams between slabs)."""
    volume, mask = make_blob_volume()
    composed = A.Compose(
        [A.Affine3D(rotate=((-30, 30),) * 3, scale=(0.8, 1.2), max_chunk_voxels=128, p=1.0)],
        seed=3,
    )
    out = composed(volume=volume, mask3d=mask)
    com_volume = np.array(ndimage.center_of_mass(out["volume"] > 0.5))
    com_mask = np.array(ndimage.center_of_mass(out["mask3d"] == 1))
    assert np.abs(com_volume - com_mask).max() < 1.0
    assert set(np.unique(out["mask3d"])) <= {0, 1, 2}


# ---------------------------------------------------------------------------
# 5. Inverse transforms and invertibility marking
# ---------------------------------------------------------------------------


def test_affine3d_inverse_restores_volume():
    """The inverse resampling map restores the original space up to interpolation error."""
    volume = ndimage.gaussian_filter(np.random.default_rng(0).random((24, 24, 24)).astype(np.float32), 2.0)
    transform = A.Affine3D(rotate=((-15, 15),) * 3, scale=(0.9, 1.1), flip_prob=0.5, p=1.0)
    transform.set_random_seed(3)
    transformed = transform(volume=volume)["volume"]

    restored = transform.apply_to_volume(transformed, **transform.get_inverse_params())

    inner = (slice(6, -6),) * 3
    np.testing.assert_allclose(restored[inner], volume[inner], atol=0.02)


def test_affine3d_params_are_recoverable():
    """Applied params (matrix/offset) are stored for visualization and inversion."""
    volume = np.random.randint(0, 256, (6, 20, 20), dtype=np.uint8)
    transform = A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)
    transform(volume=volume)
    params = transform.get_applied_params()
    assert "matrix" in params and "offset" in params
    assert params["matrix"].shape == (3, 3)
    assert params["offset"].shape == (3,)
    # The recorded map is consistent: inverse of the inverse is the map itself
    matrix, offset = f3d.invert_affine3d(params["matrix"], params["offset"])
    matrix2, offset2 = f3d.invert_affine3d(matrix, offset)
    np.testing.assert_allclose(matrix2, params["matrix"])
    np.testing.assert_allclose(offset2, params["offset"])


def test_cubic_symmetry_inverse_is_exact():
    volume = np.random.randint(0, 256, (4, 5, 6), dtype=np.uint8)
    for index in range(48):
        transformed = f3d.transform_cube(volume, index)
        restored = f3d.transform_cube(transformed, f3d.invert_cube_index(index))
        np.testing.assert_array_equal(restored, volume)


def test_cubic_symmetry_get_inverse_params():
    volume = np.random.randint(0, 256, (4, 5, 6), dtype=np.uint8)
    transform = A.CubicSymmetry(p=1.0)
    transform.set_random_seed(1)
    transformed = transform(volume=volume)["volume"]
    restored = transform.apply_to_volume(transformed, **transform.get_inverse_params())
    np.testing.assert_array_equal(restored, volume)


def test_invertibility_is_explicitly_marked():
    """Invertible transforms are marked; lossy transforms must not pretend to be invertible."""
    assert A.Affine3D(p=1.0).is_invertible is True
    assert A.CubicSymmetry(p=1.0).is_invertible is True
    # Lossy / not exactly invertible transforms
    assert A.ElasticTransform3D(p=1.0).is_invertible is False
    assert A.CenterCrop3D(size=(2, 4, 4)).is_invertible is False
    assert A.RandomCrop3D(size=(2, 4, 4)).is_invertible is False
    assert A.CoarseDropout3D().is_invertible is False


# ---------------------------------------------------------------------------
# 6. Parameter validation (rejected before application)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": (0.0, 1.0)},       # zero scale is degenerate
        {"scale": -1.0},             # negative scale
        {"scale": (1.2, 0.8)},       # decreasing range
        {"flip_prob": 1.5},          # probability > 1
        {"flip_prob": (0.5, -0.1, 0.5)},
        {"mask_interpolation": 2},   # only 0/1 allowed
        {"interpolation": 3},
        {"max_chunk_voxels": 0},
        {"rotate": (float("nan"), 0, 0)},
        {"rotate": ((10, -10), (0, 0), (0, 0))},  # min > max
    ],
)
def test_affine3d_invalid_params_rejected(kwargs):
    with pytest.raises((ValidationError, ValueError)):
        A.Affine3D(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1.0},
        {"alpha": float("inf")},
        {"sigma": 0.5},
        {"sigma": 0.0},
        {"mask_interpolation": 2},
    ],
)
def test_elastic3d_invalid_params_rejected(kwargs):
    with pytest.raises((ValidationError, ValueError)):
        A.ElasticTransform3D(**kwargs)


def test_elastic3d_invalid_grid_rejected_before_apply():
    """A malformed displacement grid is rejected instead of causing out-of-bounds access."""
    transform = A.ElasticTransform3D(p=1.0)
    volume = np.zeros((4, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="displacement_grid"):
        transform.apply_to_volume(volume, displacement_grid=np.zeros((2, 2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# 7. Shapes, dtypes and channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(6, 20, 20), (6, 20, 20, 1), (6, 20, 20, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_affine3d_preserves_shape_and_dtype(shape, dtype):
    volume = np.random.randint(0, 256, shape).astype(dtype) if dtype == np.uint8 else np.random.rand(*shape).astype(dtype)
    transform = A.Affine3D(rotate=((-30, 30),) * 3, scale=(0.8, 1.2), p=1.0)
    transform.set_random_seed(0)
    transformed = transform(volume=volume)["volume"]
    assert transformed.shape == volume.shape
    assert transformed.dtype == volume.dtype


def test_affine3d_multichannel_matches_single_channel():
    """Each channel of a (D, H, W, C) volume is warped with the same map."""
    rng = np.random.default_rng(1)
    channels = [rng.integers(0, 256, (6, 20, 20), dtype=np.uint8) for _ in range(3)]
    volume = np.stack(channels, axis=-1)

    transform = A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)
    transform.set_random_seed(0)
    transformed = transform(volume=volume)["volume"]

    for c, channel in enumerate(channels):
        single = A.Affine3D(rotate=((-30, 30),) * 3, p=1.0)
        single.set_random_seed(0)
        np.testing.assert_array_equal(transformed[..., c], single(volume=channel)["volume"])


def test_elastic3d_per_slice_independence_is_broken():
    """Sanity check that elastic deformation is truly 3D: identical slices diverge."""
    slice_2d = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    volume = np.stack([slice_2d] * 8)  # 8 identical slices
    transform = A.ElasticTransform3D(alpha=4.0, sigma=4.0, p=1.0)
    transform.set_random_seed(0)
    transformed = transform(volume=volume)["volume"]
    # With a true 3D displacement field, slices are warped differently
    assert not np.array_equal(transformed[0], transformed[4])
