"""Geometric 3D transforms for volumetric data augmentation.

This module provides whole-volume geometric transforms (rotation, scaling, flipping and
elastic deformation) for 3D volumetric data such as medical CT/MRI scans. All transforms
in this module:

- transform the three spatial dimensions jointly, preserving 3D spatial consistency;
- apply the exact same sampled parameters to every target (volume, volumes, mask3d,
  masks3d, keypoints) in a single call, so image volumes, segmentation masks and
  multi-modal volumes stay strictly aligned;
- use nearest-neighbor interpolation for masks by default so no intermediate class
  values are introduced;
- support chunked (slab-wise) processing via ``max_chunk_voxels`` to bound memory usage
  on large volumes, with seamless chunk boundaries.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from pydantic import field_validator

from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.transforms_interface import Transform3D
from albumentations.core.type_definitions import Targets

__all__ = ["Affine3D", "ElasticTransform3D"]

NUM_DIMENSIONS = 3

PlaneName = Literal["all", "xy", "xz", "yz"]

# Which rotation axis each plane corresponds to: rotating in the xy plane is a
# rotation around the z axis (index 2), xz around y (index 1), yz around x (index 0).
PLANE_TO_AXIS: dict[str, int] = {"yz": 0, "xz": 1, "xy": 2}

RotateType = float | tuple[float, float, float] | tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


class BaseGeometric3D(Transform3D):
    """Base class for whole-volume geometric 3D transforms.

    Provides shared handling of interpolation settings, fill values, chunked
    processing and multi-modal inputs for geometric 3D transforms.

    Args:
        interpolation (int): Interpolation order for image volumes: 0 (nearest neighbor)
            or 1 (linear). Default: 1.
        mask_interpolation (int): Interpolation order for 3D masks: 0 (nearest neighbor)
            or 1 (linear). Default: 0 (nearest neighbor), which guarantees that no
            intermediate class values are introduced in segmentation masks.
        fill (tuple[float, ...] | float): Value used for voxels sampled outside the volume.
        fill_mask (tuple[float, ...] | float): Value used for mask voxels sampled outside
            the volume.
        max_chunk_voxels (int | None): If set, the output volume is computed in depth
            slabs of at most this many voxels, bounding peak memory usage. Chunk
            boundaries are seamless: the result is identical to processing the whole
            volume at once. If None, the whole volume is processed at once. Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Note:
        This is a base class and not intended to be used directly. Use Affine3D or
        ElasticTransform3D instead.

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(Transform3D.InitSchema):
        interpolation: Literal[0, 1]
        mask_interpolation: Literal[0, 1]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        max_chunk_voxels: int | None

        @field_validator("max_chunk_voxels")
        @classmethod
        def validate_max_chunk_voxels(cls, v: int | None) -> int | None:
            """Validate that max_chunk_voxels is a positive integer."""
            if v is not None and v < 1:
                raise ValueError(f"max_chunk_voxels must be >= 1, got {v}")
            return v

    def __init__(
        self,
        interpolation: int = 1,
        mask_interpolation: int = 0,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        max_chunk_voxels: int | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.fill = fill
        self.fill_mask = fill_mask
        self.max_chunk_voxels = max_chunk_voxels

    @staticmethod
    def _get_volume_shape(data: dict[str, Any]) -> tuple[int, int, int]:
        """Extract the spatial (D, H, W) shape from any available 3D target."""
        if "volume" in data and data["volume"] is not None:
            return cast("tuple[int, int, int]", data["volume"].shape[:3])
        if "volumes" in data and data["volumes"] is not None:
            volumes = data["volumes"]
            if isinstance(volumes, np.ndarray):
                return cast("tuple[int, int, int]", tuple(volumes.shape[1:4]))
            first = next((v for v in volumes if v is not None), None)
            if first is not None:
                return cast("tuple[int, int, int]", first.shape[:3])
        if "mask3d" in data and data["mask3d"] is not None:
            return cast("tuple[int, int, int]", data["mask3d"].shape[:3])
        if "masks3d" in data and data["masks3d"] is not None:
            masks3d = data["masks3d"]
            if isinstance(masks3d, np.ndarray):
                return cast("tuple[int, int, int]", tuple(masks3d.shape[1:4]))
            first = next((m for m in masks3d if m is not None), None)
            if first is not None:
                return cast("tuple[int, int, int]", first.shape[:3])
        raise ValueError(
            "3D geometric transforms require at least one of 'volume', 'volumes', "
            "'mask3d' or 'masks3d' in the input data.",
        )

    def apply_to_volumes(
        self,
        volumes: np.ndarray | list[np.ndarray | None],
        *args: Any,
        **params: Any,
    ) -> np.ndarray | list[np.ndarray | None]:
        """Apply the transform to a batch of (multi-modal) volumes.

        Every volume is transformed with the exact same sampled parameters, so
        modalities stay aligned. Volumes are processed one at a time to bound
        peak memory. When a list is passed, missing modalities (None entries)
        are preserved as None in the output.
        """
        if isinstance(volumes, (list, tuple)):
            return [self.apply_to_volume(volume, **params) if volume is not None else None for volume in volumes]
        return np.stack([self.apply_to_volume(volume, **params) for volume in volumes])

    def apply_to_masks3d(
        self,
        masks3d: np.ndarray | list[np.ndarray | None],
        *args: Any,
        **params: Any,
    ) -> np.ndarray | list[np.ndarray | None]:
        """Apply the transform to a batch of 3D masks with shared parameters."""
        if isinstance(masks3d, (list, tuple)):
            return [self.apply_to_mask3d(mask, **params) if mask is not None else None for mask in masks3d]
        return np.stack([self.apply_to_mask3d(mask, **params) for mask in masks3d])


class Affine3D(BaseGeometric3D):
    """Apply a random affine transform (rotation, scaling, flipping) to a 3D volume as a whole.

    The volume is transformed jointly in all three dimensions (not slice-by-slice), so
    the 3D spatial structure is preserved. The same sampled transform is applied to the
    volume, its segmentation mask (with nearest-neighbor interpolation by default),
    multi-modal volumes and keypoints, keeping them strictly aligned.

    Args:
        rotate (float | tuple[float, float, float] | tuple[tuple[float, float], ...]):
            Rotation angles in degrees around the (x, y, z) axes. Can be:
                * float - angles are sampled uniformly from [-value, value] for all axes
                * tuple of 3 floats - per-axis symmetric ranges [-v, v] for (x, y, z)
                * tuple of 3 (min, max) tuples - explicit per-axis ranges for (x, y, z)
            A rotation around the z axis takes place in the xy (axial) plane, around y
            in the xz (coronal) plane and around x in the yz (sagittal) plane.
            Default: 0.0 (no rotation).
        scale (float | tuple[float, float]): Isotropic scale factor. A float applies an
            exact scale; a (min, max) tuple samples uniformly from the range.
            Must be positive. Default: 1.0.
        flip_prob (float | tuple[float, float, float]): Probability of flipping each of
            the (x, y, z) axes. A single float is used for all three axes. Default: 0.0.
        plane (Literal["all", "xy", "xz", "yz"]): Restrict rotation to a single plane.
            "xy" is the axial plane (rotation around z only), "xz" the coronal plane
            (around y only) and "yz" the sagittal plane (around x only). With "all",
            rotations around all three axes are applied. Default: "all".
        interpolation (int): Interpolation order for image volumes: 0 (nearest) or
            1 (linear). Default: 1.
        mask_interpolation (int): Interpolation order for 3D masks: 0 (nearest) or
            1 (linear). Default: 0, which introduces no intermediate class values.
        fill (tuple[float, ...] | float): Value for voxels sampled outside the volume.
        fill_mask (tuple[float, ...] | float): Value for mask voxels sampled outside
            the volume.
        max_chunk_voxels (int | None): If set, the output is computed in depth slabs of
            at most this many voxels (seamless boundaries), bounding peak memory.
            Default: None (process the whole volume at once).
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - This transform is invertible (up to interpolation error): the applied
          parameters are stored and the inverse resampling map can be obtained with
          ``get_inverse_params()``. Use ``transform.get_applied_params()`` or
          ``ReplayCompose`` to recover the parameters of an applied transform.
        - Multi-modal data: pass modalities stacked as ``volumes`` (shape (N, D, H, W)
          or (N, D, H, W, C)) or as a list; all modalities share one set of sampled
          random parameters.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.random.randint(0, 256, (20, 128, 128), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (20, 128, 128), dtype=np.uint8)    # (D, H, W)
        >>> transform = A.Compose([
        ...     A.Affine3D(rotate=(0, 0, 30), scale=(0.9, 1.1), flip_prob=(0, 0, 0.5), p=1.0),
        ... ])
        >>> transformed = transform(volume=volume, mask3d=mask3d)
        >>> transformed_volume, transformed_mask3d = transformed["volume"], transformed["mask3d"]
        >>>
        >>> # Invert the transform (up to interpolation error)
        >>> affine = A.Affine3D(rotate=(10, 20, 30), p=1.0)
        >>> out = affine(volume=volume)["volume"]
        >>> restored = affine.apply_to_volume(out, **affine.get_inverse_params())

    """

    is_invertible = True

    class InitSchema(BaseGeometric3D.InitSchema):
        rotate: RotateType
        scale: float | tuple[float, float]
        flip_prob: float | tuple[float, float, float]
        plane: PlaneName

        @field_validator("scale")
        @classmethod
        def validate_scale(cls, v: float | tuple[float, float]) -> float | tuple[float, float]:
            """Validate that scale bounds are positive."""
            values = v if isinstance(v, tuple) else (v,)
            if any(s <= 0 for s in values):
                raise ValueError(f"Scale must be positive, got {v}")
            if isinstance(v, tuple) and (len(v) != 2 or v[0] > v[1]):
                raise ValueError(f"Scale range must be a (min, max) tuple with min <= max, got {v}")
            return v

        @field_validator("flip_prob")
        @classmethod
        def validate_flip_prob(
            cls,
            v: float | tuple[float, float, float],
        ) -> float | tuple[float, float, float]:
            """Validate that flip probabilities are in [0, 1]."""
            values = v if isinstance(v, tuple) else (v,)
            if any(not 0 <= prob <= 1 for prob in values):
                raise ValueError(f"flip_prob values must be in [0, 1], got {v}")
            return v

        @field_validator("rotate")
        @classmethod
        def validate_rotate(cls, v: RotateType) -> RotateType:
            """Validate that rotation ranges are finite and non-decreasing."""
            if isinstance(v, (int, float)):
                if not np.isfinite(v):
                    raise ValueError(f"Rotation angles must be finite, got {v}")
                return v
            if len(v) != NUM_DIMENSIONS:
                raise ValueError(f"rotate must have 3 entries (one per axis), got {v}")
            for entry in v:
                entry_values = entry if isinstance(entry, tuple) else (entry,)
                if not all(np.isfinite(angle) for angle in entry_values):
                    raise ValueError(f"Rotation angles must be finite, got {v}")
                if isinstance(entry, tuple) and (len(entry) != 2 or entry[0] > entry[1]):
                    raise ValueError(f"Rotation ranges must be (min, max) with min <= max, got {v}")
            return v

    def __init__(
        self,
        rotate: RotateType = 0.0,
        scale: float | tuple[float, float] = 1.0,
        flip_prob: float | tuple[float, float, float] = 0.0,
        plane: PlaneName = "all",
        interpolation: int = 1,
        mask_interpolation: int = 0,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        max_chunk_voxels: int | None = None,
        p: float = 1.0,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            fill=fill,
            fill_mask=fill_mask,
            max_chunk_voxels=max_chunk_voxels,
            p=p,
        )
        self.rotate = rotate
        self.scale = scale
        self.flip_prob = flip_prob
        self.plane = plane

    def _normalize_rotate_ranges(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Normalize the rotate argument to per-axis (min, max) ranges for (x, y, z)."""
        if isinstance(self.rotate, (int, float)):
            value = float(self.rotate)
            return ((-value, value),) * NUM_DIMENSIONS
        ranges = []
        for entry in self.rotate:
            if isinstance(entry, tuple):
                ranges.append((float(entry[0]), float(entry[1])))
            else:
                ranges.append((-float(entry), float(entry)))
        return cast("tuple[tuple[float, float], tuple[float, float], tuple[float, float]]", tuple(ranges))

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Sample one set of affine parameters shared by all targets.

        Args:
            params (dict[str, Any]): Dictionary of existing parameters.
            data (dict[str, Any]): Dictionary containing input data with volume, mask3d, etc.

        Returns:
            dict[str, Any]: Dictionary with the resampling map (matrix, offset) in
                (z, y, x) index order plus the sampled angles, scale and flips.

        """
        shape_zyx = self._get_volume_shape(data)
        depth, height, width = shape_zyx

        # Sample one angle per axis (x, y, z); zero out axes outside the selected plane
        rotate_ranges = self._normalize_rotate_ranges()
        angles_xyz = [self.random_generator.uniform(lo, hi) for lo, hi in rotate_ranges]
        if self.plane != "all":
            keep_axis = PLANE_TO_AXIS[self.plane]
            angles_xyz = [angle if axis == keep_axis else 0.0 for axis, angle in enumerate(angles_xyz)]

        scale_range = self.scale if isinstance(self.scale, tuple) else (self.scale, self.scale)
        scale = float(self.random_generator.uniform(*scale_range))

        flip_probs = self.flip_prob if isinstance(self.flip_prob, tuple) else (self.flip_prob,) * NUM_DIMENSIONS
        flips = tuple(bool(self.py_random.random() < prob) for prob in flip_probs)

        rotation_matrix = f3d.get_rotation_matrix_3d(tuple(angles_xyz))
        forward_matrix = f3d.get_affine3d_forward_matrix(rotation_matrix, scale, flips)

        # Reject degenerate transforms before applying anything
        determinant = float(np.linalg.det(forward_matrix))
        if not np.isfinite(determinant) or abs(determinant) < 1e-8:
            raise ValueError(
                f"Degenerate 3D affine transform (determinant={determinant}); "
                f"check rotate/scale/flip parameters.",
            )

        center_xyz = ((width - 1) / 2.0, (height - 1) / 2.0, (depth - 1) / 2.0)
        matrix, offset = f3d.resampling_map_from_forward(forward_matrix, center_xyz)

        return {
            "matrix": matrix,
            "offset": offset,
            "angles_xyz": angles_xyz,
            "scale": scale,
            "flips": flips,
        }

    def apply_to_volume(self, volume: np.ndarray, matrix: np.ndarray, offset: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the affine warp to a 3D volume.

        Args:
            volume (np.ndarray): Input volume of shape (D, H, W) or (D, H, W, C).
            matrix (np.ndarray): Resampling matrix (3, 3) in (z, y, x) index order.
            offset (np.ndarray): Resampling offset (3,) in (z, y, x) index order.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Warped volume with the same shape and dtype as the input.

        """
        return f3d.warp_affine3d(volume, matrix, offset, self.interpolation, self.fill, self.max_chunk_voxels)

    def apply_to_mask3d(self, mask3d: np.ndarray, matrix: np.ndarray, offset: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the affine warp to a 3D mask with mask interpolation (nearest by default).

        Args:
            mask3d (np.ndarray): Input mask of shape (D, H, W) or (D, H, W, C).
            matrix (np.ndarray): Resampling matrix (3, 3) in (z, y, x) index order.
            offset (np.ndarray): Resampling offset (3,) in (z, y, x) index order.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Warped mask with the same shape and dtype as the input.

        """
        return f3d.warp_affine3d(mask3d, matrix, offset, self.mask_interpolation, self.fill_mask, self.max_chunk_voxels)

    def apply_to_keypoints(self, keypoints: np.ndarray, matrix: np.ndarray, offset: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the forward affine transform to keypoints.

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
                The first three columns are x, y, z coordinates.
            matrix (np.ndarray): Resampling matrix (3, 3) in (z, y, x) index order.
            offset (np.ndarray): Resampling offset (3,) in (z, y, x) index order.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed keypoints with the same shape as the input.

        """
        return f3d.apply_affine3d_to_keypoints(keypoints, matrix, offset)

    def get_inverse_params(self, params: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        """Get the resampling map of the inverse transform.

        Args:
            params (dict[str, Any] | None): Parameters of the applied transform.
                If None, the parameters of the last application are used.

        Returns:
            dict[str, np.ndarray]: Dictionary with "matrix" and "offset" of the inverse
                resampling map, suitable for ``apply_to_volume``/``apply_to_mask3d``.
                Applying it restores the original space up to interpolation error.

        """
        params = self.params if params is None else params
        if not params or "matrix" not in params:
            raise ValueError("No applied parameters available; apply the transform first or pass params explicitly.")
        matrix, offset = f3d.invert_affine3d(params["matrix"], params["offset"])
        return {"matrix": matrix, "offset": offset}


class ElasticTransform3D(BaseGeometric3D):
    """Apply a random elastic deformation to a 3D volume as a whole.

    A smooth 3D displacement field is generated by trilinear interpolation of a coarse
    random control grid and applied jointly to all three dimensions. The same field is
    applied to the volume, its segmentation mask (with nearest-neighbor interpolation
    by default), multi-modal volumes and keypoints, keeping them strictly aligned.

    Args:
        alpha (float): Maximum displacement in voxels. Control point displacements are
            sampled uniformly from [-alpha, alpha]. Default: 1.0.
        sigma (float): Spacing between control points of the displacement grid, in
            voxels. Larger values produce smoother, more global deformations.
            Must be >= 1. Default: 8.0.
        interpolation (int): Interpolation order for image volumes: 0 (nearest) or
            1 (linear). Default: 1.
        mask_interpolation (int): Interpolation order for 3D masks: 0 (nearest) or
            1 (linear). Default: 0, which introduces no intermediate class values.
        fill (tuple[float, ...] | float): Value for voxels sampled outside the volume.
        fill_mask (tuple[float, ...] | float): Value for mask voxels sampled outside
            the volume.
        max_chunk_voxels (int | None): If set, the output is computed in depth slabs of
            at most this many voxels, bounding peak memory. Because the displacement
            field is interpolated from one shared global grid, slab boundaries are
            seamless. Default: None (process the whole volume at once).
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - This transform is NOT exactly invertible (``is_invertible == False``):
          inverting an elastic displacement field requires solving a nonlinear
          problem. The applied displacement grid is stored in the transform params
          (recoverable via ``get_applied_params()`` or ``ReplayCompose``) for
          visualization and analysis.
        - Multi-modal data: pass modalities stacked as ``volumes`` (shape (N, D, H, W)
          or (N, D, H, W, C)) or as a list; all modalities share one sampled
          displacement grid.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.random.randint(0, 256, (20, 128, 128), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (20, 128, 128), dtype=np.uint8)    # (D, H, W)
        >>> transform = A.Compose([
        ...     A.ElasticTransform3D(alpha=5.0, sigma=16.0, p=1.0),
        ... ])
        >>> transformed = transform(volume=volume, mask3d=mask3d)
        >>> transformed_volume, transformed_mask3d = transformed["volume"], transformed["mask3d"]

    """

    class InitSchema(BaseGeometric3D.InitSchema):
        alpha: float
        sigma: float

        @field_validator("alpha")
        @classmethod
        def validate_alpha(cls, v: float) -> float:
            """Validate that alpha is a finite non-negative number."""
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"alpha must be a finite non-negative number, got {v}")
            return v

        @field_validator("sigma")
        @classmethod
        def validate_sigma(cls, v: float) -> float:
            """Validate that sigma is finite and >= 1."""
            if not np.isfinite(v) or v < 1:
                raise ValueError(f"sigma must be a finite number >= 1, got {v}")
            return v

    def __init__(
        self,
        alpha: float = 1.0,
        sigma: float = 8.0,
        interpolation: int = 1,
        mask_interpolation: int = 0,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        max_chunk_voxels: int | None = None,
        p: float = 1.0,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            fill=fill,
            fill_mask=fill_mask,
            max_chunk_voxels=max_chunk_voxels,
            p=p,
        )
        self.alpha = alpha
        self.sigma = sigma

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Sample one displacement grid shared by all targets.

        Args:
            params (dict[str, Any]): Dictionary of existing parameters.
            data (dict[str, Any]): Dictionary containing input data with volume, mask3d, etc.

        Returns:
            dict[str, Any]: Dictionary with the sampled displacement grid.

        """
        shape_zyx = self._get_volume_shape(data)
        grid = f3d.generate_elastic_displacement_grid3d(shape_zyx, self.alpha, self.sigma, self.random_generator)
        return {"displacement_grid": grid}

    def apply_to_volume(self, volume: np.ndarray, displacement_grid: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the elastic deformation to a 3D volume.

        Args:
            volume (np.ndarray): Input volume of shape (D, H, W) or (D, H, W, C).
            displacement_grid (np.ndarray): Displacement grid of shape (3, nz, ny, nx).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Warped volume with the same shape and dtype as the input.

        """
        return f3d.warp_elastic3d(volume, displacement_grid, self.sigma, self.interpolation, self.fill, self.max_chunk_voxels)

    def apply_to_mask3d(self, mask3d: np.ndarray, displacement_grid: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the elastic deformation to a 3D mask with mask interpolation (nearest by default).

        Args:
            mask3d (np.ndarray): Input mask of shape (D, H, W) or (D, H, W, C).
            displacement_grid (np.ndarray): Displacement grid of shape (3, nz, ny, nx).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Warped mask with the same shape and dtype as the input.

        """
        return f3d.warp_elastic3d(
            mask3d,
            displacement_grid,
            self.sigma,
            self.mask_interpolation,
            self.fill_mask,
            self.max_chunk_voxels,
        )

    def apply_to_keypoints(self, keypoints: np.ndarray, displacement_grid: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the elastic displacement field to keypoints.

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
                The first three columns are x, y, z coordinates.
            displacement_grid (np.ndarray): Displacement grid of shape (3, nz, ny, nx).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Displaced keypoints with the same shape as the input.

        """
        return f3d.apply_elastic3d_to_keypoints(keypoints, displacement_grid, self.sigma)
