"""Module containing functional implementations of 3D transformations.

This module provides a collection of utility functions for manipulating and transforming
3D volumetric data (such as medical imaging volumes). The functions here implement the core
algorithms for operations like padding, cropping, rotation, and other spatial manipulations
specifically designed for 3D data.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
from scipy import ndimage

from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.type_definitions import NUM_VOLUME_DIMENSIONS

NUM_SPATIAL_DIMENSIONS = 3


def adjust_padding_by_position3d(
    paddings: list[tuple[int, int]],  # [(front, back), (top, bottom), (left, right)]
    position: Literal["center", "random"],
    py_random: random.Random,
) -> tuple[int, int, int, int, int, int]:
    """Adjust padding values based on desired position for 3D data.

    Args:
        paddings (list[tuple[int, int]]): List of tuples containing padding pairs
            for each dimension [(d_pad), (h_pad), (w_pad)]
        position (Literal["center", "random"]): Position of the image after padding.
        py_random (random.Random): Random number generator

    Returns:
        tuple[int, int, int, int, int, int]: Final padding values (d_front, d_back, h_top, h_bottom, w_left, w_right)

    """
    if position == "center":
        return (
            paddings[0][0],  # d_front
            paddings[0][1],  # d_back
            paddings[1][0],  # h_top
            paddings[1][1],  # h_bottom
            paddings[2][0],  # w_left
            paddings[2][1],  # w_right
        )

    # For random position, redistribute padding for each dimension
    d_pad = sum(paddings[0])
    h_pad = sum(paddings[1])
    w_pad = sum(paddings[2])

    return (
        py_random.randint(0, d_pad),  # d_front
        d_pad - py_random.randint(0, d_pad),  # d_back
        py_random.randint(0, h_pad),  # h_top
        h_pad - py_random.randint(0, h_pad),  # h_bottom
        py_random.randint(0, w_pad),  # w_left
        w_pad - py_random.randint(0, w_pad),  # w_right
    )


def pad_3d_with_params(
    volume: np.ndarray,
    padding: tuple[int, int, int, int, int, int],
    value: tuple[float, ...] | float,
) -> np.ndarray:
    """Pad 3D volume with given parameters.

    Args:
        volume (np.ndarray): Input volume with shape (depth, height, width) or (depth, height, width, channels)
        padding (tuple[int, int, int, int, int, int]): Padding values in format:
            (depth_front, depth_back, height_top, height_bottom, width_left, width_right)
            where:
            - depth_front/back: padding at start/end of depth axis (z)
            - height_top/bottom: padding at start/end of height axis (y)
            - width_left/right: padding at start/end of width axis (x)
        value (tuple[float, ...] | float): Value to fill the padding

    Returns:
        np.ndarray: Padded volume with same number of dimensions as input

    Note:
        The padding order matches the volume dimensions (depth, height, width).
        For each dimension, the first value is padding at the start (smaller indices),
        and the second value is padding at the end (larger indices).

    """
    depth_front, depth_back, height_top, height_bottom, width_left, width_right = padding

    # Skip if no padding is needed
    if all(p == 0 for p in padding):
        return volume

    # Handle both 3D and 4D arrays
    pad_width = [
        (depth_front, depth_back),  # depth (z) padding
        (height_top, height_bottom),  # height (y) padding
        (width_left, width_right),  # width (x) padding
    ]

    # Add channel padding if 4D array
    if volume.ndim == NUM_VOLUME_DIMENSIONS:
        pad_width.append((0, 0))  # no padding for channels

    return np.pad(
        volume,
        pad_width=pad_width,
        mode="constant",
        constant_values=value,
    )


def crop3d(
    volume: np.ndarray,
    crop_coords: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    """Crop 3D volume using coordinates.

    Args:
        volume (np.ndarray): Input volume with shape (z, y, x) or (z, y, x, channels)
        crop_coords (tuple[int, int, int, int, int, int]):
            (z_min, z_max, y_min, y_max, x_min, x_max) coordinates for cropping

    Returns:
        np.ndarray: Cropped volume with same number of dimensions as input

    """
    z_min, z_max, y_min, y_max, x_min, x_max = crop_coords

    return volume[z_min:z_max, y_min:y_max, x_min:x_max]


def cutout3d(volume: np.ndarray, holes: np.ndarray, fill: tuple[float, ...] | float) -> np.ndarray:
    """Cut out holes in 3D volume and fill them with a given value.

    Args:
        volume (np.ndarray): Input volume with shape (depth, height, width) or (depth, height, width, channels)
        holes (np.ndarray): Array of holes with shape (num_holes, 6).
            Each hole is represented as [z1, y1, x1, z2, y2, x2]
        fill (tuple[float, ...] | float): Value to fill the holes

    Returns:
        np.ndarray: Volume with holes filled with the given value

    """
    volume = volume.copy()
    for z1, y1, x1, z2, y2, x2 in holes:
        volume[z1:z2, y1:y2, x1:x2] = fill
    return volume


def transform_cube(cube: np.ndarray, index: int) -> np.ndarray:
    """Transform cube by index (0-47)

    Args:
        cube (np.ndarray): Input array with shape (D, H, W) or (D, H, W, C)
        index (int): Integer from 0 to 47 specifying which transformation to apply

    Returns:
        np.ndarray: Transformed cube with same shape as input

    """
    if not (0 <= index < 48):
        raise ValueError("Index must be between 0 and 47")

    transformations = {
        # First 4: rotate around axis 0 (indices 0-3)
        0: lambda x: x,
        1: lambda x: np.rot90(x, k=1, axes=(1, 2)),
        2: lambda x: np.rot90(x, k=2, axes=(1, 2)),
        3: lambda x: np.rot90(x, k=3, axes=(1, 2)),
        # Next 4: flip 180° about axis 1, then rotate around axis 0 (indices 4-7)
        4: lambda x: x[::-1, :, ::-1],  # was: np.flip(x, axis=(0, 2))
        5: lambda x: np.rot90(np.rot90(x, k=2, axes=(0, 2)), k=1, axes=(1, 2)),
        6: lambda x: x[::-1, ::-1, :],  # was: np.flip(x, axis=(0, 1))
        7: lambda x: np.rot90(np.rot90(x, k=2, axes=(0, 2)), k=3, axes=(1, 2)),
        # Next 8: split between 90° and 270° about axis 1, then rotate around axis 2 (indices 8-15)
        8: lambda x: np.rot90(x, k=1, axes=(0, 2)),
        9: lambda x: np.rot90(np.rot90(x, k=1, axes=(0, 2)), k=1, axes=(0, 1)),
        10: lambda x: np.rot90(np.rot90(x, k=1, axes=(0, 2)), k=2, axes=(0, 1)),
        11: lambda x: x.transpose(1, 2, 0, *range(3, x.ndim)),
        12: lambda x: np.rot90(x, k=-1, axes=(0, 2)),
        13: lambda x: np.rot90(np.rot90(x, k=-1, axes=(0, 2)), k=1, axes=(0, 1)),
        14: lambda x: np.rot90(np.rot90(x, k=-1, axes=(0, 2)), k=2, axes=(0, 1)),
        15: lambda x: np.rot90(np.rot90(x, k=-1, axes=(0, 2)), k=3, axes=(0, 1)),
        # Final 8: split between rotations about axis 2, then rotate around axis 1 (indices 16-23)
        16: lambda x: np.rot90(x, k=1, axes=(0, 1)),
        17: lambda x: np.rot90(np.rot90(x, k=1, axes=(0, 1)), k=1, axes=(0, 2)),
        18: lambda x: np.rot90(np.rot90(x, k=1, axes=(0, 1)), k=2, axes=(0, 2)),
        19: lambda x: x.transpose(2, 0, 1, *range(3, x.ndim)),
        20: lambda x: np.rot90(x, k=-1, axes=(0, 1)),
        21: lambda x: np.rot90(np.rot90(x, k=-1, axes=(0, 1)), k=1, axes=(0, 2)),
        22: lambda x: np.rot90(np.rot90(x, k=-1, axes=(0, 1)), k=2, axes=(0, 2)),
        23: lambda x: np.rot90(np.rot90(x, k=-1, axes=(0, 1)), k=3, axes=(0, 2)),
        # Reflected versions (24-47) - same as above but with initial reflection
        24: lambda x: x[:, :, ::-1],  # was: np.flip(x, axis=2)
        25: lambda x: x.transpose(0, 2, 1, *range(3, x.ndim)),
        26: lambda x: x[:, ::-1, :],  # was: np.flip(x, axis=1)
        27: lambda x: np.rot90(x[:, :, ::-1], k=3, axes=(1, 2)),
        28: lambda x: x[::-1, :, :],  # was: np.flip(x, axis=0)
        29: lambda x: np.rot90(x[::-1, :, :], k=1, axes=(1, 2)),
        30: lambda x: x[::-1, ::-1, ::-1],  # was: np.flip(x, axis=(0, 1, 2))
        31: lambda x: np.rot90(x[::-1, :, :], k=-1, axes=(1, 2)),
        32: lambda x: x.transpose(2, 1, 0, *range(3, x.ndim)),
        33: lambda x: x.transpose(1, 2, 0, *range(3, x.ndim))[::-1, :, :],
        34: lambda x: x.transpose(2, 1, 0, *range(3, x.ndim))[::-1, ::-1, :],
        35: lambda x: x.transpose(1, 2, 0, *range(3, x.ndim))[:, ::-1, :],
        36: lambda x: np.rot90(x[:, :, ::-1], k=-1, axes=(0, 2)),
        37: lambda x: x.transpose(1, 2, 0, *range(3, x.ndim))[::-1, ::-1, ::-1],
        38: lambda x: x.transpose(2, 1, 0, *range(3, x.ndim))[:, ::-1, ::-1],
        39: lambda x: x.transpose(1, 2, 0, *range(3, x.ndim))[:, :, ::-1],
        40: lambda x: np.rot90(x[:, :, ::-1], k=1, axes=(0, 1)),
        41: lambda x: x.transpose(2, 0, 1, *range(3, x.ndim))[:, :, ::-1],
        42: lambda x: x.transpose(1, 0, 2, *range(3, x.ndim)),
        43: lambda x: x.transpose(2, 0, 1, *range(3, x.ndim))[::-1, :, :],
        44: lambda x: np.rot90(x[:, :, ::-1], k=-1, axes=(0, 1)),
        45: lambda x: x.transpose(2, 0, 1, *range(3, x.ndim))[:, ::-1, :],
        46: lambda x: x.transpose(1, 0, 2, *range(3, x.ndim))[::-1, ::-1, :],
        47: lambda x: x.transpose(2, 0, 1, *range(3, x.ndim))[::-1, ::-1, ::-1],
    }

    return transformations[index](cube.copy())


@handle_empty_array("keypoints")
def filter_keypoints_in_holes3d(keypoints: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Filter out keypoints that are inside any of the 3D holes.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
                               The first three columns are x, y, z coordinates.
        holes (np.ndarray): Array of holes with shape (num_holes, 6).
                           Each hole is represented as [z1, y1, x1, z2, y2, x2].

    Returns:
        np.ndarray: Array of keypoints that are not inside any hole.

    """
    if holes.size == 0:
        return keypoints

    # Broadcast keypoints and holes for vectorized comparison
    # Convert keypoints from XYZ to ZYX for comparison with holes
    kp_z = keypoints[:, 2][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_y = keypoints[:, 1][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_x = keypoints[:, 0][:, np.newaxis]  # Shape: (num_keypoints, 1)

    # Extract hole coordinates (in ZYX order)
    hole_z1 = holes[:, 0]  # Shape: (num_holes,)
    hole_y1 = holes[:, 1]
    hole_x1 = holes[:, 2]
    hole_z2 = holes[:, 3]
    hole_y2 = holes[:, 4]
    hole_x2 = holes[:, 5]

    # Check if each keypoint is inside each hole
    inside_hole = (
        (kp_z >= hole_z1)
        & (kp_z < hole_z2)
        & (kp_y >= hole_y1)
        & (kp_y < hole_y2)
        & (kp_x >= hole_x1)
        & (kp_x < hole_x2)
    )

    # A keypoint is valid if it's not inside any hole
    valid_keypoints = ~np.any(inside_hole, axis=1)

    # Return filtered keypoints with same dtype as input
    result = keypoints[valid_keypoints]
    if len(result) == 0:
        # Ensure empty result has correct shape and dtype
        return np.array([], dtype=keypoints.dtype).reshape(0, keypoints.shape[1])
    return result


def keypoints_rot90(
    keypoints: np.ndarray,
    k: int,
    axes: tuple[int, int],
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Rotate keypoints 90 degrees k times around the specified axes.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
                               The first three columns are x, y, z coordinates.
        k (int): Number of times to rotate by 90 degrees.
        axes (tuple[int, int]): Axes to rotate around.
        volume_shape (tuple[int, int, int]): Shape of the volume (depth, height, width).

    Returns:
        np.ndarray: Rotated keypoints with same shape as input.

    """
    if k == 0 or len(keypoints) == 0:
        return keypoints

    # Normalize factor to range [0, 3]
    k = ((k % 4) + 4) % 4

    result = keypoints.copy()

    # Get dimensions for the rotation axes
    dims = [volume_shape[ax] for ax in axes]

    # Get coordinates to rotate
    coords1 = result[:, axes[0]].copy()
    coords2 = result[:, axes[1]].copy()

    # Apply rotation based on factor (counterclockwise)
    if k == 1:  # 90 degrees CCW
        result[:, axes[0]] = (dims[1] - 1) - coords2
        result[:, axes[1]] = coords1
    elif k == 2:  # 180 degrees
        result[:, axes[0]] = (dims[0] - 1) - coords1
        result[:, axes[1]] = (dims[1] - 1) - coords2
    elif k == 3:  # 270 degrees CCW
        result[:, axes[0]] = coords2
        result[:, axes[1]] = (dims[0] - 1) - coords1

    return result


@handle_empty_array("keypoints")
def transform_cube_keypoints(
    keypoints: np.ndarray,
    index: int,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Transform keypoints according to the cube transformation specified by index.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
                               The first three columns are x, y, z coordinates.
        index (int): Integer from 0 to 47 specifying which transformation to apply.
        volume_shape (tuple[int, int, int]): Shape of the volume (depth, height, width).

    Returns:
        np.ndarray: Transformed keypoints with same shape as input.

    """
    if not (0 <= index < 48):
        raise ValueError("Index must be between 0 and 47")

    # Create working copy preserving all columns
    working_points = keypoints.copy()

    # Convert only XYZ coordinates to HWD, keeping other columns unchanged
    xyz = working_points[:, :3]  # Get first 3 columns (XYZ)
    xyz = xyz[:, [2, 1, 0]]  # XYZ -> HWD
    working_points[:, :3] = xyz  # Put back transformed coordinates

    current_shape = volume_shape

    # Handle reflection first (indices 24-47)
    if index >= 24:
        working_points[:, 2] = current_shape[2] - 1 - working_points[:, 2]  # Reflect W axis

    rotation_index = index % 24

    # Apply the same rotation logic as transform_cube
    if rotation_index < 4:
        # First 4: rotate around axis 0
        result = keypoints_rot90(working_points, k=rotation_index, axes=(1, 2), volume_shape=current_shape)
    elif rotation_index < 8:
        # Next 4: flip 180° about axis 1, then rotate around axis 0
        temp = keypoints_rot90(working_points, k=2, axes=(0, 2), volume_shape=current_shape)
        result = keypoints_rot90(temp, k=rotation_index - 4, axes=(1, 2), volume_shape=volume_shape)
    elif rotation_index < 16:
        if rotation_index < 12:
            temp = keypoints_rot90(working_points, k=1, axes=(0, 2), volume_shape=current_shape)
            temp_shape = (current_shape[2], current_shape[1], current_shape[0])
            result = keypoints_rot90(temp, k=rotation_index - 8, axes=(0, 1), volume_shape=temp_shape)
        else:
            temp = keypoints_rot90(working_points, k=3, axes=(0, 2), volume_shape=current_shape)
            temp_shape = (current_shape[2], current_shape[1], current_shape[0])
            result = keypoints_rot90(temp, k=rotation_index - 12, axes=(0, 1), volume_shape=temp_shape)
    elif rotation_index < 20:
        temp = keypoints_rot90(working_points, k=1, axes=(0, 1), volume_shape=current_shape)
        temp_shape = (current_shape[1], current_shape[0], current_shape[2])
        result = keypoints_rot90(temp, k=rotation_index - 16, axes=(0, 2), volume_shape=temp_shape)
    else:
        temp = keypoints_rot90(working_points, k=3, axes=(0, 1), volume_shape=current_shape)
        temp_shape = (current_shape[1], current_shape[0], current_shape[2])
        result = keypoints_rot90(temp, k=rotation_index - 20, axes=(0, 2), volume_shape=temp_shape)

    # Convert back from HWD to XYZ coordinates for first 3 columns only
    xyz = result[:, :3]
    xyz = xyz[:, [2, 1, 0]]  # HWD -> XYZ
    result[:, :3] = xyz

    return result


def get_rotation_matrix_3d(angles_xyz: tuple[float, float, float]) -> np.ndarray:
    """Build a 3D rotation matrix from per-axis angles.

    Args:
        angles_xyz (tuple[float, float, float]): Rotation angles in degrees around
            the x, y and z axes respectively. A rotation around the z axis takes
            place in the xy (axial) plane, around y in the xz (coronal) plane and
            around x in the yz (sagittal) plane.

    Returns:
        np.ndarray: Rotation matrix of shape (3, 3) in (x, y, z) coordinates.
            The combined rotation applies x first, then y, then z: ``R = Rz @ Ry @ Rx``.

    """
    ax, ay, az = (np.deg2rad(angle) for angle in angles_xyz)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return rotation_z @ rotation_y @ rotation_x


def get_affine3d_forward_matrix(
    rotation_matrix: np.ndarray,
    scale: float,
    flip_axes: tuple[bool, bool, bool],
) -> np.ndarray:
    """Build the forward affine matrix mapping input to output coordinates.

    The forward transform is ``p_out = A @ (p_in - center) + center`` with
    ``A = scale * R @ F`` where ``F`` is a diagonal matrix of +/-1 flips.

    Args:
        rotation_matrix (np.ndarray): Rotation matrix (3, 3) in (x, y, z) coordinates.
        scale (float): Isotropic scale factor. Must be non-zero.
        flip_axes (tuple[bool, bool, bool]): Whether to flip the x, y and z axes.

    Returns:
        np.ndarray: Forward affine matrix of shape (3, 3) in (x, y, z) coordinates.

    """
    if scale == 0:
        raise ValueError("Scale must be non-zero for a 3D affine transform")
    flips = np.array([-1.0 if flip else 1.0 for flip in flip_axes])
    return scale * rotation_matrix @ np.diag(flips)


def resampling_map_from_forward(
    forward_matrix: np.ndarray,
    center_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a forward affine matrix to a resampling map in voxel index order.

    Resampling requires the output-to-input mapping ``p_in = M @ p_out + b``
    expressed in (z, y, x) index coordinates, which is what
    ``scipy.ndimage.affine_transform`` consumes.

    Args:
        forward_matrix (np.ndarray): Forward affine matrix (3, 3) in (x, y, z) coordinates.
        center_xyz (tuple[float, float, float]): Center of the transform in (x, y, z) coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(matrix, offset)`` in (z, y, x) index order such that
            ``p_in = matrix @ p_out + offset``.

    """
    center = np.asarray(center_xyz, dtype=np.float64)
    inverse = np.linalg.inv(forward_matrix)
    offset_xyz = center - inverse @ center
    # Convert (x, y, z) -> (z, y, x): P @ M @ P with P the reversal permutation (P = P^-1)
    matrix_zyx = inverse[::-1, ::-1]
    offset_zyx = offset_xyz[::-1]
    return matrix_zyx, offset_zyx


def invert_affine3d(
    matrix: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a resampling map.

    Args:
        matrix (np.ndarray): Resampling matrix (3, 3) such that ``p_in = matrix @ p_out + offset``.
        offset (np.ndarray): Resampling offset (3,).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(matrix, offset)`` of the inverse resampling map.
            Applying it to the transformed volume restores the original space
            (up to interpolation error).

    """
    inverse = np.linalg.inv(matrix)
    return inverse, -inverse @ offset


def _validate_interpolation(interpolation: int) -> None:
    if interpolation not in {0, 1}:
        raise ValueError(
            f"Only nearest-neighbor (0) and linear (1) interpolation are supported for 3D "
            f"geometric transforms, got {interpolation}. Higher-order splines are not "
            f"supported because they cannot be applied seamlessly in chunks.",
        )


def _iter_depth_slabs(
    shape_zyx: tuple[int, int, int],
    max_chunk_voxels: int | None,
) -> list[tuple[int, int]]:
    """Split the depth axis into slabs of at most max_chunk_voxels voxels each.

    Args:
        shape_zyx (tuple[int, int, int]): Spatial shape (depth, height, width).
        max_chunk_voxels (int | None): Maximum number of voxels per slab. If None,
            a single slab covering the whole volume is returned.

    Returns:
        list[tuple[int, int]]: List of (z_start, z_stop) depth ranges.

    """
    depth, height, width = shape_zyx
    if max_chunk_voxels is None:
        return [(0, depth)]
    if max_chunk_voxels < 1:
        raise ValueError(f"max_chunk_voxels must be >= 1, got {max_chunk_voxels}")
    slab_depth = max(1, max_chunk_voxels // (height * width))
    return [(z0, min(depth, z0 + slab_depth)) for z0 in range(0, depth, slab_depth)]


def _split_channels(volume: np.ndarray) -> tuple[list[np.ndarray], tuple[int, ...]]:
    """Split a (D, H, W) or (D, H, W, C) volume into a list of (D, H, W) channel arrays."""
    if volume.ndim == NUM_SPATIAL_DIMENSIONS:
        return [volume], ()
    if volume.ndim == NUM_VOLUME_DIMENSIONS:
        return [volume[..., c] for c in range(volume.shape[-1])], (volume.shape[-1],)
    raise ValueError(f"Volume must be 3D (D, H, W) or 4D (D, H, W, C), got shape {volume.shape}")


def _normalize_fill(fill: tuple[float, ...] | float, num_channels: int) -> list[float]:
    """Broadcast a scalar or per-channel fill value to a per-channel list."""
    if isinstance(fill, (tuple, list)):
        if len(fill) != num_channels:
            raise ValueError(
                f"Fill value has {len(fill)} entries but the volume has {num_channels} channel(s). "
                f"Pass a scalar or one value per channel.",
            )
        return [float(v) for v in fill]
    return [float(fill)] * num_channels


def warp_affine3d(
    volume: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    interpolation: int,
    fill: tuple[float, ...] | float,
    max_chunk_voxels: int | None = None,
) -> np.ndarray:
    """Warp a 3D volume with an affine resampling map, optionally in depth slabs.

    Applies ``p_in = matrix @ p_out + offset`` to resample the volume. Only the
    three spatial axes are transformed; trailing channel axes are preserved and
    every channel is warped with the same map, so multi-channel and multi-modal
    data stay aligned.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W) or (D, H, W, C).
        matrix (np.ndarray): Resampling matrix (3, 3) in (z, y, x) index order.
        offset (np.ndarray): Resampling offset (3,) in (z, y, x) index order.
        interpolation (int): 0 (nearest neighbor) or 1 (linear). Use 0 for label masks
            so no intermediate class values are introduced.
        fill (tuple[float, ...] | float): Value used for voxels sampled outside the volume.
        max_chunk_voxels (int | None): If set, the output is computed in depth slabs of at
            most this many voxels, bounding peak memory. The resampling map is a global
            function of the output coordinates, so slab boundaries are seamless and the
            result is identical to processing the whole volume at once.

    Returns:
        np.ndarray: Warped volume with the same shape and dtype as the input.

    """
    _validate_interpolation(interpolation)
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if matrix.shape != (NUM_SPATIAL_DIMENSIONS, NUM_SPATIAL_DIMENSIONS) or offset.shape != (NUM_SPATIAL_DIMENSIONS,):
        raise ValueError(f"matrix must have shape (3, 3) and offset shape (3,), got {matrix.shape} and {offset.shape}")

    channels, channel_shape = _split_channels(volume)
    fills = _normalize_fill(fill, len(channels))

    spatial_shape = volume.shape[:3]
    result = np.empty(spatial_shape + channel_shape, dtype=volume.dtype)

    for z0, z1 in _iter_depth_slabs(spatial_shape, max_chunk_voxels):
        # Shift the map so that slab-local coordinates index into the global output space
        slab_offset = offset + matrix @ np.array([z0, 0.0, 0.0])
        slab_shape = (z1 - z0, *spatial_shape[1:])
        for c, (channel, cval) in enumerate(zip(channels, fills)):
            slab_out = result[z0:z1] if not channel_shape else result[z0:z1, ..., c]
            ndimage.affine_transform(
                channel,
                matrix,
                offset=slab_offset,
                output_shape=slab_shape,
                output=slab_out,
                order=interpolation,
                mode="constant",
                cval=cval,
                prefilter=False,
            )
    return result


def generate_elastic_displacement_grid3d(
    shape_zyx: tuple[int, int, int],
    alpha: float,
    sigma: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate a coarse random displacement grid for 3D elastic deformation.

    Displacements are sampled at control points spaced roughly ``sigma`` voxels
    apart; the dense displacement field is obtained by trilinear interpolation
    of this grid, so the field is smooth and continuous across the whole volume
    (and across chunk boundaries when processing in slabs).

    Args:
        shape_zyx (tuple[int, int, int]): Spatial shape (depth, height, width) of the volume.
        alpha (float): Maximum displacement in voxels (sampled uniformly in [-alpha, alpha]).
        sigma (float): Spacing between control points in voxels. Must be >= 1.
        random_generator (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Displacement grid of shape (3, nz, ny, nx), float32, where the first
            axis holds the z, y and x displacement components and control point (i, j, k)
            corresponds to voxel coordinate (i * sigma, j * sigma, k * sigma).

    """
    if sigma < 1:
        raise ValueError(f"sigma must be >= 1, got {sigma}")
    grid_shape = tuple(max(2, int(np.ceil(d / sigma)) + 1) for d in shape_zyx)
    return random_generator.uniform(-alpha, alpha, size=(NUM_SPATIAL_DIMENSIONS, *grid_shape)).astype(np.float32)


def warp_elastic3d(
    volume: np.ndarray,
    displacement_grid: np.ndarray,
    grid_step: float,
    interpolation: int,
    fill: tuple[float, ...] | float,
    max_chunk_voxels: int | None = None,
) -> np.ndarray:
    """Warp a 3D volume with a coarse displacement grid, optionally in depth slabs.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W) or (D, H, W, C).
        displacement_grid (np.ndarray): Grid of shape (3, nz, ny, nx) as produced by
            ``generate_elastic_displacement_grid3d``. The same grid must be used for the
            volume and its mask to keep them aligned.
        grid_step (float): Spacing between control points in voxels.
        interpolation (int): 0 (nearest neighbor) or 1 (linear). Use 0 for label masks.
        fill (tuple[float, ...] | float): Value used for voxels sampled outside the volume.
        max_chunk_voxels (int | None): If set, the output is computed in depth slabs of at
            most this many voxels. Because displacements are interpolated from the shared
            global grid, slab boundaries are seamless and the result is identical to
            processing the whole volume at once.

    Returns:
        np.ndarray: Warped volume with the same shape and dtype as the input.

    """
    _validate_interpolation(interpolation)
    if displacement_grid.ndim != NUM_VOLUME_DIMENSIONS or displacement_grid.shape[0] != NUM_SPATIAL_DIMENSIONS:
        raise ValueError(f"displacement_grid must have shape (3, nz, ny, nx), got {displacement_grid.shape}")

    channels, channel_shape = _split_channels(volume)
    fills = _normalize_fill(fill, len(channels))

    spatial_shape = volume.shape[:3]
    height, width = spatial_shape[1:]
    result = np.empty(spatial_shape + channel_shape, dtype=volume.dtype)

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    for z0, z1 in _iter_depth_slabs(spatial_shape, max_chunk_voxels):
        zz = np.arange(z0, z1, dtype=np.float32)[:, None, None]
        zz, yy3, xx3 = np.broadcast_arrays(zz, yy[None], xx[None])
        grid_coords = [zz / grid_step, yy3 / grid_step, xx3 / grid_step]
        # Interpolate the shared global displacement grid at this slab's coordinates
        sample_coords = [
            coord + ndimage.map_coordinates(displacement_grid[c], grid_coords, order=1, mode="nearest")
            for c, coord in enumerate((zz, yy3, xx3))
        ]
        for c, (channel, cval) in enumerate(zip(channels, fills)):
            slab_out = result[z0:z1] if not channel_shape else result[z0:z1, ..., c]
            ndimage.map_coordinates(
                channel,
                sample_coords,
                output=slab_out,
                order=interpolation,
                mode="constant",
                cval=cval,
                prefilter=False,
            )
    return result


def elastic_displacement_at_points3d(
    displacement_grid: np.ndarray,
    grid_step: float,
    points_zyx: np.ndarray,
) -> np.ndarray:
    """Sample the displacement field at arbitrary points.

    Args:
        displacement_grid (np.ndarray): Grid of shape (3, nz, ny, nx).
        grid_step (float): Spacing between control points in voxels.
        points_zyx (np.ndarray): Points of shape (N, 3) in (z, y, x) voxel coordinates.

    Returns:
        np.ndarray: Displacements of shape (N, 3) in (z, y, x) order.

    """
    if points_zyx.size == 0:
        return np.zeros((0, NUM_SPATIAL_DIMENSIONS), dtype=np.float32)
    grid_coords = np.ascontiguousarray(points_zyx.T / grid_step)
    return np.stack(
        [ndimage.map_coordinates(displacement_grid[c], grid_coords, order=1, mode="nearest") for c in range(3)],
        axis=1,
    )


@handle_empty_array("keypoints")
def apply_affine3d_to_keypoints(
    keypoints: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Apply the forward 3D affine transform to keypoints.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
            The first three columns are x, y, z coordinates.
        matrix (np.ndarray): Resampling matrix (3, 3) in (z, y, x) index order, as used
            for warping the volume (``p_in = matrix @ p_out + offset``).
        offset (np.ndarray): Resampling offset (3,) in (z, y, x) index order.

    Returns:
        np.ndarray: Transformed keypoints with the same shape as the input.

    """
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    inverse = np.linalg.inv(matrix)

    result = keypoints.copy().astype(np.float64)
    # Keypoints are (x, y, z); the resampling map works in (z, y, x) index order
    zyx = result[:, :3][:, ::-1]
    # Forward map: p_out = M^-1 @ (p_in - offset)
    transformed = (zyx - offset) @ inverse.T
    result[:, :3] = transformed[:, ::-1]
    return result.astype(keypoints.dtype) if np.issubdtype(keypoints.dtype, np.integer) else result


@handle_empty_array("keypoints")
def apply_elastic3d_to_keypoints(
    keypoints: np.ndarray,
    displacement_grid: np.ndarray,
    grid_step: float,
) -> np.ndarray:
    """Apply a 3D elastic displacement field to keypoints.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
            The first three columns are x, y, z coordinates.
        displacement_grid (np.ndarray): Grid of shape (3, nz, ny, nx).
        grid_step (float): Spacing between control points in voxels.

    Returns:
        np.ndarray: Displaced keypoints with the same shape as the input.

    """
    result = keypoints.copy()
    zyx = result[:, :3][:, ::-1].astype(np.float32)
    displacement = elastic_displacement_at_points3d(displacement_grid, grid_step, zyx)
    result[:, :3] = (zyx + displacement)[:, ::-1]
    return result


_CUBE_INVERSE_TABLE: list[int] | None = None


def _build_cube_inverse_table() -> list[int]:
    """Build the inverse lookup table for the 48 cubic symmetry transformations.

    The 48 transformations form the octahedral group, so every transformation has
    its inverse inside the group. The table is computed once by probing with a
    small asymmetric-valued cube.
    """
    probe = np.arange(27).reshape(3, 3, 3)
    transformed = [transform_cube(probe, index) for index in range(48)]
    table = []
    for index in range(48):
        for candidate in range(48):
            if np.array_equal(transform_cube(transformed[index], candidate), probe):
                table.append(candidate)
                break
        else:  # pragma: no cover - the cube group is closed under inversion
            raise ValueError(f"No inverse found for cube transformation {index}")
    return table


def invert_cube_index(index: int) -> int:
    """Return the index of the inverse cubic symmetry transformation.

    Args:
        index (int): Transformation index (0-47) as used by ``transform_cube``.

    Returns:
        int: Index j such that ``transform_cube(transform_cube(x, index), j) == x``.

    """
    if not (0 <= index < 48):
        raise ValueError("Index must be between 0 and 47")
    global _CUBE_INVERSE_TABLE
    if _CUBE_INVERSE_TABLE is None:
        _CUBE_INVERSE_TABLE = _build_cube_inverse_table()
    return _CUBE_INVERSE_TABLE[index]
