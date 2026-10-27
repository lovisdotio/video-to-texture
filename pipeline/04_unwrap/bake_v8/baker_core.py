#!/usr/bin/env python3
"""Small CPU reference primitives for the future v8 GPU baker.

These routines intentionally operate on synthetic depth/ownership arrays only.
They establish the two invariants that differ materially from baker v3:

* visibility uses a full and an opaque z-buffer;
* every texel, winner and fill domain is owned by ``piece_id``, not object ID.

Nothing in this module reads video, creates an RGB atlas or contacts a remote
service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DualZ:
    full_depth: np.ndarray
    full_piece_index: np.ndarray
    opaque_depth: np.ndarray
    opaque_piece_index: np.ndarray


def _nearest_layer(
    depth_layers: np.ndarray,
    piece_layers: np.ndarray,
    allowed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.where(allowed, depth_layers, np.inf)
    layer = np.argmin(candidate, axis=0)
    depth = np.take_along_axis(candidate, layer[None], axis=0)[0]
    piece = np.take_along_axis(piece_layers, layer[None], axis=0)[0]
    valid = np.isfinite(depth)
    return np.where(valid, depth, np.inf), np.where(valid, piece, -1).astype(np.int32)


def build_dual_z(
    depth_layers: np.ndarray,
    piece_layers: np.ndarray,
    piece_ids: Sequence[tuple[int, int]],
    opaque_z_excluded_object_ids: Sequence[int],
) -> DualZ:
    """Resolve layered fragments into full-z and transmissive-stripped z.

    ``depth_layers`` and ``piece_layers`` have shape ``[L,H,W]``.  Invalid
    fragments use piece index -1.  A valid depth must be finite and positive.
    """

    depth_layers = np.asarray(depth_layers, dtype=np.float64)
    piece_layers = np.asarray(piece_layers, dtype=np.int32)
    if depth_layers.shape != piece_layers.shape or depth_layers.ndim != 3:
        raise ValueError("depth_layers and piece_layers must have identical [L,H,W] shape")
    valid_index = (piece_layers >= 0) & (piece_layers < len(piece_ids))
    invalid_index = (piece_layers != -1) & ~valid_index
    if np.any(invalid_index):
        raise ValueError("A fragment references an unknown piece index")
    valid = valid_index & np.isfinite(depth_layers) & (depth_layers > 0.0)

    object_by_piece = np.asarray([piece[0] for piece in piece_ids], dtype=np.int32)
    safe_piece = np.clip(piece_layers, 0, max(0, len(piece_ids) - 1))
    fragment_object = object_by_piece[safe_piece]
    excluded = np.isin(
        fragment_object,
        np.asarray(tuple(opaque_z_excluded_object_ids), dtype=np.int32),
    )
    full_depth, full_piece = _nearest_layer(depth_layers, piece_layers, valid)
    opaque_depth, opaque_piece = _nearest_layer(
        depth_layers, piece_layers, valid & ~excluded
    )
    return DualZ(full_depth, full_piece, opaque_depth, opaque_piece)


def visibility_gate(
    target_depth: np.ndarray,
    target_piece_index: np.ndarray,
    piece_ids: Sequence[tuple[int, int]],
    dual_z: DualZ,
    opaque_z_excluded_object_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact piece/depth visibility and selected buffer code.

    Buffer code 1 means full-z and 0 means opaque-z.  The gate is deliberately
    stricter than object ownership: two subpieces of object 14 cannot authorize
    one another.
    """

    target_depth = np.asarray(target_depth, dtype=np.float64)
    target_piece_index = np.asarray(target_piece_index, dtype=np.int32)
    if target_depth.shape != target_piece_index.shape:
        raise ValueError("target depth and piece arrays must have identical shape")
    if target_depth.shape != dual_z.full_depth.shape:
        raise ValueError("Target arrays and dual-z images have different shapes")
    if np.any((target_piece_index < 0) | (target_piece_index >= len(piece_ids))):
        raise ValueError("A target references an unknown piece index")

    target_object = np.asarray(
        [piece_ids[int(index)][0] for index in target_piece_index.reshape(-1)],
        dtype=np.int32,
    ).reshape(target_piece_index.shape)
    use_full = np.isin(
        target_object,
        np.asarray(tuple(opaque_z_excluded_object_ids), dtype=np.int32),
    )
    selected_depth = np.where(use_full, dual_z.full_depth, dual_z.opaque_depth)
    selected_piece = np.where(
        use_full, dual_z.full_piece_index, dual_z.opaque_piece_index
    )
    tolerance = np.maximum(0.003, np.abs(target_depth) * 0.0015)
    visible = (
        np.isfinite(target_depth)
        & (target_depth > 0.0)
        & (selected_piece == target_piece_index)
        & (np.abs(selected_depth - target_depth) <= tolerance)
    )
    return visible, use_full.astype(np.uint8)


@dataclass(frozen=True)
class OwnerRaster:
    owner: np.ndarray
    coverage: np.ndarray
    collision_pixels: int


def rasterize_piece_owners(
    uv_triangles: np.ndarray,
    triangle_piece_index: np.ndarray,
    width: int,
    height: int,
) -> OwnerRaster:
    """Rasterize a tiny CPU ownership map for tests and preflight diagnostics.

    This is not a texture baker.  It writes only integer piece owners and an
    overlap count at pixel centres.
    """

    uv_triangles = np.asarray(uv_triangles, dtype=np.float64)
    triangle_piece_index = np.asarray(triangle_piece_index, dtype=np.int32)
    if uv_triangles.ndim != 3 or uv_triangles.shape[1:] != (3, 2):
        raise ValueError("uv_triangles must be [T,3,2]")
    if triangle_piece_index.shape != (len(uv_triangles),):
        raise ValueError("triangle_piece_index must contain one owner per triangle")
    if width <= 0 or height <= 0 or np.any(triangle_piece_index < 0):
        raise ValueError("Raster dimensions and piece indices must be positive")

    owner = np.full((height, width), -1, dtype=np.int32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    for triangle, piece_index in zip(uv_triangles, triangle_piece_index):
        if not np.all(np.isfinite(triangle)) or float(triangle.min()) < 0.0 or float(triangle.max()) > 1.0:
            raise ValueError("Synthetic UV triangle leaves the unit tile")
        a, b, c = triangle
        signed_twice_area = np.cross(b - a, c - a)
        if abs(float(signed_twice_area)) <= 1e-14:
            raise ValueError("Synthetic UV triangle is degenerate")
        x0 = max(0, int(np.floor(float(triangle[:, 0].min()) * width - 0.5)))
        x1 = min(width - 1, int(np.ceil(float(triangle[:, 0].max()) * width - 0.5)))
        y0 = max(0, int(np.floor(float(triangle[:, 1].min()) * height - 0.5)))
        y1 = min(height - 1, int(np.ceil(float(triangle[:, 1].max()) * height - 0.5)))
        for y in range(y0, y1 + 1):
            py = (y + 0.5) / height
            for x in range(x0, x1 + 1):
                px = (x + 0.5) / width
                p = np.asarray((px, py), dtype=np.float64)
                w0 = np.cross(b - p, c - p) / signed_twice_area
                w1 = np.cross(c - p, a - p) / signed_twice_area
                w2 = 1.0 - w0 - w1
                if min(float(w0), float(w1), float(w2)) >= -1e-12:
                    coverage[y, x] += 1
                    if owner[y, x] == -1:
                        owner[y, x] = int(piece_index)
    return OwnerRaster(
        owner=owner,
        coverage=coverage,
        collision_pixels=int(np.count_nonzero(coverage > 1)),
    )


def transfer_source_piece_ownership(
    source_to_proxy_face: np.ndarray,
    proxy_piece_index: np.ndarray,
) -> np.ndarray:
    source_to_proxy_face = np.asarray(source_to_proxy_face, dtype=np.int64)
    proxy_piece_index = np.asarray(proxy_piece_index, dtype=np.int32)
    if source_to_proxy_face.ndim != 1 or proxy_piece_index.ndim != 1:
        raise ValueError("Ownership transfer arrays must be one-dimensional")
    if len(proxy_piece_index) == 0 or np.any(source_to_proxy_face < 0) or np.any(
        source_to_proxy_face >= len(proxy_piece_index)
    ):
        raise ValueError("Source-to-proxy face map leaves the proxy domain")
    return proxy_piece_index[source_to_proxy_face]


def best_owned_scalar_sample(
    texel_owner: np.ndarray,
    sample_texel: np.ndarray,
    sample_piece: np.ndarray,
    sample_score: np.ndarray,
    sample_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference winner pass that rejects cross-piece samples before scoring."""

    texel_owner = np.asarray(texel_owner, dtype=np.int32).reshape(-1)
    sample_texel = np.asarray(sample_texel, dtype=np.int64).reshape(-1)
    sample_piece = np.asarray(sample_piece, dtype=np.int32).reshape(-1)
    sample_score = np.asarray(sample_score, dtype=np.float64).reshape(-1)
    sample_value = np.asarray(sample_value, dtype=np.float64).reshape(-1)
    lengths = {len(sample_texel), len(sample_piece), len(sample_score), len(sample_value)}
    if len(lengths) != 1:
        raise ValueError("All sample arrays must have equal length")
    if np.any(sample_texel < 0) or np.any(sample_texel >= len(texel_owner)):
        raise ValueError("A sample references a missing texel")
    best_score = np.full(len(texel_owner), -np.inf, dtype=np.float64)
    best_value = np.full(len(texel_owner), np.nan, dtype=np.float64)
    for texel, piece, score, value in zip(
        sample_texel, sample_piece, sample_score, sample_value
    ):
        if piece != texel_owner[texel] or not np.isfinite(score):
            continue
        if score > best_score[texel]:
            best_score[texel] = score
            best_value[texel] = value
    return best_value, best_score

