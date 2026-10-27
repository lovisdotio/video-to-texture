#!/usr/bin/env python3
"""Reproject one approved v8/v6 atlas onto one final page-repack UV layout.

The tool is deliberately local and CPU-only.  It reads the additive
``fixed_keep_placement_report.json`` and every ``page_layout_sidecar`` named by
that report, then maps each destination pixel centre through the new triangle
to barycentric coordinates and back into the corresponding old UV triangle.
The approved Seedream v5 (or BODY v6) atlas is sampled bilinearly.

Outputs are written as a new directory through an atomic directory rename:

* padded RGB PNG;
* occupied-only object/chart owner PNGs and occupancy mask;
* padded object/chart owner PNGs and padded mask;
* a coverage/hash report and a SHA-256 inventory.

Gutters use one deterministic nearest occupied donor.  RGB values are never
averaged across object/chart domains.  This file never launches Blender, Fal,
Seedream, GitHub, a texture bake, or a canonical build.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping

import cv2
import numpy as np


SCHEMA = "car-city.ferrari-page-texture-reprojection/v9"
LAYOUT_SCHEMA = "car-city.ferrari-page-repack-uv-sidecar/v9"
EXPECTED_PLACEMENT_SCHEMA = "car-city.ferrari-targeted-fixed-keep-placement/v9"
PADDING_PIXELS = 16
PNG_COMPRESSION = 6
ROW_CHUNK = 128
UV_TOLERANCE = 2.0e-6
BLACK_THRESHOLD = 0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    # Match targeted_reunwrap_v9.metrics_v9.hash_array exactly.  Page-layout
    # sidecars bind dtype and shape as well as payload bytes, so a raw-byte-only
    # digest would reject every legitimate RUNNING/BODY authority.
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def scalar(payload: Mapping[str, np.ndarray], name: str) -> Any:
    require(name in payload, f"Missing scalar field {name}")
    value = np.asarray(payload[name])
    require(value.size == 1, f"Field {name} is not scalar")
    return value.reshape(-1)[0].item()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    require(path.is_file(), f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name].copy() for name in handle.files}


def resolve_input_path(value: str | Path, report_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    alternatives = [candidate]
    if not candidate.is_absolute():
        alternatives.append(report_dir / candidate)
    for path in alternatives:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(value)


def triangle_double_area(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.float64)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    return edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]


def uv_area(triangles: np.ndarray) -> float:
    return float(0.5 * np.abs(triangle_double_area(triangles)).sum())


def load_source_geometry(source: Path) -> tuple[np.ndarray, np.ndarray, str]:
    payload = read_npz(source)
    if "proxy_faces" in payload and "proxy_corner_uv" in payload:
        faces = np.asarray(payload["proxy_faces"], dtype=np.int32)
        corner_uv = np.asarray(payload["proxy_corner_uv"], dtype=np.float32)
        kind = "nonbody_proxy"
    elif "render_faces" in payload and "render_corner_uv" in payload:
        faces = np.asarray(payload["render_faces"], dtype=np.int32)
        corner_uv = np.asarray(payload["render_corner_uv"], dtype=np.float32)
        kind = "body_render_proxy"
    else:
        raise RuntimeError(f"Unsupported page-layout source payload: {source}")
    require(faces.ndim == 2 and faces.shape[1] == 3, f"Bad source faces: {source}")
    require(corner_uv.shape == (len(faces), 3, 2), f"Bad source UVs: {source}")
    require(bool(np.all(np.isfinite(corner_uv))), f"Non-finite source UV: {source}")
    require(
        float(corner_uv.min()) >= -UV_TOLERANCE
        and float(corner_uv.max()) <= 1.0 + UV_TOLERANCE,
        f"Source UV leaves the unit page: {source}",
    )
    return faces, corner_uv, kind


def validate_placement_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).expanduser().resolve()
    require(path.is_file(), f"Missing page-repack report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("schema") == EXPECTED_PLACEMENT_SCHEMA, "Placement report schema drift")
    require(report.get("status") == "passed", "Page-repack report did not pass")
    require(report.get("mode") == "page_repack", "Texture reprojection requires page_repack mode")
    gates = report.get("hard_gates")
    require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()), "Placement hard gates are not all true")
    page_layout = report.get("page_layout")
    require(isinstance(page_layout, dict), "Placement report has no final page_layout")
    require(page_layout.get("schema") == "car-city.ferrari-page-repack-uv-layout/v9", "Page-layout schema drift")
    require(page_layout.get("page_id") == report.get("page_id"), "Page ID differs between placement and layout")
    resolution = report.get("resolution")
    require(
        isinstance(resolution, list)
        and len(resolution) == 2
        and resolution[0] == resolution[1]
        and page_layout.get("resolution") == resolution,
        "Page resolution contract drift",
    )
    objects = page_layout.get("objects")
    require(isinstance(objects, list) and objects, "Page layout has no object sidecars")
    return report, objects


def load_layout_object(
    row: Mapping[str, Any],
    *,
    report_dir: Path,
    page_id: str,
    native_resolution: int,
    chart_offset: int,
) -> tuple[dict[str, Any], int]:
    layout_path = resolve_input_path(str(row.get("path", "")), report_dir)
    expected_layout_sha = str(row.get("sha256", ""))
    actual_layout_sha = sha256_file(layout_path)
    require(actual_layout_sha == expected_layout_sha, f"Layout SHA drift: {layout_path}")
    payload = read_npz(layout_path)
    require(str(scalar(payload, "schema")) == LAYOUT_SCHEMA, f"Layout schema drift: {layout_path}")
    object_id = int(scalar(payload, "source_object_id"))
    require(object_id == int(row.get("object_id", -1)), f"Object identity drift: {layout_path}")
    require(str(scalar(payload, "target_page_id")) == page_id, f"Layout page drift: {layout_path}")
    require(int(scalar(payload, "page_resolution")) == native_resolution, f"Layout resolution drift: {layout_path}")

    source_path = resolve_input_path(str(scalar(payload, "source_path")), report_dir)
    source_sha = str(scalar(payload, "source_sha256"))
    require(sha256_file(source_path) == source_sha, f"Source authority SHA drift: {source_path}")
    faces, old_uv, source_kind = load_source_geometry(source_path)
    require(
        sha256_array(faces) == str(scalar(payload, "source_faces_sha256")),
        f"Source face hash drift: {source_path}",
    )
    require(
        sha256_array(old_uv) == str(scalar(payload, "baseline_corner_uv_sha256")),
        f"Baseline UV hash drift: {source_path}",
    )
    new_uv = np.asarray(payload["final_proxy_corner_uv"], dtype=np.float32)
    local_chart = np.asarray(payload["final_proxy_chart_id"], dtype=np.int32)
    target_mask = np.asarray(payload["targeted_face_mask"], dtype=np.uint8)
    require(new_uv.shape == old_uv.shape, f"Old/new face correspondence drift: {layout_path}")
    require(local_chart.shape == target_mask.shape == (len(faces),), f"Layout face metadata drift: {layout_path}")
    require(bool(np.all(np.isfinite(new_uv))), f"Non-finite final UV: {layout_path}")
    require(
        float(new_uv.min()) >= -UV_TOLERANCE
        and float(new_uv.max()) <= 1.0 + UV_TOLERANCE,
        f"Final UV leaves the unit page: {layout_path}",
    )
    require(bool(np.all(local_chart >= 0)), f"Negative chart ID: {layout_path}")
    unique_chart = np.unique(local_chart)
    require(
        np.array_equal(unique_chart, np.arange(len(unique_chart), dtype=np.int32)),
        f"Local chart IDs are not dense: {layout_path}",
    )
    global_chart = local_chart + int(chart_offset)
    next_offset = int(chart_offset + len(unique_chart))
    record = {
        "object_id": object_id,
        "layout_path": layout_path,
        "layout_sha256": actual_layout_sha,
        "source_path": source_path,
        "source_sha256": source_sha,
        "source_kind": source_kind,
        "faces": faces,
        "old_uv": old_uv,
        "new_uv": new_uv,
        "global_chart": global_chart,
        "local_chart_count": int(len(unique_chart)),
        "target_mask": target_mask.astype(bool),
        "analytic_uv_area_fraction": uv_area(new_uv),
    }
    return record, next_offset


def bilinear_sample_bgr(
    source_bgr: np.ndarray,
    uv: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample an image using UV edge coordinates and border clamping.

    PNG rows are top-to-bottom while UV v grows bottom-to-top.  Pixel centres
    therefore use ``x=u*W-.5`` and ``y=(1-v)*H-.5``.  Interpolation happens in
    display-referred uint8/sRGB code values and is rounded back to uint8.
    """

    uv = np.asarray(uv, dtype=np.float64)
    require(uv.ndim == 2 and uv.shape[1] == 2, "Bilinear sample UV must be Nx2")
    require(bool(np.all(np.isfinite(uv))), "Bilinear sample UV is non-finite")
    outside = (
        (uv[:, 0] < -UV_TOLERANCE)
        | (uv[:, 0] > 1.0 + UV_TOLERANCE)
        | (uv[:, 1] < -UV_TOLERANCE)
        | (uv[:, 1] > 1.0 + UV_TOLERANCE)
    )
    require(not bool(np.any(outside)), "Mapped source UV leaves the unit atlas")
    height, width = source_bgr.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.clip(u * width - 0.5, 0.0, width - 1.0)
    y = np.clip((1.0 - v) * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    top = (
        source_bgr[y0, x0].astype(np.float32) * (1.0 - fx[:, None])
        + source_bgr[y0, x1].astype(np.float32) * fx[:, None]
    )
    bottom = (
        source_bgr[y1, x0].astype(np.float32) * (1.0 - fx[:, None])
        + source_bgr[y1, x1].astype(np.float32) * fx[:, None]
    )
    sampled = np.rint(top * (1.0 - fy[:, None]) + bottom * fy[:, None])
    sampled = np.clip(sampled, 0, 255).astype(np.uint8)

    # A bilinear footprint can round to exact black while its nearest source
    # texel is non-black.  Such newly introduced black is replaced with that
    # exact source texel.  If nearest is also black, black is source-authentic
    # and is retained (important for carbon, leather and dark trim).
    nearest_x = np.minimum(np.floor(u * width).astype(np.int32), width - 1)
    nearest_y = np.minimum(np.floor((1.0 - v) * height).astype(np.int32), height - 1)
    nearest = source_bgr[nearest_y, nearest_x]
    bilinear_black = np.max(sampled, axis=1) <= BLACK_THRESHOLD
    nearest_black = np.max(nearest, axis=1) <= BLACK_THRESHOLD
    fallback = bilinear_black & ~nearest_black
    sampled[fallback] = nearest[fallback]
    retained_black = np.max(sampled, axis=1) <= BLACK_THRESHOLD
    unexplained_black = retained_black & ~nearest_black
    require(not bool(np.any(unexplained_black)), "A newly introduced black texel survived source fallback")
    return sampled, {
        "samples": int(len(uv)),
        "bilinear_exact_black_before_fallback": int(bilinear_black.sum()),
        "nearest_source_fallback_texels": int(fallback.sum()),
        "source_confirmed_black_texels": int(retained_black.sum()),
        "unexplained_new_black_texels": int(unexplained_black.sum()),
    }


def _pixel_bounds(triangle: np.ndarray, size: int) -> tuple[int, int, int, int] | None:
    u_min = float(triangle[:, 0].min())
    u_max = float(triangle[:, 0].max())
    v_min = float(triangle[:, 1].min())
    v_max = float(triangle[:, 1].max())
    x0 = max(0, int(math.ceil(u_min * size - 0.5)))
    x1 = min(size - 1, int(math.floor(u_max * size - 0.5)))
    y0 = max(0, int(math.ceil((1.0 - v_max) * size - 0.5)))
    y1 = min(size - 1, int(math.floor((1.0 - v_min) * size - 0.5)))
    return None if x0 > x1 or y0 > y1 else (x0, x1, y0, y1)


def rasterize_object(
    *,
    source_bgr: np.ndarray,
    destination_bgr: np.ndarray,
    object_owner: np.ndarray,
    chart_owner: np.ndarray,
    old_uv: np.ndarray,
    new_uv: np.ndarray,
    global_chart: np.ndarray,
    object_id: int,
) -> dict[str, Any]:
    """Raster one object's triangles using destination pixel-centre samples."""

    size = int(destination_bgr.shape[0])
    require(destination_bgr.shape == (size, size, 3), "Destination atlas must be square BGR")
    require(old_uv.shape == new_uv.shape == (len(global_chart), 3, 2), "Raster face arrays disagree")
    old_area2 = triangle_double_area(old_uv)
    new_area2 = triangle_double_area(new_uv)
    require(bool(np.all(np.abs(old_area2) > 1.0e-14)), f"Object {object_id} contains old UV degenerates")
    require(bool(np.all(np.abs(new_area2) > 1.0e-14)), f"Object {object_id} contains final UV degenerates")

    counters = {
        "faces": int(len(new_uv)),
        "faces_with_new_texels": 0,
        "faces_without_pixel_centres": 0,
        "new_surface_texels": 0,
        "same_domain_shared_edge_samples": 0,
        "cross_domain_raster_collisions": 0,
        "bilinear_exact_black_before_fallback": 0,
        "nearest_source_fallback_texels": 0,
        "source_confirmed_black_texels": 0,
        "unexplained_new_black_texels": 0,
    }
    object_value = int(object_id) + 1
    require(0 < object_value <= np.iinfo(np.uint16).max, "Object owner exceeds uint16")
    for face_index, (old_triangle, new_triangle) in enumerate(zip(old_uv, new_uv)):
        bounds = _pixel_bounds(new_triangle, size)
        if bounds is None:
            counters["faces_without_pixel_centres"] += 1
            continue
        x0, x1, y0, y1 = bounds
        a, b, c = new_triangle.astype(np.float64)
        edge_b = b - a
        edge_c = c - a
        denominator = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]
        wrote_face = False
        face_candidate_centres = 0
        chart_value = int(global_chart[face_index]) + 1
        require(0 < chart_value <= np.iinfo(np.uint16).max, "Chart owner exceeds uint16")
        x_pixels = np.arange(x0, x1 + 1, dtype=np.int32)
        u_values = (x_pixels.astype(np.float64) + 0.5) / float(size)
        for y in range(y0, y1 + 1):
            v_value = 1.0 - (float(y) + 0.5) / float(size)
            relative_x = u_values - a[0]
            relative_y = v_value - a[1]
            weight_b = (relative_x * edge_c[1] - relative_y * edge_c[0]) / denominator
            weight_c = (edge_b[0] * relative_y - edge_b[1] * relative_x) / denominator
            weight_a = 1.0 - weight_b - weight_c
            inside = (
                (weight_a >= -1.0e-10)
                & (weight_b >= -1.0e-10)
                & (weight_c >= -1.0e-10)
            )
            if not bool(np.any(inside)):
                continue
            xs = x_pixels[inside]
            wa = weight_a[inside]
            wb = weight_b[inside]
            wc = weight_c[inside]
            face_candidate_centres += int(len(xs))
            current_object = object_owner[y, xs]
            current_chart = chart_owner[y, xs]
            empty = current_object == 0
            same = (~empty) & (current_object == object_value) & (current_chart == chart_value)
            cross = (~empty) & ~same
            counters["same_domain_shared_edge_samples"] += int(same.sum())
            counters["cross_domain_raster_collisions"] += int(cross.sum())
            if not bool(np.any(empty)):
                continue
            write_x = xs[empty]
            source_uv = (
                wa[empty, None] * old_triangle[0]
                + wb[empty, None] * old_triangle[1]
                + wc[empty, None] * old_triangle[2]
            )
            sampled, sample_stats = bilinear_sample_bgr(source_bgr, source_uv)
            destination_bgr[y, write_x] = sampled
            object_owner[y, write_x] = object_value
            chart_owner[y, write_x] = chart_value
            wrote_face = True
            counters["new_surface_texels"] += int(len(write_x))
            for key in (
                "bilinear_exact_black_before_fallback",
                "nearest_source_fallback_texels",
                "source_confirmed_black_texels",
                "unexplained_new_black_texels",
            ):
                counters[key] += int(sample_stats[key])
        if wrote_face:
            counters["faces_with_new_texels"] += 1
        elif face_candidate_centres == 0:
            counters["faces_without_pixel_centres"] += 1
    return counters


def _closest_triangle_barycentric(
    point: np.ndarray,
    triangle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return barycentrics and the closest point on a non-degenerate 2D triangle.

    Raster chart completion can need a pixel whose centre is just outside a
    sub-pixel sliver.  Projecting that centre onto the destination triangle
    gives a valid barycentric source lookup instead of sampling an unrelated
    centroid or extrapolating beyond the source face.
    """

    point64 = np.asarray(point, dtype=np.float64)
    triangle64 = np.asarray(triangle, dtype=np.float64)
    require(point64.shape == (2,) and triangle64.shape == (3, 2), "Bad closest-point input")
    a, b, c = triangle64
    edge_b = b - a
    edge_c = c - a
    denominator = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]
    require(abs(float(denominator)) > 1.0e-14, "Cannot project onto a degenerate triangle")
    relative = point64 - a
    weight_b = (relative[0] * edge_c[1] - relative[1] * edge_c[0]) / denominator
    weight_c = (edge_b[0] * relative[1] - edge_b[1] * relative[0]) / denominator
    weight_a = 1.0 - weight_b - weight_c
    unconstrained = np.asarray([weight_a, weight_b, weight_c], dtype=np.float64)
    if bool(np.all(unconstrained >= -1.0e-12)):
        barycentric = np.maximum(unconstrained, 0.0)
        barycentric /= barycentric.sum()
        closest = barycentric @ triangle64
        return barycentric, closest, float(np.sum((point64 - closest) ** 2))

    # A convex triangle's closest boundary point lies on one of its edges.
    # Edge order is fixed, which also fixes exact-distance tie resolution.
    best: tuple[tuple[float, int], np.ndarray, np.ndarray] | None = None
    for edge_order, (start_index, end_index) in enumerate(((0, 1), (1, 2), (2, 0))):
        start = triangle64[start_index]
        end = triangle64[end_index]
        edge = end - start
        length_squared = float(np.dot(edge, edge))
        require(length_squared > 0.0, "Triangle contains a zero-length edge")
        t = float(np.clip(np.dot(point64 - start, edge) / length_squared, 0.0, 1.0))
        closest = start + t * edge
        distance_squared = float(np.sum((point64 - closest) ** 2))
        barycentric = np.zeros(3, dtype=np.float64)
        barycentric[start_index] = 1.0 - t
        barycentric[end_index] = t
        candidate = ((distance_squared, edge_order), barycentric, closest)
        if best is None or candidate[0] < best[0]:
            best = candidate
    require(best is not None, "Closest-point edge search failed")
    return best[1], best[2], best[0][0]


def complete_missing_chart_texels(
    *,
    source_bgr: np.ndarray,
    destination_bgr: np.ndarray,
    object_owner: np.ndarray,
    chart_owner: np.ndarray,
    seed_sources: list[dict[str, Any]],
    expected_chart_count: int,
    search_radius: int = PADDING_PIXELS,
) -> dict[str, Any]:
    """Seed exactly one free texel for every chart missed by centre sampling.

    Charts are processed by ascending page-global ID.  Candidate pixels are
    scored by distance from their centre to a destination face, then distance
    from that face's centroid, source face index and pixel coordinates.  Only
    an owner-free pixel can win.  The selected pixel centre is projected to
    the face and mapped through barycentric correspondence to the old UV face.
    No existing owner is ever overwritten.
    """

    size = int(destination_bgr.shape[0])
    require(destination_bgr.shape == (size, size, 3), "Chart completion destination must be square")
    require(object_owner.shape == chart_owner.shape == (size, size), "Chart completion owner shape drift")
    require(object_owner.dtype == chart_owner.dtype == np.uint16, "Chart completion owners must be uint16")
    require(np.array_equal(object_owner > 0, chart_owner > 0), "Owner supports differ before chart completion")
    require(0 < expected_chart_count < np.iinfo(np.uint16).max, "Expected chart count leaves uint16 range")
    require(search_radius >= 0, "Negative chart-completion search radius")

    expected = np.arange(1, expected_chart_count + 1, dtype=np.uint16)
    present_before = np.unique(chart_owner[chart_owner > 0])
    unexpected_before = np.setdiff1d(present_before, expected, assume_unique=True)
    require(len(unexpected_before) == 0, "Unexpected chart owner ID before completion")
    missing = np.setdiff1d(expected, present_before, assume_unique=True)

    source_by_chart: dict[int, dict[str, Any]] = {}
    for source in seed_sources:
        global_values = np.asarray(source["global_chart"], dtype=np.int64) + 1
        relevant = np.intersect1d(global_values, missing, assume_unique=False)
        for chart_value_raw in relevant.tolist():
            chart_value = int(chart_value_raw)
            require(chart_value not in source_by_chart, f"Chart {chart_value} spans multiple objects")
            face_indices = np.flatnonzero(global_values == chart_value).astype(np.int32)
            require(len(face_indices) > 0, f"Chart {chart_value} has no source faces")
            source_by_chart[chart_value] = {
                "object_id": int(source["object_id"]),
                "face_indices": face_indices,
                "old_uv": np.asarray(source["old_uv"])[face_indices],
                "new_uv": np.asarray(source["new_uv"])[face_indices],
            }
    require(set(source_by_chart) == set(int(value) for value in missing.tolist()), "Missing chart source lookup is incomplete")

    records: list[dict[str, Any]] = []
    occupied_candidate_rejections = 0
    sample_totals = {
        "bilinear_exact_black_before_fallback": 0,
        "nearest_source_fallback_texels": 0,
        "source_confirmed_black_texels": 0,
        "unexplained_new_black_texels": 0,
    }
    for chart_value_raw in missing.tolist():
        chart_value = int(chart_value_raw)
        domain = source_by_chart[chart_value]
        best: tuple[tuple[float, float, int, int, int], dict[str, Any]] | None = None
        for local_index, (old_triangle_raw, new_triangle_raw) in enumerate(
            zip(domain["old_uv"], domain["new_uv"])
        ):
            old_triangle = np.asarray(old_triangle_raw, dtype=np.float64)
            new_triangle = np.asarray(new_triangle_raw, dtype=np.float64)
            centroid = new_triangle.mean(axis=0)
            centroid_x = float(centroid[0] * size - 0.5)
            centroid_y = float((1.0 - centroid[1]) * size - 0.5)
            base_x = int(math.floor(centroid_x + 0.5))
            base_y = int(math.floor(centroid_y + 0.5))
            source_face_index = int(domain["face_indices"][local_index])
            for y in range(base_y - search_radius, base_y + search_radius + 1):
                if y < 0 or y >= size:
                    continue
                for x in range(base_x - search_radius, base_x + search_radius + 1):
                    if x < 0 or x >= size:
                        continue
                    if object_owner[y, x] != 0 or chart_owner[y, x] != 0:
                        occupied_candidate_rejections += 1
                        continue
                    pixel_uv = np.asarray(
                        [(float(x) + 0.5) / size, 1.0 - (float(y) + 0.5) / size],
                        dtype=np.float64,
                    )
                    barycentric, closest_uv, distance_squared_uv = _closest_triangle_barycentric(
                        pixel_uv, new_triangle
                    )
                    distance_squared_pixels = float(distance_squared_uv * size * size)
                    centroid_distance_squared = float(
                        (float(x) - centroid_x) ** 2 + (float(y) - centroid_y) ** 2
                    )
                    score = (
                        distance_squared_pixels,
                        centroid_distance_squared,
                        source_face_index,
                        y,
                        x,
                    )
                    if best is None or score < best[0]:
                        best = (
                            score,
                            {
                                "x": x,
                                "y": y,
                                "source_face_index": source_face_index,
                                "old_triangle": old_triangle,
                                "barycentric": barycentric,
                                "closest_uv": closest_uv,
                                "pixel_uv": pixel_uv,
                                "distance_pixels": math.sqrt(max(distance_squared_pixels, 0.0)),
                                "centroid_distance_pixels": math.sqrt(max(centroid_distance_squared, 0.0)),
                            },
                        )
        require(best is not None, f"No free conservative seed pixel found for chart {chart_value}")
        selected = best[1]
        x = int(selected["x"])
        y = int(selected["y"])
        require(object_owner[y, x] == 0 and chart_owner[y, x] == 0, f"Chart {chart_value} seed collided")
        source_uv = np.asarray(selected["barycentric"]) @ np.asarray(selected["old_triangle"])
        sampled, sample_stats = bilinear_sample_bgr(source_bgr, source_uv.reshape(1, 2))
        destination_bgr[y, x] = sampled[0]
        object_owner[y, x] = int(domain["object_id"]) + 1
        chart_owner[y, x] = chart_value
        for key in sample_totals:
            sample_totals[key] += int(sample_stats[key])
        records.append(
            {
                "chart_owner_id": chart_value,
                "source_object_id": int(domain["object_id"]),
                "source_face_index": int(selected["source_face_index"]),
                "destination_pixel_xy": [x, y],
                "destination_pixel_uv": [float(value) for value in selected["pixel_uv"]],
                "closest_destination_face_uv": [float(value) for value in selected["closest_uv"]],
                "source_uv": [float(value) for value in source_uv],
                "barycentric": [float(value) for value in selected["barycentric"]],
                "distance_to_face_pixels": float(selected["distance_pixels"]),
                "distance_to_face_centroid_pixels": float(selected["centroid_distance_pixels"]),
                "owner_pixel_was_free": True,
            }
        )

    present_after = np.unique(chart_owner[chart_owner > 0])
    ids_exact = bool(np.array_equal(present_after, expected))
    support_exact = bool(np.array_equal(object_owner > 0, chart_owner > 0))
    require(ids_exact, "Chart completion did not produce every expected chart owner ID")
    require(support_exact, "Owner supports differ after chart completion")
    return {
        "policy": (
            "ascending_missing_chart_id;free_pixel_only;minimum_pixel_center_to_face_distance;"
            "closest_point_barycentric_source_mapping"
        ),
        "expected_chart_ids": {
            "first": 1,
            "last": int(expected_chart_count),
            "count": int(expected_chart_count),
        },
        "present_before": int(len(present_before)),
        "missing_before": [int(value) for value in missing.tolist()],
        "seeded_texels": int(len(records)),
        "present_after": int(len(present_after)),
        "occupied_candidate_rejections": int(occupied_candidate_rejections),
        "maximum_seed_distance_to_face_pixels": float(
            max((record["distance_to_face_pixels"] for record in records), default=0.0)
        ),
        "no_owner_overwrites": True,
        "all_expected_chart_ids_present": ids_exact,
        "object_chart_support_exact": support_exact,
        "sampling": sample_totals,
        "seeds": records,
    }


def grow_owned_gutters_in_place(
    surface_bgr: np.ndarray,
    object_owner: np.ndarray,
    chart_owner: np.ndarray,
    *,
    padding: int = PADDING_PIXELS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow exact donor values into a finite 8-neighbour gutter band.

    ``DIST_LABEL_PIXEL`` assigns a stable label to each occupied zero pixel.
    A gutter texel copies the RGB, object and chart of exactly one donor; no
    arithmetic combines samples from two domains.
    """

    require(surface_bgr.dtype == np.uint8 and surface_bgr.ndim == 3, "Gutter RGB must be uint8 HxWx3")
    require(object_owner.dtype == chart_owner.dtype == np.uint16, "Gutter owners must be uint16")
    require(object_owner.shape == chart_owner.shape == surface_bgr.shape[:2], "Gutter array shape drift")
    occupied = object_owner > 0
    require(np.array_equal(occupied, chart_owner > 0), "Object/chart occupancy differs")
    occupied_count = int(occupied.sum())
    require(occupied_count > 0, "Cannot grow gutters from empty occupancy")
    source = np.where(occupied, 0, 255).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_C,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    del source
    seed_labels = labels[occupied]
    require(bool(np.all(seed_labels > 0)), "Occupied donors have missing distance labels")
    label_slots = int(labels.max()) + 1
    donor_bgr = np.zeros((label_slots, 3), dtype=np.uint8)
    donor_object = np.zeros(label_slots, dtype=np.uint16)
    donor_chart = np.zeros(label_slots, dtype=np.uint16)
    donor_bgr[seed_labels] = surface_bgr[occupied]
    donor_object[seed_labels] = object_owner[occupied]
    donor_chart[seed_labels] = chart_owner[occupied]
    del seed_labels

    padded_mask = np.zeros(object_owner.shape, dtype=np.uint8)
    gutter_texels = 0
    height = object_owner.shape[0]
    for y0 in range(0, height, ROW_CHUNK):
        y1 = min(height, y0 + ROW_CHUNK)
        local_occupied = occupied[y0:y1]
        local_band = (labels[y0:y1] > 0) & (distance[y0:y1] <= float(padding))
        padded_mask[y0:y1][local_band] = 255
        gutter = local_band & ~local_occupied
        if not bool(np.any(gutter)):
            continue
        selected = labels[y0:y1][gutter]
        surface_bgr[y0:y1][gutter] = donor_bgr[selected]
        object_owner[y0:y1][gutter] = donor_object[selected]
        chart_owner[y0:y1][gutter] = donor_chart[selected]
        gutter_texels += int(gutter.sum())
    del distance, labels, donor_bgr, donor_object, donor_chart
    gc.collect()
    padded = padded_mask > 0
    require(np.array_equal(object_owner[padded] > 0, chart_owner[padded] > 0), "Padded owner domains differ")
    require(not bool(np.any((object_owner > 0) & ~padded)), "Owner escaped the finite gutter band")
    return padded_mask, {
        "algorithm": "opencv_DIST_C_labelled_nearest_pixel_donor",
        "padding_pixels": int(padding),
        "occupied_texels": occupied_count,
        "gutter_texels": int(gutter_texels),
        "padded_texels": int(np.count_nonzero(padded_mask)),
        "cross_domain_averaging": False,
        "rgb_owner_chart_copied_from_same_single_donor": True,
        "tie_resolution": "deterministic_opencv_pixel_label_order",
    }


def write_png(path: Path, image: np.ndarray, *, semantic: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(image)
    require(
        bool(
            cv2.imwrite(
                str(path),
                contiguous,
                [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
            )
        ),
        f"PNG write failed: {path}",
    )
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(decoded is not None, f"PNG reopen failed: {path}")
    require(decoded.shape == contiguous.shape and decoded.dtype == contiguous.dtype, f"PNG mode drift: {path}")
    expected_pixels = sha256_array(contiguous)
    actual_pixels = sha256_array(decoded)
    require(expected_pixels == actual_pixels, f"PNG decoded pixels drift: {path}")
    return {
        "path": path.as_posix(),
        "semantic": semantic,
        "file_sha256": sha256_file(path),
        "decoded_pixel_sha256": actual_pixels,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "bytes": int(path.stat().st_size),
    }


def write_hash_inventory(root: Path) -> dict[str, Any]:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "hashes.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    inventory = root / "hashes.sha256"
    inventory.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": inventory.relative_to(root).as_posix(),
        "sha256": sha256_file(inventory),
        "entries": len(lines),
    }


def build_reprojection(
    *,
    placement_report_path: Path,
    source_atlas_path: Path,
    output_dir: Path,
    preview_size: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    placement_report_path = Path(placement_report_path).expanduser().resolve()
    source_atlas_path = Path(source_atlas_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    report, layout_rows = validate_placement_report(placement_report_path)
    page_id = str(report["page_id"])
    native_resolution = int(report["resolution"][0])
    output_size = native_resolution if preview_size is None else int(preview_size)
    require(256 <= output_size <= native_resolution, "Preview/output size leaves supported range")
    require(source_atlas_path.is_file(), f"Missing source atlas: {source_atlas_path}")
    source_bgr = cv2.imread(str(source_atlas_path), cv2.IMREAD_COLOR)
    require(source_bgr is not None, f"Source atlas decode failed: {source_atlas_path}")
    require(
        source_bgr.shape == (native_resolution, native_resolution, 3)
        and source_bgr.dtype == np.uint8,
        "Source atlas must be native-resolution uint8 RGB PNG",
    )
    require(not output_dir.exists(), f"Refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging.",
            dir=str(output_dir.parent),
        )
    )
    try:
        destination_bgr = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        object_owner = np.zeros((output_size, output_size), dtype=np.uint16)
        chart_owner = np.zeros((output_size, output_size), dtype=np.uint16)
        chart_offset = 0
        object_reports = []
        source_inputs = []
        seed_sources = []
        total_analytic_area = 0.0
        total_target_faces = 0
        aggregate = {
            "faces": 0,
            "faces_with_new_texels": 0,
            "faces_without_pixel_centres": 0,
            "new_surface_texels": 0,
            "same_domain_shared_edge_samples": 0,
            "cross_domain_raster_collisions": 0,
            "bilinear_exact_black_before_fallback": 0,
            "nearest_source_fallback_texels": 0,
            "source_confirmed_black_texels": 0,
            "unexplained_new_black_texels": 0,
        }
        for row in layout_rows:
            record, chart_offset = load_layout_object(
                row,
                report_dir=placement_report_path.parent,
                page_id=page_id,
                native_resolution=native_resolution,
                chart_offset=chart_offset,
            )
            raster = rasterize_object(
                source_bgr=source_bgr,
                destination_bgr=destination_bgr,
                object_owner=object_owner,
                chart_owner=chart_owner,
                old_uv=record["old_uv"],
                new_uv=record["new_uv"],
                global_chart=record["global_chart"],
                object_id=record["object_id"],
            )
            for key in aggregate:
                aggregate[key] += int(raster[key])
            total_analytic_area += float(record["analytic_uv_area_fraction"])
            total_target_faces += int(record["target_mask"].sum())
            object_reports.append(
                {
                    "object_id": record["object_id"],
                    "source_kind": record["source_kind"],
                    "faces": int(len(record["faces"])),
                    "targeted_faces": int(record["target_mask"].sum()),
                    "charts": record["local_chart_count"],
                    "analytic_uv_area_fraction": record["analytic_uv_area_fraction"],
                    "layout_sidecar": str(record["layout_path"]),
                    "layout_sidecar_sha256": record["layout_sha256"],
                    "source_geometry": str(record["source_path"]),
                    "source_geometry_sha256": record["source_sha256"],
                    "raster": raster,
                }
            )
            source_inputs.append(
                {
                    "layout": str(record["layout_path"]),
                    "layout_sha256": record["layout_sha256"],
                    "source": str(record["source_path"]),
                    "source_sha256": record["source_sha256"],
                }
            )
            # Retain only the correspondence arrays needed for the post-raster
            # conservative chart completion pass.  This is much smaller than
            # retaining source geometry payloads or per-pixel candidates.
            seed_sources.append(
                {
                    "object_id": record["object_id"],
                    "old_uv": record["old_uv"],
                    "new_uv": record["new_uv"],
                    "global_chart": record["global_chart"],
                }
            )
            del record

        require(aggregate["cross_domain_raster_collisions"] == 0, "Cross-domain raster collision detected")
        require(aggregate["unexplained_new_black_texels"] == 0, "Unexplained new black texels detected")
        chart_completion = complete_missing_chart_texels(
            source_bgr=source_bgr,
            destination_bgr=destination_bgr,
            object_owner=object_owner,
            chart_owner=chart_owner,
            seed_sources=seed_sources,
            expected_chart_count=int(chart_offset),
        )
        for key in (
            "bilinear_exact_black_before_fallback",
            "nearest_source_fallback_texels",
            "source_confirmed_black_texels",
            "unexplained_new_black_texels",
        ):
            aggregate[key] += int(chart_completion["sampling"][key])
        del seed_sources
        occupied = object_owner > 0
        owner_support_exact = bool(np.array_equal(occupied, chart_owner > 0))
        require(owner_support_exact, "Object/chart surface owners differ")
        expected_chart_owner_ids = np.arange(1, int(chart_offset) + 1, dtype=np.uint16)
        observed_chart_owner_ids = np.unique(chart_owner[chart_owner > 0])
        all_expected_chart_ids_present = bool(
            np.array_equal(observed_chart_owner_ids, expected_chart_owner_ids)
        )
        require(all_expected_chart_ids_present, "One or more page-global charts have no owned surface texel")
        surface_texels = int(occupied.sum())
        require(surface_texels > 0, "Page raster produced no occupied texels")
        expected_area_texels = float(total_analytic_area * output_size * output_size)

        diagnostic_dir = stage / "diagnostics" / page_id
        texture_name = (
            f"{page_id}_reprojected_v9.png"
            if preview_size is None
            else f"{page_id}_reprojected_v9_preview_{output_size}.png"
        )
        artifacts = {}
        artifacts["occupancy_mask"] = write_png(
            diagnostic_dir / "occupancy_mask.png",
            occupied.astype(np.uint8) * 255,
            semantic="occupied_destination_pixel_centres",
        )
        artifacts["object_owner"] = write_png(
            diagnostic_dir / "object_owner_id.png",
            object_owner,
            semantic="occupied_only_source_object_id_plus_one",
        )
        artifacts["chart_owner"] = write_png(
            diagnostic_dir / "chart_owner_id.png",
            chart_owner,
            semantic="occupied_only_page_global_chart_id_plus_one",
        )

        # The source atlas is no longer needed after all surface samples.  Drop
        # it before the distance transform allocates its two full-page buffers.
        del source_bgr
        gc.collect()
        padded_mask, gutter_report = grow_owned_gutters_in_place(
            destination_bgr,
            object_owner,
            chart_owner,
            padding=PADDING_PIXELS,
        )
        artifacts["texture"] = write_png(
            stage / "textures" / texture_name,
            destination_bgr,
            semantic="standard_png_rgb_encoded_from_cv2_bgr_working_array",
        )
        artifacts["padded_mask"] = write_png(
            diagnostic_dir / "padded_mask.png",
            padded_mask,
            semantic="occupied_plus_16px_deterministic_gutter",
        )
        artifacts["gutter_object_owner"] = write_png(
            diagnostic_dir / "gutter_object_owner_id.png",
            object_owner,
            semantic="padded_single_donor_source_object_id_plus_one",
        )
        artifacts["gutter_chart_owner"] = write_png(
            diagnostic_dir / "gutter_chart_owner_id.png",
            chart_owner,
            semantic="padded_single_donor_page_global_chart_id_plus_one",
        )

        # Store paths relative to the final output, never to the temporary
        # staging directory used for the atomic build.
        for artifact in artifacts.values():
            artifact["path"] = Path(artifact["path"]).relative_to(stage).as_posix()
        report_payload = {
            "schema": SCHEMA,
            "status": "passed",
            "mode": "preview" if preview_size is not None else "native",
            "page_id": page_id,
            "native_resolution": [native_resolution, native_resolution],
            "output_resolution": [output_size, output_size],
            "preview_size": int(preview_size) if preview_size is not None else None,
            "inputs": {
                "placement_report": str(placement_report_path),
                "placement_report_sha256": sha256_file(placement_report_path),
                "source_atlas": str(source_atlas_path),
                "source_atlas_sha256": sha256_file(source_atlas_path),
                "source_atlas_shape": [native_resolution, native_resolution, 3],
                "layout_objects": source_inputs,
            },
            "routing": {
                "triangles": aggregate["faces"],
                "targeted_triangles": total_target_faces,
                "objects": len(object_reports),
                "page_global_charts": int(chart_offset),
                "destination_pixel_sampling": "one_pixel_center_sample_no_antialias",
                "destination_uv": "u=(x+.5)/W;v=1-(y+.5)/H",
                "source_uv_mapping": "face_correspondence_then_barycentric_old_triangle",
                "source_sampling": "bilinear_uint8_srgb_border_clamp",
                "source_pixel_mapping": "x=u*W-.5;y=(1-v)*H-.5",
                "face_order": "placement_report_object_order_then_source_face_index",
                "subpixel_chart_completion": (
                    "one_free_texel_per_missing_chart_from_closest_destination_face_point_"
                    "then_barycentric_old_uv"
                ),
            },
            "coverage": {
                "surface_texels": surface_texels,
                "surface_fraction": surface_texels / float(output_size * output_size),
                "analytic_uv_area_fraction": total_analytic_area,
                "analytic_expected_texels": expected_area_texels,
                "raster_to_analytic_area_ratio": surface_texels / max(expected_area_texels, 1.0),
                "faces_with_new_texels": aggregate["faces_with_new_texels"],
                "faces_without_pixel_centres": aggregate["faces_without_pixel_centres"],
                "same_domain_shared_edge_samples": aggregate["same_domain_shared_edge_samples"],
                "cross_domain_raster_collisions": aggregate["cross_domain_raster_collisions"],
                "conservative_chart_seed_texels": chart_completion["seeded_texels"],
            },
            "black_fallback": {
                "policy": (
                    "if_bilinear_rounds_to_exact_black_and_nearest_source_is_nonblack_"
                    "copy_exact_nearest_source;retain_source_confirmed_black"
                ),
                "threshold_uint8": BLACK_THRESHOLD,
                "bilinear_exact_black_before_fallback": aggregate[
                    "bilinear_exact_black_before_fallback"
                ],
                "nearest_source_fallback_texels": aggregate[
                    "nearest_source_fallback_texels"
                ],
                "source_confirmed_black_texels": aggregate[
                    "source_confirmed_black_texels"
                ],
                "unexplained_new_black_texels": aggregate[
                    "unexplained_new_black_texels"
                ],
            },
            "chart_completion": chart_completion,
            "gutter": gutter_report,
            "objects": object_reports,
            "artifacts": artifacts,
            "gates": {
                "placement_report_passed": True,
                "all_layout_and_source_hashes_verified": True,
                "surface_nonempty": surface_texels > 0,
                "object_chart_occupancy_exact": bool(
                    owner_support_exact and all_expected_chart_ids_present
                ),
                "all_expected_chart_ids_have_surface_texels": all_expected_chart_ids_present,
                "chart_completion_never_overwrote_an_owner": bool(
                    chart_completion["no_owner_overwrites"]
                ),
                "zero_cross_domain_raster_collisions": aggregate[
                    "cross_domain_raster_collisions"
                ]
                == 0,
                "zero_unexplained_new_black_texels": aggregate[
                    "unexplained_new_black_texels"
                ]
                == 0,
                "deterministic_16px_single_donor_gutters": True,
            },
            "provenance": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "blender_run": False,
                "fal_called": False,
                "github_called": False,
                "texture_bake_performed": False,
                "canonical_inputs_modified": False,
                "atomic_output_directory": True,
            },
            "runtime_seconds": float(time.perf_counter() - started),
        }
        require(all(report_payload["gates"].values()), "A reprojection gate failed")
        report_path = stage / "reproject_page_texture_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        inventory = write_hash_inventory(stage)
        os.replace(stage, output_dir)
        return {
            "status": "passed",
            "output": str(output_dir),
            "report": str(output_dir / report_path.relative_to(stage)),
            "report_sha256": sha256_file(output_dir / report_path.relative_to(stage)),
            "hash_inventory": {
                "path": str(output_dir / inventory["path"]),
                "sha256": inventory["sha256"],
                "entries": inventory["entries"],
            },
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def run_self_test() -> dict[str, Any]:
    """Small deterministic test for orientation, barycentrics and gutters."""

    # PNG/top-row orientation and exact bilinear centre.
    source = np.asarray(
        [
            [[0, 0, 255], [0, 255, 0]],      # top: red, green in BGR storage
            [[255, 0, 0], [255, 255, 255]],  # bottom: blue, white
        ],
        dtype=np.uint8,
    )
    sampled, stats = bilinear_sample_bgr(
        source,
        np.asarray([[0.25, 0.75], [0.5, 0.5]], dtype=np.float64),
    )
    require(np.array_equal(sampled[0], source[0, 0]), "Self-test UV/PNG orientation failed")
    require(np.array_equal(sampled[1], np.asarray([128, 128, 128], dtype=np.uint8)), "Self-test bilinear centre failed")
    require(stats["unexplained_new_black_texels"] == 0, "Self-test black fallback failed")

    constant = np.zeros((16, 16, 3), dtype=np.uint8)
    constant[:] = [23, 117, 211]
    old_uv = np.asarray(
        [
            [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8]],
            [[0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        ],
        dtype=np.float32,
    )
    new_uv = np.asarray(
        [
            [[0.1, 0.15], [0.65, 0.15], [0.1, 0.7]],
            [[0.65, 0.15], [0.65, 0.7], [0.1, 0.7]],
        ],
        dtype=np.float32,
    )

    def raster_once() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        color = np.zeros((32, 32, 3), dtype=np.uint8)
        owner = np.zeros((32, 32), dtype=np.uint16)
        chart = np.zeros((32, 32), dtype=np.uint16)
        report = rasterize_object(
            source_bgr=constant,
            destination_bgr=color,
            object_owner=owner,
            chart_owner=chart,
            old_uv=old_uv,
            new_uv=new_uv,
            global_chart=np.asarray([0, 0], dtype=np.int32),
            object_id=4,
        )
        return color, owner, chart, report

    color_a, owner_a, chart_a, raster_a = raster_once()
    color_b, owner_b, chart_b, raster_b = raster_once()
    require(np.array_equal(color_a, color_b), "Self-test raster RGB is not deterministic")
    require(np.array_equal(owner_a, owner_b), "Self-test raster owner is not deterministic")
    require(np.array_equal(chart_a, chart_b), "Self-test raster chart is not deterministic")
    occupied = owner_a > 0
    require(bool(np.any(occupied)), "Self-test raster is empty")
    require(np.all(color_a[occupied] == [23, 117, 211]), "Self-test barycentric reprojection changed constant RGB")
    require(raster_a == raster_b and raster_a["cross_domain_raster_collisions"] == 0, "Self-test raster report drift")

    # A sub-pixel sliver with no covered pixel centre must receive exactly one
    # deterministic, collision-free conservative seed.  An existing chart-1
    # texel beside it proves the completion pass never overwrites owners.
    tiny_old = np.asarray(
        [[[0.2, 0.2], [0.3, 0.2], [0.2, 0.3]]],
        dtype=np.float32,
    )
    tiny_new = np.asarray(
        [[[0.4990, 0.4990], [0.5010, 0.4990], [0.4990, 0.5010]]],
        dtype=np.float32,
    )

    def complete_tiny_once() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        color = np.zeros((32, 32, 3), dtype=np.uint8)
        owner = np.zeros((32, 32), dtype=np.uint16)
        chart = np.zeros((32, 32), dtype=np.uint16)
        color[16, 16] = [91, 92, 93]
        owner[16, 16] = 2
        chart[16, 16] = 1
        raster = rasterize_object(
            source_bgr=constant,
            destination_bgr=color,
            object_owner=owner,
            chart_owner=chart,
            old_uv=tiny_old,
            new_uv=tiny_new,
            global_chart=np.asarray([1], dtype=np.int32),
            object_id=7,
        )
        require(raster["new_surface_texels"] == 0, "Self-test sliver unexpectedly covered a pixel centre")
        completion = complete_missing_chart_texels(
            source_bgr=constant,
            destination_bgr=color,
            object_owner=owner,
            chart_owner=chart,
            seed_sources=[
                {
                    "object_id": 7,
                    "old_uv": tiny_old,
                    "new_uv": tiny_new,
                    "global_chart": np.asarray([1], dtype=np.int32),
                }
            ],
            expected_chart_count=2,
        )
        return color, owner, chart, completion

    tiny_color_a, tiny_owner_a, tiny_chart_a, tiny_completion_a = complete_tiny_once()
    tiny_color_b, tiny_owner_b, tiny_chart_b, tiny_completion_b = complete_tiny_once()
    require(tiny_completion_a == tiny_completion_b, "Self-test conservative chart completion is not deterministic")
    require(np.array_equal(tiny_color_a, tiny_color_b), "Self-test conservative seed RGB drift")
    require(np.array_equal(tiny_owner_a, tiny_owner_b), "Self-test conservative seed owner drift")
    require(np.array_equal(tiny_chart_a, tiny_chart_b), "Self-test conservative seed chart drift")
    require(tiny_completion_a["seeded_texels"] == 1, "Self-test conservative seed count drift")
    require(np.array_equal(np.unique(tiny_chart_a[tiny_chart_a > 0]), [1, 2]), "Self-test chart ID completion failed")
    require(np.array_equal(tiny_color_a[16, 16], [91, 92, 93]), "Self-test completion overwrote an owner")
    require(int(tiny_owner_a[tiny_chart_a == 2][0]) == 8, "Self-test conservative object owner drift")
    require(np.array_equal(tiny_color_a[tiny_chart_a == 2][0], [23, 117, 211]), "Self-test conservative source mapping drift")

    # Two isolated domains prove finite padding copies a single donor value and
    # never produces an averaged purple texel.
    gutter_color = np.zeros((17, 17, 3), dtype=np.uint8)
    gutter_object = np.zeros((17, 17), dtype=np.uint16)
    gutter_chart = np.zeros((17, 17), dtype=np.uint16)
    gutter_color[4, 4] = [0, 0, 255]
    gutter_object[4, 4] = 2
    gutter_chart[4, 4] = 3
    gutter_color[12, 12] = [255, 0, 0]
    gutter_object[12, 12] = 8
    gutter_chart[12, 12] = 9
    original = gutter_color.copy()
    padded, gutter_stats = grow_owned_gutters_in_place(
        gutter_color, gutter_object, gutter_chart, padding=2
    )
    values = {tuple(value) for value in gutter_color[padded > 0].tolist()}
    require(values == {(0, 0, 255), (255, 0, 0)}, "Self-test gutter mixed domain RGB")
    require(np.array_equal(gutter_color[4, 4], original[4, 4]), "Self-test gutter changed source RGB")
    require(gutter_stats["cross_domain_averaging"] is False, "Self-test gutter policy drift")

    # Exercise the complete file contract, atomic directory handoff, PNG
    # round-trips, report and hash inventory on a tiny synthetic page.
    with tempfile.TemporaryDirectory(prefix="v9_reproject_selftest.") as temporary:
        temporary_root = Path(temporary)
        source_npz = temporary_root / "source_proxy.npz"
        faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        np.savez_compressed(
            source_npz,
            proxy_faces=faces,
            proxy_corner_uv=old_uv,
        )
        layout_npz = temporary_root / "object_04.page_uv_v9.npz"
        np.savez_compressed(
            layout_npz,
            schema=np.asarray(LAYOUT_SCHEMA),
            source_object_id=np.asarray(4, dtype=np.int16),
            target_page_id=np.asarray("synthetic_page"),
            page_resolution=np.asarray(256, dtype=np.int32),
            source_path=np.asarray(str(source_npz)),
            source_sha256=np.asarray(sha256_file(source_npz)),
            source_faces_sha256=np.asarray(sha256_array(faces)),
            baseline_corner_uv_sha256=np.asarray(sha256_array(old_uv)),
            final_proxy_corner_uv=new_uv,
            final_proxy_chart_id=np.asarray([0, 0], dtype=np.int32),
            targeted_face_mask=np.asarray([1, 1], dtype=np.uint8),
        )
        placement = {
            "schema": EXPECTED_PLACEMENT_SCHEMA,
            "status": "passed",
            "mode": "page_repack",
            "page_id": "synthetic_page",
            "resolution": [256, 256],
            "hard_gates": {
                "keep_uv_policy": True,
                "zero_positive_area_overlap": True,
                "inter_chart_margin_16px": True,
                "page_boundary_margin_16px": True,
                "in_unit_page": True,
                "occupancy_within_page_budget": True,
            },
            "page_layout": {
                "schema": "car-city.ferrari-page-repack-uv-layout/v9",
                "page_id": "synthetic_page",
                "resolution": [256, 256],
                "objects": [
                    {
                        "object_id": 4,
                        "faces": 2,
                        "targeted_faces": 2,
                        "path": str(layout_npz),
                        "sha256": sha256_file(layout_npz),
                    }
                ],
            },
        }
        placement_path = temporary_root / "fixed_keep_placement_report.json"
        placement_path.write_text(json.dumps(placement, indent=2) + "\n", encoding="utf-8")
        atlas = np.zeros((256, 256, 3), dtype=np.uint8)
        atlas[:] = [23, 117, 211]
        atlas_path = temporary_root / "source_atlas.png"
        require(bool(cv2.imwrite(str(atlas_path), atlas)), "Self-test source atlas write failed")
        output_dir = temporary_root / "output"
        built = build_reprojection(
            placement_report_path=placement_path,
            source_atlas_path=atlas_path,
            output_dir=output_dir,
            preview_size=None,
        )
        require(built["status"] == "passed" and output_dir.is_dir(), "Self-test atomic output failed")
        built_report = json.loads((output_dir / "reproject_page_texture_report.json").read_text())
        require(built_report["status"] == "passed", "Self-test file report failed")
        require(all(built_report["gates"].values()), "Self-test file gate failed")
        inventory_lines = (output_dir / "hashes.sha256").read_text().splitlines()
        require(len(inventory_lines) == built["hash_inventory"]["entries"], "Self-test inventory count drift")
        for line in inventory_lines:
            digest, relative = line.split("  ", 1)
            require(sha256_file(output_dir / relative) == digest, "Self-test inventory hash drift")
        require(
            not any(path.name.startswith(f".{output_dir.name}.staging.") for path in temporary_root.iterdir()),
            "Self-test atomic staging directory leaked",
        )
    return {
        "schema": f"{SCHEMA}/synthetic-test",
        "status": "passed",
        "bilinear_orientation_and_centre": True,
        "barycentric_constant_reprojection": True,
        "repeat_deterministic": True,
        "zero_cross_domain_raster_collisions": True,
        "subpixel_chart_completion": True,
        "chart_completion_never_overwrites_owner": True,
        "single_donor_gutters_no_average": True,
        "atomic_full_file_contract": True,
        "png_roundtrip_and_hash_inventory": True,
        "surface_texels": int(occupied.sum()),
        "gutter_texels": gutter_stats["gutter_texels"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-repack-report", type=Path)
    parser.add_argument("--source-atlas", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--preview-size",
        type=int,
        help="Raster a smaller square preview (use 2048 for the review pass).",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print("V9_REPROJECT_SELF_TEST=" + json.dumps(run_self_test(), sort_keys=True))
        return 0
    require(args.page_repack_report is not None, "--page-repack-report is required")
    require(args.source_atlas is not None, "--source-atlas is required")
    require(args.output_dir is not None, "--output-dir is required")
    result = build_reprojection(
        placement_report_path=args.page_repack_report,
        source_atlas_path=args.source_atlas,
        output_dir=args.output_dir,
        preview_size=args.preview_size,
    )
    print("V9_PAGE_TEXTURE_REPROJECT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
