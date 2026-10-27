#!/usr/bin/env python3
"""Dependency-light UV topology, overlap and distortion metrics for v9."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np

from contract_v9 import PACK_MARGIN_PIXELS, MICROCHART_MAX_FACES, UV_EPSILON, QualityGate, require


def hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


class DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.rank = np.zeros(count, dtype=np.int8)

    def find(self, item: int) -> int:
        while int(self.parent[item]) != item:
            self.parent[item] = self.parent[int(self.parent[item])]
            item = int(self.parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def edge_uses(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    owners: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int32).tolist()):
        for c0, c1 in ((0, 1), (1, 2), (2, 0)):
            a, b = int(face[c0]), int(face[c1])
            owners[(a, b) if a < b else (b, a)].append((face_index, c0, c1))
    return owners


def face_components(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dsu = DisjointSet(len(faces))
    for linked in edge_uses(faces).values():
        for other in linked[1:]:
            dsu.union(int(linked[0][0]), int(other[0]))
    roots = np.asarray([dsu.find(index) for index in range(len(faces))], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), np.bincount(labels).astype(np.int32)


def _edge_endpoint_uv(
    faces: np.ndarray,
    uv: np.ndarray,
    edge: tuple[int, int],
    use: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    face_index, c0, c1 = use
    mapped = {
        int(faces[face_index, c0]): uv[face_index, c0],
        int(faces[face_index, c1]): uv[face_index, c1],
    }
    return mapped[edge[0]], mapped[edge[1]]


def uv_islands(faces: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dsu = DisjointSet(len(faces))
    for edge, linked in edge_uses(faces).items():
        if len(linked) != 2:
            continue
        left = _edge_endpoint_uv(faces, uv, edge, linked[0])
        right = _edge_endpoint_uv(faces, uv, edge, linked[1])
        if max(float(np.max(np.abs(left[0] - right[0]))), float(np.max(np.abs(left[1] - right[1])))) <= UV_EPSILON:
            dsu.union(int(linked[0][0]), int(linked[1][0]))
    roots = np.asarray([dsu.find(index) for index in range(len(faces))], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), np.bincount(labels).astype(np.int32)


def signed_double_area(uv: np.ndarray) -> np.ndarray:
    return (
        (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
        - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0])
    )


def strict_triangle_overlap(a: np.ndarray, b: np.ndarray, epsilon: float = 1.0e-12) -> bool:
    for triangle in (a, b):
        for index in range(3):
            edge = triangle[(index + 1) % 3] - triangle[index]
            axis = np.asarray((-edge[1], edge[0]), dtype=np.float64)
            length = float(np.linalg.norm(axis))
            if length <= epsilon:
                continue
            pa, pb = a @ axis, b @ axis
            amount = min(float(pa.max()), float(pb.max())) - max(float(pa.min()), float(pb.min()))
            if amount <= epsilon * length:
                return False
    return True


def overlap_audit(uv: np.ndarray, chart_id: np.ndarray, include_pairs: bool = False) -> dict[str, Any]:
    uv = np.asarray(uv, dtype=np.float64)
    tri_min, tri_max = uv.min(axis=1), uv.max(axis=1)
    valid = np.abs(signed_double_area(uv)) > 1.0e-14
    grid_size = 192
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(valid).tolist():
        low = np.clip(np.floor(tri_min[index] * grid_size).astype(int), 0, grid_size - 1)
        high = np.clip(np.floor(tri_max[index] * grid_size).astype(int), 0, grid_size - 1)
        for x in range(int(low[0]), int(high[0]) + 1):
            for y in range(int(low[1]), int(high[1]) + 1):
                cells[(x, y)].append(index)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    examples: list[list[int]] = []
    inter = 0
    for candidates in cells.values():
        for a_index in range(len(candidates)):
            for b_index in range(a_index + 1, len(candidates)):
                a, b = sorted((candidates[a_index], candidates[b_index]))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                if np.any(tri_max[a] <= tri_min[b] + 1.0e-12) or np.any(tri_max[b] <= tri_min[a] + 1.0e-12):
                    continue
                if strict_triangle_overlap(uv[a], uv[b]):
                    pairs.append((a, b))
                    inter += int(chart_id[a] != chart_id[b])
                    if len(examples) < 20:
                        examples.append([a, b])
    report: dict[str, Any] = {
        "positive_area_triangle_overlap_pairs": len(pairs),
        "positive_area_inter_chart_overlap_pairs": inter,
        "positive_area_intra_chart_overlap_pairs": len(pairs) - inter,
        "positive_area_overlap_faces": len({face for pair in pairs for face in pair}),
        "overlap_examples": examples,
        "candidate_pairs_tested": len(seen),
    }
    if include_pairs:
        report["pairs"] = pairs
    return report


def _point_segment_distance(point: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    edge = right - left
    length2 = float(edge @ edge)
    if length2 <= 1.0e-30:
        return float(np.linalg.norm(point - left))
    amount = float(np.clip(((point - left) @ edge) / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (left + amount * edge)))


def triangle_distance(a: np.ndarray, b: np.ndarray) -> float:
    if strict_triangle_overlap(a, b):
        return 0.0
    best = math.inf
    for point in a:
        for edge in range(3):
            best = min(best, _point_segment_distance(point, b[edge], b[(edge + 1) % 3]))
    for point in b:
        for edge in range(3):
            best = min(best, _point_segment_distance(point, a[edge], a[(edge + 1) % 3]))
    return float(best)


def chart_margin_audit(
    uv: np.ndarray,
    chart_id: np.ndarray,
    resolution: int,
    margin_pixels: float,
) -> dict[str, Any]:
    """Exact triangle-distance audit, spatially bounded to the margin radius."""

    uv = np.asarray(uv, dtype=np.float64)
    requested = float(margin_pixels) / float(resolution)
    tri_min = uv.min(axis=1) - requested
    tri_max = uv.max(axis=1) + requested
    grid_size = 256
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in range(len(uv)):
        low = np.clip(np.floor(tri_min[index] * grid_size).astype(int), 0, grid_size - 1)
        high = np.clip(np.floor(tri_max[index] * grid_size).astype(int), 0, grid_size - 1)
        for x in range(int(low[0]), int(high[0]) + 1):
            for y in range(int(low[1]), int(high[1]) + 1):
                cells[(x, y)].append(index)
    seen: set[tuple[int, int]] = set()
    violations: list[list[float]] = []
    minimum = math.inf
    for candidates in cells.values():
        for local_a in range(len(candidates)):
            for local_b in range(local_a + 1, len(candidates)):
                a, b = sorted((candidates[local_a], candidates[local_b]))
                if (a, b) in seen or int(chart_id[a]) == int(chart_id[b]):
                    continue
                seen.add((a, b))
                delta = np.maximum(0.0, np.maximum(uv[a].min(axis=0) - uv[b].max(axis=0), uv[b].min(axis=0) - uv[a].max(axis=0)))
                if float(np.linalg.norm(delta)) >= requested + 1.0e-12:
                    continue
                distance = triangle_distance(uv[a], uv[b])
                minimum = min(minimum, distance)
                if distance < requested - 1.0e-10 and len(violations) < 40:
                    violations.append([int(a), int(b), float(distance * resolution)])
    return {
        "requested_margin_pixels": float(margin_pixels),
        "minimum_inter_chart_triangle_distance_pixels": None if not math.isfinite(minimum) else float(minimum * resolution),
        "inter_chart_margin_violation_count_capped": len(violations),
        "inter_chart_margin_violation_examples": violations,
        "candidate_pairs_tested": len(seen),
        "passed": not violations,
    }


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    require(len(values) == len(weights) and len(values) > 0, "Invalid weighted percentile input")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    require(float(cumulative[-1]) > 0.0, "Zero percentile weight")
    target = np.clip(percentile / 100.0, 0.0, 1.0) * float(cumulative[-1])
    return float(ordered[min(int(np.searchsorted(cumulative, target, side="left")), len(ordered) - 1)])


def triangle_distortion(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, resolution: int) -> dict[str, Any]:
    triangle = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int32)]
    edge_x = triangle[:, 1] - triangle[:, 0]
    edge_y = triangle[:, 2] - triangle[:, 0]
    length_x = np.linalg.norm(edge_x, axis=1)
    unit_x = edge_x / np.maximum(length_x[:, None], 1.0e-30)
    local_y_x = np.einsum("ij,ij->i", edge_y, unit_x)
    normal_y = edge_y - local_y_x[:, None] * unit_x
    local_y_y = np.linalg.norm(normal_y, axis=1)
    geom_double_area = length_x * local_y_y
    uv_edge_x = uv[:, 1] - uv[:, 0]
    uv_edge_y = uv[:, 2] - uv[:, 0]
    uv_double_area = np.abs(signed_double_area(uv))
    valid = (geom_double_area > 1.0e-15) & (uv_double_area > 1.0e-15)
    require(bool(np.all(valid)), "Degenerate 3D/UV triangle in distortion audit")

    condition = np.empty(len(faces), dtype=np.float64)
    for index in range(len(faces)):
        source = np.asarray(((length_x[index], local_y_x[index]), (0.0, local_y_y[index])), dtype=np.float64)
        target = np.column_stack((uv_edge_x[index], uv_edge_y[index]))
        jacobian = target @ np.linalg.inv(source)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        condition[index] = float(singular[0] / max(singular[-1], 1.0e-30))
    geom_area = 0.5 * geom_double_area
    density = np.sqrt(uv_double_area / geom_double_area) * float(resolution)
    p05 = weighted_percentile(density, geom_area, 5.0)
    p50 = weighted_percentile(density, geom_area, 50.0)
    p95 = weighted_percentile(density, geom_area, 95.0)
    return {
        "condition_p50": weighted_percentile(condition, geom_area, 50.0),
        "condition_p95": weighted_percentile(condition, geom_area, 95.0),
        "condition_p99": weighted_percentile(condition, geom_area, 99.0),
        "density_p05": p05,
        "density_p50": p50,
        "density_p95": p95,
        "density_p95_p05": p95 / max(p05, 1.0e-30),
        "geometry_area": float(geom_area.sum()),
        "uv_area": float((0.5 * uv_double_area).sum()),
        "effective_texels": float((0.5 * uv_double_area).sum() * resolution**2),
    }


def seam_metrics(faces: np.ndarray, uv: np.ndarray) -> dict[str, Any]:
    manifold = cut = 0
    for edge, linked in edge_uses(faces).items():
        if len(linked) != 2:
            continue
        manifold += 1
        left = _edge_endpoint_uv(faces, uv, edge, linked[0])
        right = _edge_endpoint_uv(faces, uv, edge, linked[1])
        if max(float(np.max(np.abs(left[0] - right[0]))), float(np.max(np.abs(left[1] - right[1])))) > UV_EPSILON:
            cut += 1
    return {
        "manifold_edges": manifold,
        "cut_internal_edges": cut,
        "seam_edge_ratio": cut / max(manifold, 1),
        "seam_edge_percent": 100.0 * cut / max(manifold, 1),
    }


def uv_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    resolution: int,
    orientation_corner_order: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    uv = np.asarray(uv, dtype=np.float64)
    require(uv.shape == (len(faces), 3, 2), "UV/face shape mismatch")
    require(np.isfinite(uv).all(), "Non-finite UV")
    chart_id, chart_sizes = uv_islands(faces, uv)
    if orientation_corner_order is None:
        orientation_corner_order = np.tile(np.arange(3, dtype=np.int8), (len(faces), 1))
    rows = np.arange(len(faces), dtype=np.int32)[:, None]
    oriented_faces = faces[rows, orientation_corner_order]
    oriented_uv = uv[rows, orientation_corner_order]
    signed = signed_double_area(oriented_uv)
    topology_component, topology_sizes = face_components(faces)
    avoidable_micro = 0
    for chart in range(len(chart_sizes)):
        members = np.flatnonzero(chart_id == chart)
        if len(members) <= MICROCHART_MAX_FACES:
            component_values = np.unique(topology_component[members])
            if any(int(topology_sizes[value]) > MICROCHART_MAX_FACES for value in component_values):
                avoidable_micro += 1
    uv_min = uv.reshape(-1, 2).min(axis=0)
    uv_max = uv.reshape(-1, 2).max(axis=0)
    report = {
        "faces": int(len(faces)),
        "charts": int(len(chart_sizes)),
        "chart_faces_min": int(chart_sizes.min()),
        "chart_faces_median": float(np.median(chart_sizes)),
        "chart_faces_max": int(chart_sizes.max()),
        "avoidable_microcharts_le4_faces": int(avoidable_micro),
        "topology_components": int(len(topology_sizes)),
        "uv_degenerate": int(np.count_nonzero(np.abs(signed) <= 1.0e-14)),
        "uv_flipped_in_solver_orientation": int(np.count_nonzero(signed < -1.0e-14)),
        "source_order_negative_signed_faces": int(np.count_nonzero(signed_double_area(uv) < -1.0e-14)),
        "uv_bounds_min": uv_min.tolist(),
        "uv_bounds_max": uv_max.tolist(),
        "minimum_boundary_margin_pixels": float(min(uv_min.min(), (1.0 - uv_max).min()) * resolution),
        **seam_metrics(faces, uv),
        **triangle_distortion(vertices, oriented_faces, oriented_uv, resolution),
        "overlap": overlap_audit(uv, chart_id),
    }
    return report, chart_id


def gate_report(metrics: dict[str, Any], gate: QualityGate) -> dict[str, Any]:
    checks = {
        "charts_min": int(metrics["charts"]) >= gate.charts_min,
        "charts_max": int(metrics["charts"]) <= gate.charts_max,
        "avoidable_microcharts": int(metrics["avoidable_microcharts_le4_faces"]) <= gate.avoidable_microcharts_max,
        "seam_edge_percent": float(metrics["seam_edge_percent"]) <= gate.seam_edge_percent_max,
        "condition_p95": float(metrics["condition_p95"]) <= gate.condition_p95_max,
        "density_p95_p05": float(metrics["density_p95_p05"]) <= gate.density_p95_p05_max,
        "zero_uv_degenerate": int(metrics["uv_degenerate"]) == 0,
        "zero_uv_flipped_solver_orientation": int(metrics["uv_flipped_in_solver_orientation"]) == 0,
        "zero_positive_area_overlap": int(metrics["overlap"]["positive_area_triangle_overlap_pairs"]) == 0,
        "in_unit_page": min(metrics["uv_bounds_min"]) >= -1.0e-7 and max(metrics["uv_bounds_max"]) <= 1.0 + 1.0e-7,
        "local_boundary_margin_16px": float(metrics["minimum_boundary_margin_pixels"]) >= PACK_MARGIN_PIXELS - 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "limits": {
            "charts": [gate.charts_min, gate.charts_max],
            "avoidable_microcharts_le4_faces_max": gate.avoidable_microcharts_max,
            "seam_edge_percent_max": gate.seam_edge_percent_max,
            "condition_p95_max": gate.condition_p95_max,
            "density_p95_p05_max": gate.density_p95_p05_max,
        },
    }
