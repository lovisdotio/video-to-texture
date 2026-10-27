#!/usr/bin/env python3
"""Headless targeted UV candidate solver; emits additive local overlays only.

Run with Blender ``--background --factory-startup --python ... -- [args]``.
The command refuses more than two targets unless explicitly unlocked.  It
never opens/saves a production blend and never mutates a source sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bpy
import bmesh
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_ROOT.parent
for value in (SCRIPT_ROOT, V8_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contract_v9 import (  # noqa: E402
    PACK_MARGIN_PIXELS,
    PATCH_SCHEMA,
    ROOT,
    SCHEMA,
    parse_selectors,
    require,
    safe_output,
    sha256_file,
    verify_authorities,
)
from metrics_v9 import (  # noqa: E402
    face_components,
    gate_report,
    hash_array,
    overlap_audit,
    uv_islands,
    uv_metrics,
)
from source_v9 import load_target, orient_faces_solver_only  # noqa: E402
import unwrap_pack_nonbody_v8_blender as v8_builder  # noqa: E402


core = v8_builder.core
SUPPORTED_METHODS = ("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")
SUPPORTED_STRATEGIES = ("global_candidates", "componentwise_v8_minimum_stretch")
COMPONENTWISE_TARGET_IDS = (17, 18, 19, 28, 36)
SUPPORTED_SEAM_MODES = (
    "natural",
    "v8_long_50",
    "v8",
    "baseline_long_10",
    "baseline_long_25",
    "baseline_component_smallest",
    "baseline_component_largest",
    "baseline",
)


def blender_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def clear_scene() -> None:
    core.clear_scene()


def _unwrap_pack(obj: bpy.types.Object, method: str, resolution: int) -> dict[str, Any]:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    previous_sync = bpy.context.scene.tool_settings.use_uv_select_sync
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    editable = bmesh.from_edit_mesh(obj.data)
    editable.faces.ensure_lookup_table()
    for face in editable.faces:
        face.select_set(True)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    kwargs: dict[str, Any] = {
        "method": method,
        "fill_holes": False,
        "correct_aspect": True,
        "margin_method": "FRACTION",
        "margin": 0.0,
        "no_flip": True,
    }
    if method == "MINIMUM_STRETCH":
        # Twenty iterations left measurable stretch on the large cockpit
        # components.  One hundred stays bounded while converging reliably.
        kwargs["iterations"] = 100
    started = time.perf_counter()
    unwrap_result = sorted(bpy.ops.uv.unwrap(**kwargs))
    unwrap_seconds = time.perf_counter() - started
    scale_result = sorted(bpy.ops.uv.average_islands_scale(scale_uv=True, shear=True))
    pack_started = time.perf_counter()
    pack_result = sorted(
        bpy.ops.uv.pack_islands(
            udim_source="CLOSEST_UDIM",
            rotate=True,
            rotate_method="ANY",
            scale=True,
            merge_overlap=False,
            margin_method="FRACTION",
            margin=PACK_MARGIN_PIXELS / resolution,
            pin=False,
            shape_method="CONCAVE",
        )
    )
    pack_seconds = time.perf_counter() - pack_started
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.scene.tool_settings.use_uv_select_sync = previous_sync
    obj.data.update()
    return {
        "unwrap_operator": unwrap_result,
        "average_islands_scale_operator": scale_result,
        "pack_operator": pack_result,
        "unwrap_seconds": unwrap_seconds,
        "pack_seconds": pack_seconds,
    }


def _repack_existing_uv(
    obj: bpy.types.Object,
    uv: np.ndarray,
    resolution: int,
    *,
    shape_method: str = "CONCAVE",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Common-density repack without changing any island's local shape."""

    core.write_face_uv(obj.data, uv)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    previous_sync = bpy.context.scene.tool_settings.use_uv_select_sync
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    started = time.perf_counter()
    scale_result = sorted(bpy.ops.uv.average_islands_scale(scale_uv=True, shear=True))
    pack_result = sorted(
        bpy.ops.uv.pack_islands(
            udim_source="CLOSEST_UDIM",
            rotate=True,
            rotate_method="ANY",
            scale=True,
            merge_overlap=False,
            margin_method="FRACTION",
            margin=PACK_MARGIN_PIXELS / resolution,
            pin=False,
            shape_method=shape_method,
        )
    )
    seconds = time.perf_counter() - started
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.scene.tool_settings.use_uv_select_sync = previous_sync
    obj.data.update()
    return core.face_uv_array(obj.data), {
        "mode": "average_common_geometric_density_then_pack_existing_islands",
        "average_islands_scale_operator": scale_result,
        "pack_operator": pack_result,
        "seconds": seconds,
        "margin_pixels": PACK_MARGIN_PIXELS,
        "resolution": resolution,
        "shape_method": shape_method,
    }


def _normalize_solver_orientation(
    faces: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    face_island, sizes = core.uv_islands(faces, uv)
    signed = core.signed_double_area(uv)
    zero = np.flatnonzero(np.abs(signed) <= 1.0e-14)
    require(not len(zero), f"UV degenerates after unwrap: {zero[:20].tolist()}")
    mixed = []
    result = uv.copy()
    mirrored = 0
    for island in np.unique(face_island).tolist():
        mask = face_island == island
        local = signed[mask]
        if bool(np.any(local > 0.0)) and bool(np.any(local < 0.0)):
            mixed.append(int(island))
            continue
        if bool(np.all(local < 0.0)):
            centre = 0.5 * (float(result[mask, :, 0].min()) + float(result[mask, :, 0].max()))
            result[mask, :, 0] = 2.0 * centre - result[mask, :, 0]
            mirrored += 1
    require(not mixed, f"Mixed UV winding islands after direct unwrap: {mixed[:20]}")
    return result, {
        "solver_charts": int(len(sizes)),
        "whole_solver_islands_mirrored": mirrored,
        "mixed_solver_islands": 0,
    }


def _edge_lengths(vertices: np.ndarray, seams: np.ndarray) -> np.ndarray:
    if not len(seams):
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(vertices[seams[:, 1]] - vertices[seams[:, 0]], axis=1)


def seam_candidates(target, modes: tuple[str, ...]) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    for mode in modes:
        if mode == "natural":
            seams = np.empty((0, 2), dtype=np.int32)
        elif mode == "v8":
            seams = target.v8_seams.copy()
        elif mode == "baseline":
            seams = target.baseline_seams.copy()
        elif mode in {"baseline_long_10", "baseline_long_25"}:
            seams = target.baseline_seams.copy()
            if len(seams) > 1:
                fraction = 0.10 if mode.endswith("10") else 0.25
                length = _edge_lengths(target.compact_vertices, seams)
                order = np.argsort(-length, kind="mergesort")
                seams = seams[order[: max(1, int(math.ceil(len(seams) * fraction)))]]
                seams = np.asarray(sorted(map(tuple, seams.tolist())), dtype=np.int32)
        elif mode in {"baseline_component_smallest", "baseline_component_largest"}:
            source = target.baseline_seams.copy()
            vertex_to_edges: dict[int, list[int]] = {}
            for edge_index, edge in enumerate(source.tolist()):
                for vertex in edge:
                    vertex_to_edges.setdefault(int(vertex), []).append(edge_index)
            unseen = set(range(len(source)))
            components: list[list[int]] = []
            while unseen:
                seed = min(unseen)
                unseen.remove(seed)
                stack = [seed]
                group = []
                while stack:
                    edge_index = stack.pop()
                    group.append(edge_index)
                    for vertex in source[edge_index].tolist():
                        for neighbor in vertex_to_edges[int(vertex)]:
                            if neighbor in unseen:
                                unseen.remove(neighbor)
                                stack.append(neighbor)
                components.append(sorted(group))
            require(components, f"{target.spec.selector}: no baseline seam component")
            components.sort(key=lambda group: (len(group), group))
            chosen = components[0] if mode.endswith("smallest") else components[-1]
            seams = source[np.asarray(chosen, dtype=np.int32)]
        elif mode == "v8_long_50":
            seams = target.v8_seams.copy()
            if len(seams) > 1:
                length = _edge_lengths(target.compact_vertices, seams)
                order = np.argsort(-length, kind="mergesort")
                seams = seams[order[: max(1, int(math.ceil(len(seams) * 0.5)))]]
                seams = np.asarray(sorted(map(tuple, seams.tolist())), dtype=np.int32)
        else:
            raise RuntimeError(f"Unsupported seam mode: {mode}")
        digest = hash_array(np.asarray(seams, dtype=np.int32).reshape(-1, 2))
        if any(hash_array(existing) == digest for _, existing in rows):
            continue
        rows.append((mode, np.asarray(seams, dtype=np.int32).reshape(-1, 2)))
    return rows


def solve_candidate(target, seam_mode: str, seams: np.ndarray, method: str) -> dict[str, Any]:
    started = time.perf_counter()
    core.ATLAS_RESOLUTION = target.spec.resolution
    core.PACK_MARGIN_UV = PACK_MARGIN_PIXELS / target.spec.resolution
    oriented_faces, corner_order, orientation = orient_faces_solver_only(target.compact_faces)
    solver_vertices, solver_faces, solver_source, topology = v8_builder.split_for_parameterization(
        target.compact_vertices,
        oriented_faces,
        seams,
    )
    obj = core.build_mesh(
        f"v9_{target.spec.slug}_{seam_mode}_{method}",
        solver_vertices,
        solver_faces,
    )
    try:
        operators = _unwrap_pack(obj, method, target.spec.resolution)
        solver_uv = core.face_uv_array(obj.data)
        solver_uv, orientation_uv = _normalize_solver_orientation(solver_faces, solver_uv)
        restored_uv = np.empty_like(solver_uv)
        rows = np.arange(len(solver_uv), dtype=np.int32)[:, None]
        restored_uv[rows, corner_order] = solver_uv
        metrics, chart_id = uv_metrics(
            target.compact_vertices,
            target.compact_faces,
            restored_uv,
            target.spec.resolution,
            orientation_corner_order=corner_order,
        )
        residual_repairs = []
        for repair_round in range(2):
            if metrics["overlap"]["positive_area_triangle_overlap_pairs"] == 0:
                break
            solver_chart, _ = core.uv_islands(solver_faces, solver_uv)
            overlap = core.overlap_audit(solver_uv, solver_chart, include_all_pairs=True)
            pairs = overlap.pop("_all_pairs")
            proven_faces = {int(face) for pair in pairs for face in pair}
            require(proven_faces, "Overlap metric disagreement before bounded repair")
            core.write_face_uv(obj.data, solver_uv)
            repair = core.smart_project_proven_faces(obj, proven_faces)
            solver_uv = core.face_uv_array(obj.data)
            solver_uv, normalized = _normalize_solver_orientation(solver_faces, solver_uv)
            restored_uv = np.empty_like(solver_uv)
            restored_uv[rows, corner_order] = solver_uv
            metrics, chart_id = uv_metrics(
                target.compact_vertices,
                target.compact_faces,
                restored_uv,
                target.spec.resolution,
                orientation_corner_order=corner_order,
            )
            residual_repairs.append({
                "round": repair_round + 1,
                "proven_overlap_pairs": len(pairs),
                "proven_faces": len(proven_faces),
                "repair": repair,
                "orientation": normalized,
                "remaining_overlap_pairs": metrics["overlap"]["positive_area_triangle_overlap_pairs"],
            })
        quality = gate_report(metrics, target.spec.gate)
        return {
            "status": "passed_local_gate" if quality["passed"] else "failed_local_gate",
            "strategy": "global_candidates",
            "seam_mode": seam_mode,
            "unwrap_method": method,
            "requested_seams": int(len(seams)),
            "requested_seams_sha256": hash_array(seams),
            "solver_orientation": orientation,
            "solver_topology": topology,
            "operators": operators,
            "orientation_uv": orientation_uv,
            "bounded_overlap_repairs": residual_repairs,
            "metrics": metrics,
            "quality_gate": quality,
            "runtime_seconds": float(time.perf_counter() - started),
            "replacement_uv": restored_uv.astype(np.float32),
            "local_chart_id": chart_id.astype(np.int32),
            "solver_corner_to_source_corner": corner_order.astype(np.int8),
            "solver_vertex_source": solver_source.astype(np.int32),
        }
    finally:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _component_compact_input(target, face_indices: np.ndarray) -> dict[str, np.ndarray]:
    selected_faces = np.asarray(target.compact_faces[face_indices], dtype=np.int32)
    selected_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    selected_vertices = selected_vertices.astype(np.int32)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    old_to_new = np.full(len(target.compact_vertices), -1, dtype=np.int32)
    old_to_new[selected_vertices] = np.arange(len(selected_vertices), dtype=np.int32)
    raw_seams = np.asarray(target.v8_seams, dtype=np.int32).reshape(-1, 2)
    if len(raw_seams):
        mask = np.all(old_to_new[raw_seams] >= 0, axis=1)
        seams = np.sort(old_to_new[raw_seams[mask]], axis=1).astype(np.int32)
        seams = np.unique(seams, axis=0)
    else:
        seams = np.empty((0, 2), dtype=np.int32)
    return {
        "face_indices": np.asarray(face_indices, dtype=np.int32),
        "vertex_indices": selected_vertices,
        "vertices": np.asarray(target.compact_vertices[selected_vertices], dtype=np.float64),
        "faces": faces,
        "baseline_uv": np.asarray(target.compact_baseline_uv[face_indices], dtype=np.float64),
        "v8_seams": seams,
    }


def _solve_minimum_stretch_component(
    target,
    component_id: int,
    component: dict[str, np.ndarray],
) -> dict[str, Any]:
    oriented_faces, corner_order, orientation = orient_faces_solver_only(component["faces"])
    solver_vertices, solver_faces, solver_source, topology = v8_builder.split_for_parameterization(
        component["vertices"],
        oriented_faces,
        component["v8_seams"],
    )
    obj = core.build_mesh(
        f"v9_{target.spec.slug}_component_{component_id}_minimum_stretch_100",
        solver_vertices,
        solver_faces,
    )
    try:
        operators = _unwrap_pack(obj, "MINIMUM_STRETCH", target.spec.resolution)
        solver_uv = core.face_uv_array(obj.data)
        solver_uv, orientation_uv = _normalize_solver_orientation(solver_faces, solver_uv)
        restored_uv = np.empty_like(solver_uv)
        rows = np.arange(len(solver_uv), dtype=np.int32)[:, None]
        restored_uv[rows, corner_order] = solver_uv
        metrics, chart_id = uv_metrics(
            component["vertices"],
            component["faces"],
            restored_uv,
            target.spec.resolution,
            orientation_corner_order=corner_order,
        )
        require(metrics["uv_degenerate"] == 0, "Component UV degenerate")
        require(metrics["uv_flipped_in_solver_orientation"] == 0, "Component UV fold")
        require(
            metrics["overlap"]["positive_area_triangle_overlap_pairs"] == 0,
            "Component positive-area overlap",
        )
        return {
            "uv": restored_uv.astype(np.float32),
            "corner_order": corner_order.astype(np.int8),
            "chart_id": chart_id.astype(np.int32),
            "provenance": {
                "component_id": int(component_id),
                "faces": int(len(component["faces"])),
                "vertices": int(len(component["vertices"])),
                "v8_requested_seams": int(len(component["v8_seams"])),
                "selected_source": "minimum_stretch_100",
                "fallback_used": False,
                "operators": operators,
                "solver_orientation": orientation,
                "solver_topology": topology,
                "orientation_uv": orientation_uv,
                "charts": int(metrics["charts"]),
                "condition_p95": float(metrics["condition_p95"]),
                "density_p95_p05": float(metrics["density_p95_p05"]),
                "overlap_pairs": 0,
                "uv_sha256": hash_array(restored_uv.astype(np.float32)),
                "solver_vertex_source_sha256": hash_array(solver_source),
            },
        }
    finally:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _fallback_component(
    target,
    component_id: int,
    component: dict[str, np.ndarray],
    error: Exception,
) -> dict[str, Any]:
    uv = np.asarray(component["baseline_uv"], dtype=np.float32)
    corner_order = np.tile(np.arange(3, dtype=np.int8), (len(uv), 1))
    metrics, chart_id = uv_metrics(
        component["vertices"],
        component["faces"],
        uv,
        target.spec.resolution,
        orientation_corner_order=corner_order,
    )
    require(metrics["uv_degenerate"] == 0, f"Baseline fallback degenerate: component {component_id}")
    require(metrics["uv_flipped_in_solver_orientation"] == 0, f"Baseline fallback folded: component {component_id}")
    require(
        metrics["overlap"]["positive_area_triangle_overlap_pairs"] == 0,
        f"Baseline fallback overlaps: component {component_id}",
    )
    return {
        "uv": uv,
        "corner_order": corner_order,
        "chart_id": chart_id.astype(np.int32),
        "provenance": {
            "component_id": int(component_id),
            "faces": int(len(component["faces"])),
            "vertices": int(len(component["vertices"])),
            "v8_requested_seams": int(len(component["v8_seams"])),
            "selected_source": "v8_baseline_fallback",
            "fallback_used": True,
            "fallback_reason_type": type(error).__name__,
            "fallback_reason": str(error),
            "charts": int(metrics["charts"]),
            "condition_p95": float(metrics["condition_p95"]),
            "density_p95_p05": float(metrics["density_p95_p05"]),
            "overlap_pairs": 0,
            "uv_sha256": hash_array(uv),
        },
    }


def _assemble_component_results(
    target,
    component_ids: list[int],
    component_outputs: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Restore independently solved component rows to target face order."""

    combined_uv = np.empty_like(target.compact_baseline_uv, dtype=np.float32)
    combined_corner_order = np.empty((len(target.compact_faces), 3), dtype=np.int8)
    covered = np.zeros(len(target.compact_faces), dtype=bool)
    for component_id in component_ids:
        output = component_outputs[component_id]
        face_indices = output["face_indices"]
        result = output["result"]
        require(not bool(np.any(covered[face_indices])), f"Duplicate component faces: {component_id}")
        combined_uv[face_indices] = result["uv"]
        combined_corner_order[face_indices] = result["corner_order"]
        covered[face_indices] = True
    require(bool(np.all(covered)), "Component assembly did not cover every target face")
    return combined_uv, combined_corner_order


def _packed_overlap_components(
    target,
    packed_uv: np.ndarray,
    stable_components: np.ndarray,
) -> tuple[dict[str, Any], dict[int, int]]:
    """Return exact positive-area overlaps and their topology components."""

    chart_id, _ = uv_islands(target.compact_faces, packed_uv)
    overlap = overlap_audit(packed_uv, chart_id, include_pairs=True)
    pairs = overlap.pop("pairs")
    counts: dict[int, int] = {}
    for left, right in pairs:
        for component_id in {
            int(stable_components[int(left)]),
            int(stable_components[int(right)]),
        }:
            counts[component_id] = counts.get(component_id, 0) + 1
    overlap["topology_component_ids"] = sorted(counts)
    overlap["topology_component_pair_incidence"] = {
        str(component_id): int(counts[component_id]) for component_id in sorted(counts)
    }
    return overlap, counts


def _deterministic_chart_aabb_pack(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack chart AABBs on deterministic shelves using direct similarities.

    This is deliberately simpler than Blender's concave packer.  It is only a
    bounded structural fallback: every chart is translated, uniformly scaled,
    and optionally rotated by +90 degrees, so its local parameterization and
    winding cannot change.  AABB separation makes positive-area overlap
    impossible by construction.
    """

    uv = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    vertices = np.asarray(vertices, dtype=np.float64)
    chart_id, chart_sizes = uv_islands(faces, uv)
    triangle = vertices[faces]
    geometry_area = 0.5 * np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]),
        axis=1,
    )
    uv_area = 0.5 * np.abs(
        (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
        - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0])
    )
    require(bool(np.all(geometry_area > 1.0e-15)), "Degenerate geometry before deterministic pack")
    require(bool(np.all(uv_area > 1.0e-15)), "Degenerate UV before deterministic pack")

    charts: list[dict[str, Any]] = []
    for chart in range(len(chart_sizes)):
        face_indices = np.flatnonzero(chart_id == chart).astype(np.int32)
        chart_uv = uv[face_indices]
        source_min = chart_uv.reshape(-1, 2).min(axis=0)
        source_max = chart_uv.reshape(-1, 2).max(axis=0)
        density_scale = math.sqrt(
            float(geometry_area[face_indices].sum()) / float(uv_area[face_indices].sum())
        )
        local = (chart_uv - source_min[None, None, :]) * density_scale
        width, height = ((source_max - source_min) * density_scale).tolist()
        rotated = bool(height > width)
        if rotated:
            # (x, y) -> (height - y, x), determinant +1.
            local = np.stack((height - local[:, :, 1], local[:, :, 0]), axis=2)
            width, height = height, width
        require(width > 0.0 and height > 0.0, f"Zero AABB for chart {chart}")
        charts.append({
            "chart_id": chart,
            "face_indices": face_indices,
            "local_uv": local,
            "width": float(width),
            "height": float(height),
            "rotated_90": rotated,
        })

    # Tallest-first shelves are stable and leave ample headroom on these local
    # target sets.  Half a pixel of numerical guard survives float32 storage.
    padding = (PACK_MARGIN_PIXELS + 0.5) / float(resolution)
    gap = padding
    limit = 1.0 - padding
    order = sorted(
        charts,
        key=lambda row: (-row["height"], -row["width"], row["chart_id"]),
    )

    def shelf_layout(scale: float) -> dict[int, tuple[float, float]] | None:
        x = padding
        y = padding
        shelf_height = 0.0
        placement: dict[int, tuple[float, float]] = {}
        for row in order:
            width = row["width"] * scale
            height = row["height"] * scale
            if x > padding + 1.0e-15 and x + width > limit + 1.0e-14:
                y += shelf_height + gap
                x = padding
                shelf_height = 0.0
            if x + width > limit + 1.0e-14 or y + height > limit + 1.0e-14:
                return None
            placement[row["chart_id"]] = (x, y)
            x += width + gap
            shelf_height = max(shelf_height, height)
        return placement

    inner = 1.0 - 2.0 * padding
    high = min(
        inner / max(row["width"] for row in charts),
        inner / max(row["height"] for row in charts),
    )
    low = 0.0
    best: dict[int, tuple[float, float]] | None = None
    for _ in range(64):
        middle = 0.5 * (low + high)
        trial = shelf_layout(middle)
        if trial is None:
            high = middle
        else:
            low = middle
            best = trial
    require(best is not None and low > 0.0, "Deterministic AABB shelf pack has no feasible scale")
    # Back off by a tiny relative amount so exact boundary comparisons are not
    # decided by the final binary-search ulp.
    scale = low * (1.0 - 1.0e-10)
    placement = shelf_layout(scale)
    require(placement is not None, "Deterministic AABB shelf pack lost its feasible placement")

    packed = np.empty_like(uv)
    for row in charts:
        origin = np.asarray(placement[row["chart_id"]], dtype=np.float64)
        packed[row["face_indices"]] = origin[None, None, :] + scale * row["local_uv"]
    packed = packed.astype(np.float32)
    bounds_min = packed.reshape(-1, 2).min(axis=0)
    bounds_max = packed.reshape(-1, 2).max(axis=0)
    return packed, {
        "mode": "deterministic_density_normalized_aabb_shelves",
        "shape_method": "DETERMINISTIC_AABB_SHELF",
        "charts": int(len(charts)),
        "rotated_90_charts": int(sum(bool(row["rotated_90"]) for row in charts)),
        "uniform_atlas_scale": float(scale),
        "common_linear_density_texels_per_world_unit": float(scale * resolution),
        "padding_pixels": float(padding * resolution),
        "gap_pixels": float(gap * resolution),
        "uv_bounds_min": bounds_min.tolist(),
        "uv_bounds_max": bounds_max.tolist(),
        "placement_uv_sha256": hash_array(packed),
        "blender_operator_used": False,
    }


def solve_componentwise_candidate(target) -> dict[str, Any]:
    """Unwrap each stable topology component and fail over independently."""

    started = time.perf_counter()
    require(
        target.spec.object_id in COMPONENTWISE_TARGET_IDS,
        f"componentwise mode is scoped to {COMPONENTWISE_TARGET_IDS}, got {target.spec.object_id}",
    )
    core.ATLAS_RESOLUTION = target.spec.resolution
    core.PACK_MARGIN_UV = PACK_MARGIN_PIXELS / target.spec.resolution
    stable_components = np.asarray(
        target.topology_component_id[target.selected_face_indices], dtype=np.int32
    )
    recomputed, recomputed_sizes = face_components(target.compact_faces)
    require(
        len(np.unique(stable_components)) == len(recomputed_sizes),
        f"Stable/recomputed component count drift: {target.spec.selector}",
    )
    for label in np.unique(recomputed).tolist():
        require(
            len(np.unique(stable_components[recomputed == label])) == 1,
            f"Stable component crosses recomputed component: {target.spec.selector}:{label}",
        )

    component_outputs: dict[int, dict[str, Any]] = {}
    component_ids = sorted(map(int, np.unique(stable_components).tolist()))
    for component_index, component_id in enumerate(component_ids):
        face_indices = np.flatnonzero(stable_components == component_id).astype(np.int32)
        component = _component_compact_input(target, face_indices)
        try:
            result = _solve_minimum_stretch_component(target, component_id, component)
        except Exception as error:
            result = _fallback_component(target, component_id, component, error)
        component_outputs[component_id] = {
            "face_indices": face_indices,
            "component": component,
            "result": result,
        }
        if (component_index + 1) % 25 == 0 or component_index + 1 == len(component_ids):
            fallback_count = sum(
                bool(output["result"]["provenance"]["fallback_used"])
                for output in component_outputs.values()
            )
            print(
                f"V9_COMPONENTWISE_PROGRESS {target.spec.selector}: "
                f"{component_index + 1}/{len(component_ids)} components, fallbacks={fallback_count}",
                flush=True,
            )

    # Blender's concave packer can very rarely leave a positive-area collision
    # between otherwise valid islands.  Attribute every such collision back to
    # the stable topology component and fail over only newly implicated solves.
    # If the collision is purely a packer artifact between components already on
    # their V8 baseline, the final AABB pack is the bounded structural fallback.
    carrier = core.build_mesh(
        f"v9_{target.spec.slug}_componentwise_common_repack",
        target.compact_vertices,
        target.compact_faces,
    )
    repack_attempts: list[dict[str, Any]] = []
    packed_uv: np.ndarray | None = None
    combined_corner_order: np.ndarray | None = None
    final_overlap: dict[str, Any] | None = None
    try:
        for repair_round in range(4):
            combined_uv, combined_corner_order = _assemble_component_results(
                target, component_ids, component_outputs
            )
            packed_uv, operator_report = _repack_existing_uv(
                carrier,
                combined_uv,
                target.spec.resolution,
                shape_method="CONCAVE",
            )
            overlap, overlap_counts = _packed_overlap_components(
                target, packed_uv, stable_components
            )
            attempt: dict[str, Any] = {
                "attempt": len(repack_attempts) + 1,
                "repair_round": repair_round,
                "operators": operator_report,
                "overlap": overlap,
            }
            repack_attempts.append(attempt)
            final_overlap = overlap
            print(
                "V9_COMPONENTWISE_REPACK "
                + json.dumps({
                    "selector": target.spec.selector,
                    "attempt": attempt["attempt"],
                    "shape_method": operator_report["shape_method"],
                    "overlap_pairs": overlap["positive_area_triangle_overlap_pairs"],
                    "inter_chart_pairs": overlap["positive_area_inter_chart_overlap_pairs"],
                    "intra_chart_pairs": overlap["positive_area_intra_chart_overlap_pairs"],
                    "component_ids": overlap["topology_component_ids"],
                }, sort_keys=True),
                flush=True,
            )
            if overlap["positive_area_triangle_overlap_pairs"] == 0:
                attempt["action"] = "accepted"
                break

            fallback_ids = [
                component_id
                for component_id in sorted(overlap_counts)
                if not component_outputs[component_id]["result"]["provenance"]["fallback_used"]
            ]
            attempt["new_fallback_component_ids"] = fallback_ids
            if not fallback_ids:
                attempt["action"] = "retry_with_aabb_pack"
                break
            attempt["action"] = "fallback_implicated_components_to_v8"
            for component_id in fallback_ids:
                output = component_outputs[component_id]
                previous = output["result"]["provenance"]
                reason = RuntimeError(
                    "Positive-area overlap after common CONCAVE repack; "
                    f"component pair incidence={overlap_counts[component_id]}"
                )
                fallback = _fallback_component(
                    target,
                    component_id,
                    output["component"],
                    reason,
                )
                fallback["provenance"].update({
                    "fallback_stage": "post_common_repack_overlap",
                    "fallback_repair_round": repair_round,
                    "post_repack_overlap_pair_incidence": int(overlap_counts[component_id]),
                    "superseded_candidate": previous,
                })
                output["result"] = fallback

        require(packed_uv is not None and combined_corner_order is not None, "No common repack result")
        require(final_overlap is not None, "No common repack overlap audit")
        if final_overlap["positive_area_triangle_overlap_pairs"] != 0:
            combined_uv, combined_corner_order = _assemble_component_results(
                target, component_ids, component_outputs
            )
            packed_uv, operator_report = _deterministic_chart_aabb_pack(
                target.compact_vertices,
                target.compact_faces,
                combined_uv,
                target.spec.resolution,
            )
            final_overlap, _ = _packed_overlap_components(
                target, packed_uv, stable_components
            )
            repack_attempts.append({
                "attempt": len(repack_attempts) + 1,
                "repair_round": "aabb_structural_fallback",
                "operators": operator_report,
                "overlap": final_overlap,
                "action": "accepted" if final_overlap["positive_area_triangle_overlap_pairs"] == 0 else "failed",
            })
            print(
                "V9_COMPONENTWISE_REPACK "
                + json.dumps({
                    "selector": target.spec.selector,
                    "attempt": len(repack_attempts),
                    "shape_method": operator_report["shape_method"],
                    "overlap_pairs": final_overlap["positive_area_triangle_overlap_pairs"],
                    "inter_chart_pairs": final_overlap["positive_area_inter_chart_overlap_pairs"],
                    "intra_chart_pairs": final_overlap["positive_area_intra_chart_overlap_pairs"],
                    "component_ids": final_overlap["topology_component_ids"],
                }, sort_keys=True),
                flush=True,
            )
    finally:
        mesh = carrier.data
        bpy.data.objects.remove(carrier, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    require(packed_uv is not None and combined_corner_order is not None, "No packed componentwise UV")
    metrics, chart_id = uv_metrics(
        target.compact_vertices,
        target.compact_faces,
        packed_uv,
        target.spec.resolution,
        orientation_corner_order=combined_corner_order,
    )
    structural_checks = {
        "zero_uv_degenerate": metrics["uv_degenerate"] == 0,
        "zero_uv_fold_solver_orientation": metrics["uv_flipped_in_solver_orientation"] == 0,
        "zero_positive_area_overlap": metrics["overlap"]["positive_area_triangle_overlap_pairs"] == 0,
        "in_unit_page": min(metrics["uv_bounds_min"]) >= -1.0e-7 and max(metrics["uv_bounds_max"]) <= 1.0 + 1.0e-7,
        "minimum_16px_boundary_margin": metrics["minimum_boundary_margin_pixels"] >= PACK_MARGIN_PIXELS - 0.01,
    }
    require(all(structural_checks.values()), f"Componentwise final structural gate failed: {structural_checks}")
    quality = gate_report(metrics, target.spec.gate)
    component_rows = [
        component_outputs[component_id]["result"]["provenance"]
        for component_id in component_ids
    ]
    fallback_rows = [row for row in component_rows if row["fallback_used"]]
    return {
        "status": "passed_local_gate" if quality["passed"] else "passed_structural_gate",
        "strategy": "componentwise_v8_minimum_stretch",
        "seam_mode": "v8_per_topology_component",
        "unwrap_method": "MINIMUM_STRETCH_100",
        "requested_seams": int(len(target.v8_seams)),
        "requested_seams_sha256": hash_array(target.v8_seams),
        "component_provenance": component_rows,
        "component_summary": {
            "components": len(component_rows),
            "minimum_stretch_100_components": len(component_rows) - len(fallback_rows),
            "v8_baseline_fallback_components": len(fallback_rows),
            "fallback_component_ids": [int(row["component_id"]) for row in fallback_rows],
            "fallback_faces": sum(int(row["faces"]) for row in fallback_rows),
        },
        "common_repack": {
            "selected_attempt": len(repack_attempts),
            "selected_shape_method": repack_attempts[-1]["operators"]["shape_method"],
            "attempts": repack_attempts,
        },
        "structural_gate": {"passed": True, "checks": structural_checks},
        "metrics": metrics,
        "quality_gate": quality,
        "runtime_seconds": float(time.perf_counter() - started),
        "replacement_uv": packed_uv.astype(np.float32),
        "local_chart_id": chart_id.astype(np.int32),
        "solver_corner_to_source_corner": combined_corner_order.astype(np.int8),
    }


def candidate_score(row: dict[str, Any]) -> tuple[Any, ...]:
    if row["status"] == "error":
        return (2, 999, math.inf)
    metrics = row["metrics"]
    checks = row["quality_gate"]["checks"]
    failed = sum(not value for value in checks.values())
    return (
        0 if row["quality_gate"]["passed"] else 1,
        failed,
        int(metrics["avoidable_microcharts_le4_faces"]),
        int(metrics["charts"]),
        float(metrics["seam_edge_percent"]),
        float(metrics["condition_p95"]),
        float(metrics["density_p95_p05"]),
        row["seam_mode"],
        row["unwrap_method"],
    )


def serializable_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {
        "replacement_uv", "local_chart_id", "solver_corner_to_source_corner", "solver_vertex_source"
    }}


def save_patch(output: Path, target, selected: dict[str, Any]) -> tuple[Path, str]:
    patch_dir = output / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / f"{target.spec.slug}.npz"
    np.savez_compressed(
        path,
        schema=np.asarray(PATCH_SCHEMA),
        selector=np.asarray(target.spec.selector),
        source_kind=np.asarray(target.spec.source_kind),
        source_object_id=np.asarray(target.spec.object_id, dtype=np.int16),
        piece_component_id=np.asarray(target.spec.component_id, dtype=np.int16),
        object_name=np.asarray(target.object_name),
        target_page_id=np.asarray(target.spec.page_id),
        page_resolution=np.asarray(target.spec.resolution, dtype=np.int32),
        pack_margin_pixels=np.asarray(PACK_MARGIN_PIXELS, dtype=np.int16),
        source_path=np.asarray(str(target.source_path)),
        source_sha256=np.asarray(target.source_sha256),
        proxy_path=np.asarray(str(target.proxy_path)),
        proxy_sha256=np.asarray(target.proxy_sha256),
        selected_proxy_face_indices=target.selected_face_indices,
        selected_proxy_vertex_indices=target.selected_vertex_indices,
        compact_proxy_faces=target.compact_faces,
        replacement_proxy_corner_uv=selected["replacement_uv"],
        local_chart_id=selected["local_chart_id"],
        solver_corner_to_source_corner=selected["solver_corner_to_source_corner"],
        immutable_array_hashes_json=np.asarray(json.dumps(target.immutable_hashes, sort_keys=True)),
        selected_candidate_json=np.asarray(json.dumps(serializable_candidate(selected), sort_keys=True)),
        component_provenance_json=np.asarray(
            json.dumps(selected.get("component_provenance", []), sort_keys=True)
        ),
        atlas_coordinates_final=np.asarray(False),
        atlas_placement_required=np.asarray(True),
        keep_faces_may_move=np.asarray(False),
    )
    return path, sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", required=True, help="Comma-separated exact shortlist selectors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default="ANGLE_BASED,CONFORMAL")
    parser.add_argument("--seam-modes", default=",".join(SUPPORTED_SEAM_MODES))
    parser.add_argument(
        "--strategies",
        default="global_candidates",
        help="Comma-separated: global_candidates,componentwise_v8_minimum_stretch",
    )
    parser.add_argument("--allow-more-than-two", action="store_true")
    args = parser.parse_args(blender_args())
    started = time.perf_counter()
    require(bpy.app.background, "Run only with Blender --background")
    verify_authorities()
    specs = parse_selectors(args.selectors)
    require(args.allow_more_than_two or len(specs) <= 2, "Phase-v9 safety gate: solve at most two objects per smoke")
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    seam_modes = tuple(value.strip() for value in args.seam_modes.split(",") if value.strip())
    strategies = tuple(value.strip() for value in args.strategies.split(",") if value.strip())
    require(strategies and not (set(strategies) - set(SUPPORTED_STRATEGIES)), f"Unsupported strategies: {strategies}")
    if "global_candidates" in strategies:
        require(methods and not (set(methods) - set(SUPPORTED_METHODS)), f"Unsupported methods: {methods}")
        require(seam_modes and not (set(seam_modes) - set(SUPPORTED_SEAM_MODES)), f"Unsupported seam modes: {seam_modes}")
    output = safe_output(args.output_dir)
    require(not output.exists() or not any(output.iterdir()), f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    clear_scene()
    target_reports = []
    overall_pass = True
    overall_structural = True
    for spec in specs:
        target = load_target(spec)
        baseline_metrics, _ = uv_metrics(
            target.compact_vertices,
            target.compact_faces,
            target.compact_baseline_uv,
            spec.resolution,
        )
        candidates = []
        if "global_candidates" in strategies:
            for seam_mode, seams in seam_candidates(target, seam_modes):
                for method in methods:
                    try:
                        candidate = solve_candidate(target, seam_mode, seams, method)
                    except Exception as error:
                        candidate = {
                            "status": "error",
                            "strategy": "global_candidates",
                            "seam_mode": seam_mode,
                            "unwrap_method": method,
                            "requested_seams": int(len(seams)),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    candidates.append(candidate)
                    state = candidate["status"]
                    detail = "" if state == "error" else f", charts={candidate['metrics']['charts']}, cond95={candidate['metrics']['condition_p95']:.4f}"
                    print(f"V9_CANDIDATE {spec.selector} {seam_mode}/{method}: {state}{detail}", flush=True)
        if "componentwise_v8_minimum_stretch" in strategies:
            try:
                candidate = solve_componentwise_candidate(target)
            except Exception as error:
                candidate = {
                    "status": "error",
                    "strategy": "componentwise_v8_minimum_stretch",
                    "seam_mode": "v8_per_topology_component",
                    "unwrap_method": "MINIMUM_STRETCH_100",
                    "requested_seams": int(len(target.v8_seams)),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            candidates.append(candidate)
            state = candidate["status"]
            detail = "" if state == "error" else (
                f", charts={candidate['metrics']['charts']}, "
                f"fallbacks={candidate['component_summary']['v8_baseline_fallback_components']}, "
                f"cond95={candidate['metrics']['condition_p95']:.4f}"
            )
            print(f"V9_CANDIDATE {spec.selector} componentwise_v8_minimum_stretch: {state}{detail}", flush=True)
        selected = min(candidates, key=candidate_score)
        local_pass = selected["status"] != "error" and selected["quality_gate"]["passed"]
        structural_pass = selected["status"] != "error" and selected.get(
            "structural_gate", {"passed": True}
        )["passed"]
        overall_pass &= local_pass
        overall_structural &= structural_pass
        patch_path = patch_sha = None
        if selected["status"] != "error":
            patch_path, patch_sha = save_patch(output, target, selected)
        row = {
            "selector": spec.selector,
            "slug": spec.slug,
            "source_kind": spec.source_kind,
            "object_id": spec.object_id,
            "component_id": spec.component_id,
            "page_id": spec.page_id,
            "resolution": [spec.resolution, spec.resolution],
            "selected_faces": int(len(target.selected_face_indices)),
            "source": {"path": str(target.source_path), "sha256": target.source_sha256},
            "proxy": {"path": str(target.proxy_path), "sha256": target.proxy_sha256},
            "immutable_array_hashes": target.immutable_hashes,
            "baseline_metrics": baseline_metrics,
            "baseline_gate": gate_report(baseline_metrics, spec.gate),
            "candidate_count": len(candidates),
            "candidates": [serializable_candidate(value) for value in candidates],
            "selected_candidate": serializable_candidate(selected),
            "local_quality_passed": local_pass,
            "local_structural_passed": structural_pass,
            "patch": None if patch_path is None else {"path": str(patch_path), "sha256": patch_sha},
            "atlas_placement": {
                "status": "required_not_run",
                "coordinates_final": False,
                "keep_policy": "all non-target face UVs must remain bit-identical",
                "required_final_gates": [
                    "zero target-vs-KEEP and target-vs-target positive-area overlap",
                    "minimum 16px chart separation and page boundary margin",
                    "no stacking",
                    "page floors and resolution unchanged",
                    "page occupancy not below v8 baseline",
                ],
            },
        }
        target_reports.append(row)
        print(f"V9_SELECTED {spec.selector}: {selected['seam_mode']}/{selected['unwrap_method']} local_pass={local_pass}", flush=True)
    report = {
        "schema": SCHEMA + "/solver-report",
        "status": (
            "passed_local_smoke_atlas_pending"
            if overall_pass
            else "passed_structural_smoke_quality_pending_atlas_pending"
            if overall_structural
            else "failed_local_quality_atlas_pending"
        ),
        "operation": "headless_targeted_uv_overlay_candidates_only",
        "targets": target_reports,
        "target_count": len(target_reports),
        "blender_version": bpy.app.version_string,
        "blender_background": bpy.app.background,
        "blend_opened": False,
        "blend_saved": False,
        "texture_bake_performed": False,
        "fal_called": False,
        "github_called": False,
        "canonical_build_performed": False,
        "authorities_modified": False,
        "runtime_seconds": float(time.perf_counter() - started),
        "hard_gates": {
            "exact_shortlist_only": True,
            "at_most_two_smoke_targets": len(specs) <= 2,
            "all_local_quality_gates_passed": overall_pass,
            "all_local_structural_gates_passed": overall_structural,
            "geometry_topology_normals_output_omitted_and_immutable": True,
            "atlas_coordinates_not_misrepresented_as_final": True,
            "no_fal_github_texture_or_canonical_build": True,
        },
    }
    report_path = output / "targeted_reunwrap_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("V9_TARGETED_UNWRAP=" + json.dumps({"status": report["status"], "output": str(output), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
