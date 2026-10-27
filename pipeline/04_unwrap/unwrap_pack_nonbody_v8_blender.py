#!/usr/bin/env python3
"""Isolated Blender ABF unwrap and six-page semantic pack for non-BODY v8.

Run with Blender ``--background --factory-startup --python ... -- [args]``.
The builder reads only canonical v8 proxy NPZs and writes only below the v8
root.  It creates UV/proof sidecars and layout previews; it never bakes a
texture, calls Fal, or imports a production PBR/blend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from contract_v8 import (  # noqa: E402
    BODY_OBJECT_ID,
    HERO_8K_PAGE_IDS,
    PACK_MARGIN_PIXELS,
    PIECE_MIN_EFFECTIVE_TEXELS_OVERRIDES,
    ROOT,
    TARGET_PAGE_IDS,
    TARGET_PAGE_SPECS,
    UNWRAP_METHOD,
)
from proxy_core_v8 import require, sha256_file  # noqa: E402


REQUIRED_PROXY_GATES = (
    "all_same_direction_two_owner_edges_forced_solver_splits",
    "zero_solver_chart_winding_conflict_crossings",
    "representative_corner_normals_local_world_stored",
)
UV_EPSILON = 2.0e-7


def _load_validated_core():
    path = ROOT / "critical_preview/unwrap_critical_preview_blender.py"
    spec = importlib.util.spec_from_file_location("_v8_validated_critical_uv_core", path)
    require(spec is not None and spec.loader is not None, f"Cannot load validated UV core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_validated_core()


@dataclass
class Piece:
    object_id: int
    name: str
    page_id: str
    proxy_path: Path
    arrays: dict[str, np.ndarray]
    obj: bpy.types.Object
    solver_vertices: np.ndarray
    solver_faces: np.ndarray
    solver_vertex_source: np.ndarray
    uv: np.ndarray
    face_island: np.ndarray
    island_sizes: np.ndarray
    geometric_area: float
    density_weight: float
    report: dict[str, Any]


def blender_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def safe_output(path: Path) -> Path:
    value = path.expanduser().resolve()
    require(value == ROOT or ROOT in value.parents, f"Output escaped v8 root: {value}")
    return value


def load_ready_manifest(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("status") in {"passed", "passed_existing_artifacts_verified"}, "Proxy manifest is not passed")
    gates = report.get("hard_gates", {})
    missing = [gate for gate in REQUIRED_PROXY_GATES if gates.get(gate) is not True]
    require(not missing, f"Proxy manifest predates mandatory solver gates: {missing}")
    require(report["totals"]["same_direction_two_owner_solver_splits"] == 3014, "Winding split total drift")
    require(report["totals"]["proxy_faces"] == 139211, "Proxy face total drift")
    return report


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def geometric_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangle = np.asarray(vertices, dtype=np.float64)[faces]
    return float(
        (0.5 * np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)).sum()
    )


def split_for_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    requested_seams: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Validated corner-fan split plus an explicit no-leak half-edge gate.

    A cut can reconnect around a high-valence vertex even when its direct
    adjacency was excluded from the DSU.  In that rare case the second side's
    two face corners are duplicated explicitly.  This may add a safe seam next
    door, but can never lose a requested/unsafe split or reorder a face.
    """

    owners = core.edge_uses(faces)
    requested = {tuple(map(int, edge)) for edge in requested_seams.tolist()}
    require(not (requested - set(owners)), "Requested seam is absent from proxy")
    incidence_not_two = {edge for edge, uses in owners.items() if len(uses) != 2}
    same_direction = {
        edge for edge, uses in owners.items() if core.is_same_direction_conflict(uses)
    }
    split_edges = requested | incidence_not_two | same_direction
    dsu = core.DisjointSet(len(faces) * 3)
    for edge, uses in owners.items():
        if edge in split_edges or len(uses) != 2:
            continue
        left, right = uses
        for vertex in edge:
            fi, c0, c1, _, _ = left
            fj, d0, d1, _, _ = right
            ci = c0 if int(faces[fi, c0]) == vertex else c1
            cj = d0 if int(faces[fj, d0]) == vertex else d1
            dsu.union(fi * 3 + ci, fj * 3 + cj)
    roots = np.asarray([dsu.find(index) for index in range(len(faces) * 3)], dtype=np.int32)
    _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
    source_corner_vertices = faces.reshape(-1)
    solver_source = source_corner_vertices[first].astype(np.int32).tolist()
    solver_faces = inverse.reshape(-1, 3).astype(np.int32)

    forced_corner_duplicates = 0
    for _ in range(4):
        leaks = 0
        for edge in split_edges:
            seen: set[tuple[int, int]] = set()
            for face_index, c0, c1, _, _ in owners[edge]:
                solver_edge = tuple(sorted((int(solver_faces[face_index, c0]), int(solver_faces[face_index, c1]))))
                if solver_edge not in seen:
                    seen.add(solver_edge)
                    continue
                for corner in (c0, c1):
                    solver_source.append(int(faces[face_index, corner]))
                    solver_faces[face_index, corner] = len(solver_source) - 1
                    forced_corner_duplicates += 1
                leaks += 1
        if leaks == 0:
            break
    solver_source_array = np.asarray(solver_source, dtype=np.int32)
    solver_vertices = vertices[solver_source_array]
    source_key_to_solver: dict[tuple[int, int], list[int]] = {}
    for solver_edge, linked in core.edge_uses(solver_faces).items():
        source_edge = tuple(sorted((int(solver_source_array[solver_edge[0]]), int(solver_source_array[solver_edge[1]]))))
        source_key_to_solver.setdefault(source_edge, []).append(len(linked))
    leaked = [
        edge for edge in split_edges if any(owner_count > 1 for owner_count in source_key_to_solver.get(edge, []))
    ]
    require(not leaked, f"Explicit half-edge split still leaked: {leaked[:5]}")
    safe_lost = 0
    for edge, linked in owners.items():
        if len(linked) == 2 and edge not in split_edges:
            copies = source_key_to_solver.get(edge, [])
            safe_lost += int(len(copies) != 1 or copies[0] != 2)
    return solver_vertices, solver_faces, solver_source_array, {
        "source_unique_edges": len(owners),
        "source_boundary_edges": sum(len(uses) == 1 for uses in owners.values()),
        "source_incidence_not_two_edges": len(incidence_not_two),
        "source_nonmanifold_edges": sum(len(uses) > 2 for uses in owners.values()),
        "source_same_direction_conflict_edges": len(same_direction),
        "requested_hard_surface_seams": len(requested),
        "combined_solver_split_edges": len(split_edges),
        "mandatory_split_leaks": 0,
        "forced_corner_duplicates_for_no_leak": forced_corner_duplicates,
        "additional_safe_adjacencies_split": safe_lost,
        "solver_vertices": len(solver_vertices),
        "solver_faces": len(solver_faces),
    }


def uv_area(uv: np.ndarray) -> float:
    return float((0.5 * np.abs(core.signed_double_area(uv))).sum())


def orientation_allow_degenerate(
    uv: np.ndarray, face_island: np.ndarray
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    """Critical-core orientation evidence plus explicit proven zero-area faces."""

    signed = core.signed_double_area(uv)
    degenerate = set(map(int, np.flatnonzero(np.abs(signed) <= 1.0e-14)))
    mixed: dict[int, dict[str, Any]] = {}
    for island in np.unique(face_island):
        faces = np.flatnonzero(face_island == island).astype(np.int32)
        nonzero = faces[np.abs(signed[faces]) > 1.0e-14]
        positive = nonzero[signed[nonzero] > 0.0]
        negative = nonzero[signed[nonzero] < 0.0]
        if len(positive) and len(negative):
            minority = positive if len(positive) <= len(negative) else negative
            mixed[int(island)] = {
                "faces": faces,
                "positive_faces": positive,
                "negative_faces": negative,
                "minority_faces": minority,
            }
    return mixed, degenerate


def set_face_uv(mesh: bpy.types.Mesh, uv: np.ndarray) -> None:
    core.write_face_uv(mesh, uv)


def install_representative_loop_normals(
    obj: bpy.types.Object,
    normals_local: np.ndarray,
    normals_world: np.ndarray,
) -> dict[str, float]:
    expected = np.asarray(normals_local, dtype=np.float32).reshape(-1, 3)
    expected_world = np.asarray(normals_world, dtype=np.float32).reshape(-1, 3)
    require(len(expected) == len(obj.data.loops), f"{obj.name}: loop normal count drift")
    require(expected_world.shape == expected.shape, f"{obj.name}: world loop normal count drift")
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    # Blender stores a custom normal per smooth fan.  Marking every solver edge
    # sharp is intentional here: it makes each source corner its own fan, so
    # nearby but distinct representative loop normals are never averaged.
    for edge in obj.data.edges:
        edge.use_edge_sharp = True
    target = expected.astype(np.float64)
    target /= np.linalg.norm(target, axis=1, keepdims=True)

    def decode(requested: np.ndarray) -> np.ndarray:
        requested = requested / np.linalg.norm(requested, axis=1, keepdims=True)
        obj.data.normals_split_custom_set(requested.tolist())
        obj.data.update()
        return np.asarray([row.vector[:] for row in obj.data.corner_normals], dtype=np.float64)

    actual = decode(target)
    dot = np.einsum("ij,ij->i", actual, target)
    if float(dot.min()) < 0.999995:
        # Search across the compact-normal quantization cell.  Every edge is
        # sharp, so each loop is independent and may retain its own best
        # pre-compensated request from the deterministic scalar sweep.
        initial_error = target - actual
        best_request = target.copy()
        best_dot = dot.copy()
        for alpha in np.linspace(-6.0, 6.0, 97):
            candidate = target + float(alpha) * initial_error
            decoded = decode(candidate)
            candidate_dot = np.einsum("ij,ij->i", decoded, target)
            improved = candidate_dot > best_dot
            best_dot[improved] = candidate_dot[improved]
            best_request[improved] = candidate[improved]
        if float(best_dot.min()) < 0.999995:
            rng = np.random.default_rng(0xF3A8)
            axis = np.zeros_like(target)
            least = np.argmin(np.abs(target), axis=1)
            axis[np.arange(len(target)), least] = 1.0
            tangent_a = np.cross(target, axis)
            tangent_a /= np.linalg.norm(tangent_a, axis=1, keepdims=True)
            tangent_b = np.cross(target, tangent_a)
            for amplitude in (5e-4, 1e-3, 2e-3, 4e-3):
                for _ in range(32):
                    offsets = rng.uniform(-amplitude, amplitude, size=(len(target), 2))
                    candidate = target + offsets[:, :1] * tangent_a + offsets[:, 1:] * tangent_b
                    decoded = decode(candidate)
                    candidate_dot = np.einsum("ij,ij->i", decoded, target)
                    improved = candidate_dot > best_dot
                    best_dot[improved] = candidate_dot[improved]
                    best_request[improved] = candidate[improved]
        actual = decode(best_request)
        dot = np.einsum("ij,ij->i", actual, target)
    # Preserve the authoritative vectors losslessly as CORNER attributes.
    # Blender's display custom-normal codec is quantized, while these proof
    # attributes round-trip the exact proxy float32 values.
    for attribute_name, values in (
        ("v8_source_corner_normal_local", expected),
        ("v8_source_corner_normal_world", expected_world),
    ):
        attribute = obj.data.attributes.get(attribute_name) or obj.data.attributes.new(
            name=attribute_name, type="FLOAT_VECTOR", domain="CORNER"
        )
        attribute.data.foreach_set("vector", values.reshape(-1))
    obj.data.update()
    if float(dot.min()) < 0.999995:
        worst = np.argsort(dot)[:10]
        print(
            "V8_NORMAL_DIAGNOSTIC "
            + json.dumps(
                {
                    "object": obj.name,
                    "indices": worst.tolist(),
                    "dots": dot[worst].tolist(),
                    "expected": expected[worst].tolist(),
                    "actual": actual[worst].tolist(),
                }
            ),
            flush=True,
        )
    require(
        float(dot.min()) >= 0.999995,
        f"{obj.name}: custom loop normals drift before save; min_dot={float(dot.min()):.9f}, failures={int(np.count_nonzero(dot < 0.999999))}",
    )
    return {
        "decoded_custom_normal_minimum_dot": float(dot.min()),
        "exact_corner_attribute_minimum_dot": 1.0,
    }


def normalize_semantic_density(piece: Piece) -> None:
    current_uv_area = uv_area(piece.uv)
    require(current_uv_area > 1e-15 and piece.geometric_area > 1e-15, f"{piece.name}: zero density area")
    scale = math.sqrt(piece.geometric_area / current_uv_area) * piece.density_weight
    centre = piece.uv.reshape(-1, 2).mean(axis=0)
    piece.uv = (piece.uv - centre) * scale + centre
    set_face_uv(piece.obj.data, piece.uv)
    piece.report["semantic_density"] = {
        "geometric_area": piece.geometric_area,
        "pre_page_pack_uv_area": uv_area(piece.uv),
        "linear_density_weight": piece.density_weight,
        "normalization_scale": scale,
    }


def unwrap_one(
    row: dict,
    inventory_row: dict,
    checkpoint_dir: Path,
    require_checkpoint: bool = False,
) -> Piece:
    object_id = int(row["object_id"])
    path = Path(row["artifact"]["path"]).resolve()
    require(path.is_file() and sha256_file(path) == row["artifact"]["sha256"], f"Proxy artifact drift: {path}")
    arrays = read_npz(path)
    vertices = np.asarray(arrays["proxy_vertices_local"], dtype=np.float64)
    faces = np.asarray(arrays["proxy_faces"], dtype=np.int32)
    requested = np.asarray(arrays["proxy_seam_edges"], dtype=np.int32).reshape(-1, 2)
    solver_vertices, solver_faces, solver_source, topology = split_for_parameterization(
        vertices, faces, requested
    )
    name = str(arrays["object_name"])
    obj = core.build_mesh(f"v8_{object_id:02d}_{name}", solver_vertices, solver_faces)
    obj.matrix_world = Matrix(np.asarray(arrays["object_matrix_world"], dtype=float).tolist())
    obj["source_object_id"] = object_id
    obj["target_page_id"] = row["target_page_id"]
    obj["source_material_provenance_only"] = row["source_material"]
    obj["target_pbr"] = row["semantic_material"]["target_pbr"]
    obj["backface_culling"] = False
    normal_install = install_representative_loop_normals(
        obj,
        arrays["proxy_corner_normals_local"],
        arrays["proxy_corner_normals_world"],
    )

    checkpoint_path = checkpoint_dir / f"{object_id:02d}_{name}.npz"
    if checkpoint_path.is_file():
        checkpoint = read_npz(checkpoint_path)
        require(str(checkpoint["proxy_sha256"]) == row["artifact"]["sha256"], f"{name}: checkpoint proxy drift")
        require(np.array_equal(checkpoint["solver_faces"], solver_faces), f"{name}: checkpoint solver drift")
        uv = np.asarray(checkpoint["proxy_corner_uv"], dtype=np.float64)
        require(uv.shape == (len(faces), 3, 2), f"{name}: checkpoint UV shape drift")
        set_face_uv(obj.data, uv)
        face_island, sizes = core.uv_islands(solver_faces, uv)
        mixed, degenerate = orientation_allow_degenerate(uv, face_island)
        require(not mixed and not degenerate, f"{name}: checkpoint fold/degenerate drift")
        overlap = core.overlap_audit(uv, face_island)
        require(overlap["positive_area_triangle_overlap_pairs"] == 0, f"{name}: checkpoint overlap drift")
        report = json.loads(str(checkpoint["report_json"]))
        report["loaded_from_checkpoint"] = True
        report["checkpoint"] = str(checkpoint_path)
        report["checkpoint_sha256"] = sha256_file(checkpoint_path)
        return Piece(
            object_id=object_id,
            name=name,
            page_id=row["target_page_id"],
            proxy_path=path,
            arrays=arrays,
            obj=obj,
            solver_vertices=solver_vertices,
            solver_faces=solver_faces,
            solver_vertex_source=solver_source,
            uv=uv,
            face_island=face_island,
            island_sizes=sizes,
            geometric_area=geometric_area(np.asarray(arrays["proxy_vertices"], dtype=np.float64), faces),
            density_weight=float(inventory_row["semantic_allocation"]["linear_density_weight"]),
            report=report,
        )

    require(not require_checkpoint, f"{name}: required unwrap checkpoint missing: {checkpoint_path}")

    operator = core.unwrap_pack(obj)
    uv = core.face_uv_array(obj.data)
    face_island, sizes = core.uv_islands(solver_faces, uv)
    repairs: dict[str, Any] = {}
    initial_mixed, initial_degenerate = orientation_allow_degenerate(uv, face_island)
    repairs["initial_orientation"] = core.mixed_orientation_report(initial_mixed)
    repairs["initial_degenerate_uv_faces"] = sorted(initial_degenerate)
    if initial_mixed:
        repairs["corrective_abf"] = core.corrective_seam_reunwrap_mixed_islands(
            obj, solver_faces, face_island, initial_mixed
        )
        uv = core.face_uv_array(obj.data)
        face_island, sizes = core.uv_islands(solver_faces, uv)

    post_mixed, post_degenerate = orientation_allow_degenerate(uv, face_island)
    post_overlap = core.overlap_audit(uv, face_island, include_all_pairs=True)
    overlap_pairs = post_overlap.pop("_all_pairs")
    proven = {
        int(face)
        for evidence in post_mixed.values()
        for face in evidence["minority_faces"].tolist()
    }
    proven.update(initial_degenerate)
    proven.update(post_degenerate)
    proven.update(int(face) for pair in overlap_pairs for face in pair)
    repairs["post_corrective_abf_orientation"] = core.mixed_orientation_report(post_mixed)
    repairs["post_corrective_abf_overlap"] = post_overlap
    if proven:
        repairs["smart_project"] = core.smart_project_proven_faces(obj, proven)
        uv = core.face_uv_array(obj.data)
        face_island, sizes = core.uv_islands(solver_faces, uv)

    final_mixed, final_degenerate = orientation_allow_degenerate(uv, face_island)
    require(not final_mixed, f"{name}: local fold remains after bounded repair")
    require(not final_degenerate, f"{name}: local UV degenerates remain after bounded repair")
    uv, orientation = core.normalize_island_orientation(uv, face_island)
    set_face_uv(obj.data, uv)
    face_island, sizes = core.uv_islands(solver_faces, uv)
    residual_rounds = []
    for residual_round in range(2):
        overlap_full = core.overlap_audit(uv, face_island, include_all_pairs=True)
        residual_pairs = overlap_full.pop("_all_pairs")
        if not residual_pairs:
            break
        residual_faces = {int(face) for pair in residual_pairs for face in pair}
        repair = core.smart_project_proven_faces(obj, residual_faces)
        uv = core.face_uv_array(obj.data)
        face_island, sizes = core.uv_islands(solver_faces, uv)
        residual_mixed, residual_degenerate = orientation_allow_degenerate(uv, face_island)
        require(not residual_mixed and not residual_degenerate, f"{name}: residual Smart Project introduced fold")
        uv, residual_orientation = core.normalize_island_orientation(uv, face_island)
        set_face_uv(obj.data, uv)
        face_island, sizes = core.uv_islands(solver_faces, uv)
        residual_rounds.append(
            {
                "round": residual_round + 1,
                "proven_overlap_pairs": len(residual_pairs),
                "proven_faces": len(residual_faces),
                "repair": repair,
                "orientation": residual_orientation,
            }
        )
    overlap = core.overlap_audit(uv, face_island)
    require(overlap["positive_area_triangle_overlap_pairs"] == 0, f"{name}: local overlap remains")
    require(int(np.count_nonzero(core.signed_double_area(uv) <= 1e-14)) == 0, f"{name}: local flip/degenerate remains")
    component = core.component_island_report(arrays["proxy_component_id"], face_island)
    require(component["islands_crossing_source_components"] == 0, f"{name}: UV island crosses topology components")

    piece = Piece(
        object_id=object_id,
        name=name,
        page_id=row["target_page_id"],
        proxy_path=path,
        arrays=arrays,
        obj=obj,
        solver_vertices=solver_vertices,
        solver_faces=solver_faces,
        solver_vertex_source=solver_source,
        uv=uv,
        face_island=face_island,
        island_sizes=sizes,
        geometric_area=geometric_area(np.asarray(arrays["proxy_vertices"], dtype=np.float64), faces),
        density_weight=float(inventory_row["semantic_allocation"]["linear_density_weight"]),
        report={
            "object_id": object_id,
            "name": name,
            "page_id": row["target_page_id"],
            "proxy": str(path),
            "proxy_sha256": row["artifact"]["sha256"],
            "faces": int(len(faces)),
            "solver_vertices": int(len(solver_vertices)),
            "preunwrap_charts": int(len(arrays["proxy_preunwrap_chart_face_counts"])),
            "postrepair_charts": int(len(sizes)),
            "topology": topology,
            "operators": operator,
            "targeted_repair": repairs,
            "orientation": orientation,
            "component_islands": component,
            "loop_normal_install": normal_install,
            "local_overlap": overlap,
            "residual_overlap_repairs": residual_rounds,
        },
    )
    normalize_semantic_density(piece)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        schema=np.asarray("car-city.ferrari-nonbody-unwrap-checkpoint/v8"),
        proxy_sha256=np.asarray(row["artifact"]["sha256"]),
        solver_faces=piece.solver_faces,
        proxy_corner_uv=piece.uv.astype(np.float32),
        report_json=np.asarray(json.dumps(piece.report, sort_keys=True)),
    )
    return piece


def pack_page(pieces: list[Piece], resolution: int) -> dict[str, Any]:
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    for piece in pieces:
        piece.obj.select_set(True)
    bpy.context.view_layer.objects.active = pieces[0].obj
    previous_sync = bpy.context.scene.tool_settings.use_uv_select_sync
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    started = time.perf_counter()
    result = sorted(
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
    seconds = time.perf_counter() - started
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.scene.tool_settings.use_uv_select_sync = previous_sync
    chart_offset = 0
    combined_uv = []
    combined_faces = []
    combined_chart = []
    vertex_offset = 0
    for piece in pieces:
        piece.uv = core.face_uv_array(piece.obj.data)
        piece.face_island, piece.island_sizes = core.uv_islands(piece.solver_faces, piece.uv)
        mixed, degenerate = orientation_allow_degenerate(piece.uv, piece.face_island)
        require(not mixed and not degenerate, f"{piece.name}: page pack introduced fold/degenerate")
        combined_uv.append(piece.uv)
        combined_faces.append(piece.solver_faces + vertex_offset)
        combined_chart.append(piece.face_island + chart_offset)
        vertex_offset += len(piece.solver_vertices)
        chart_offset += len(piece.island_sizes)
    page_uv = np.concatenate(combined_uv)
    page_faces = np.concatenate(combined_faces)
    page_chart = np.concatenate(combined_chart)
    overlap = core.overlap_audit(page_uv, page_chart)
    require(overlap["positive_area_triangle_overlap_pairs"] == 0, f"{pieces[0].page_id}: page overlap remains")
    uv_min = page_uv.reshape(-1, 2).min(axis=0)
    uv_max = page_uv.reshape(-1, 2).max(axis=0)
    require(float(uv_min.min()) >= -1e-7 and float(uv_max.max()) <= 1.0 + 1e-7, f"{pieces[0].page_id}: page escaped 0..1")
    return {
        "operator": result,
        "seconds": seconds,
        "resolution": resolution,
        "charts": chart_offset,
        "occupancy_fraction": uv_area(page_uv),
        "uv_bounds_min": uv_min.tolist(),
        "uv_bounds_max": uv_max.tolist(),
        "minimum_boundary_margin_pixels": float(min(uv_min.min(), (1.0 - uv_max).min()) * resolution),
        "overlap": overlap,
        "page_uv": page_uv,
        "page_faces": page_faces,
        "page_chart": page_chart,
    }


def piece_floor_audit(pieces: list[Piece], resolution: int, inventory_by_id: dict[int, dict]) -> dict[str, Any]:
    rows = []
    failures = []
    for piece in pieces:
        area = 0.5 * np.abs(core.signed_double_area(piece.uv))
        component = piece.arrays["proxy_piece_component_id"]
        allocations = {
            int(row["piece_id"][1]): row for row in inventory_by_id[piece.object_id]["piece_allocations"]
        }
        for component_id in sorted(np.unique(component).tolist()):
            mask = component == component_id
            effective = float(area[mask].sum() * resolution**2)
            floor = int(allocations[int(component_id)]["minimum_effective_texels"])
            row = {
                "piece_id": [piece.object_id, int(component_id)],
                "component_role": allocations[int(component_id)]["component_role"],
                "effective_triangle_texels": effective,
                "minimum_effective_texels": floor,
                "passed": effective >= floor,
            }
            rows.append(row)
            if not row["passed"]:
                failures.append(row)
    return {"pieces": rows, "failures": failures, "passed": not failures}


def boost_failed_piece_groups(
    pieces: list[Piece], floor_audit: dict[str, Any], inventory_by_id: dict[int, dict]
) -> dict[str, float]:
    """Increase only failed semantic pieces, sharing factors across symmetry families."""

    requested: dict[str, float] = {}
    for failure in floor_audit["failures"]:
        object_id, component_id = map(int, failure["piece_id"])
        if component_id == -1:
            group = "family:" + str(inventory_by_id[object_id]["semantic_allocation"]["family"])
        else:
            group = f"piece:{object_id}:{component_id}"
        factor = math.sqrt(
            float(failure["minimum_effective_texels"])
            / max(float(failure["effective_triangle_texels"]), 1e-12)
        ) * 1.05
        requested[group] = max(requested.get(group, 1.0), factor)
    applied: dict[str, float] = {}
    for piece in pieces:
        family_group = "family:" + str(inventory_by_id[piece.object_id]["semantic_allocation"]["family"])
        component_values = np.asarray(piece.arrays["proxy_piece_component_id"], dtype=np.int16)
        for component_id in np.unique(component_values).tolist():
            group = family_group if int(component_id) == -1 else f"piece:{piece.object_id}:{int(component_id)}"
            factor = requested.get(group)
            if factor is None:
                continue
            mask = component_values == component_id
            centre = piece.uv[mask].reshape(-1, 2).mean(axis=0)
            piece.uv[mask] = (piece.uv[mask] - centre) * factor + centre
            set_face_uv(piece.obj.data, piece.uv)
            applied[group] = factor
    require(applied, "Floor boost requested but no semantic piece matched")
    return applied


def assign_source_transfer(piece: Piece, resolution: int, degenerate_start: int, chart_offset: int) -> tuple[dict[str, np.ndarray], int]:
    arrays = piece.arrays
    source_count = len(arrays["original_to_proxy_face"])
    source_uv = np.full((source_count, 3, 2), np.nan, dtype=np.float32)
    source_chart = np.full(source_count, -1, dtype=np.int32)
    mapping = arrays["original_to_proxy_face"]
    corner = arrays["original_corner_to_proxy_corner"]
    renderable = np.flatnonzero(mapping >= 0)
    mapped_uv = piece.uv[mapping[renderable, None], corner[renderable]]
    source_uv[renderable] = mapped_uv
    source_chart[renderable] = piece.face_island[mapping[renderable]] + chart_offset
    degenerate = np.flatnonzero(mapping < 0)
    for local_index, source_row in enumerate(degenerate.tolist()):
        slot = degenerate_start + local_index
        x = (24.0 + 3.0 * slot) / resolution
        y = 1.0 - 5.0 / resolution
        source_uv[source_row] = np.asarray(
            ((x, y), (x + 0.65 / resolution, y), (x, y + 0.65 / resolution)),
            dtype=np.float32,
        )
        # ``chart_offset`` already includes every preceding object's proxy
        # charts and degenerate placeholders.  ``slot`` is global only for the
        # private gutter-strip X position; adding it here would double-count
        # preceding degenerates and can collide with a later object's charts.
        source_chart[source_row] = chart_offset + len(piece.island_sizes) + local_index
    reconstructed = source_uv[renderable]
    expected = mapped_uv
    require(np.array_equal(reconstructed, expected.astype(np.float32)), f"{piece.name}: exact UV transfer drift")
    return {
        "source_face_uv": source_uv,
        "source_face_chart": source_chart,
        "source_face_indices_global": arrays["original_face_indices_global"],
        "source_faces_global": arrays["original_global_faces"],
        "source_geometry_degenerate": arrays["original_geometry_degenerate"],
        "source_to_proxy_face": mapping,
        "source_corner_to_proxy_corner": corner,
    }, len(degenerate)


def save_outputs(output: Path, page_id: str, pieces: list[Piece], resolution: int, page_report: dict, inventory_by_id: dict[int, dict], preview_size: int) -> dict[str, Any]:
    sidecar_dir = output / "sidecars"
    layout_dir = output / "layouts"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)
    chart_offset = 0
    degenerate_slot = 0
    object_files = []
    for piece in pieces:
        transfer, degenerate_count = assign_source_transfer(piece, resolution, degenerate_slot, chart_offset)
        degenerate_slot += degenerate_count
        path = sidecar_dir / f"{piece.object_id:02d}_{piece.name}.npz"
        np.savez_compressed(
            path,
            schema=np.asarray("car-city.ferrari-nonbody-packed-object/v8"),
            source_object_id=np.asarray(piece.object_id, dtype=np.int16),
            object_name=np.asarray(piece.name),
            target_page_id=np.asarray(page_id),
            page_resolution=np.asarray(resolution, dtype=np.int32),
            object_matrix_world=piece.arrays["object_matrix_world"],
            proxy_vertices_local=piece.arrays["proxy_vertices_local"],
            proxy_vertices_world=piece.arrays["proxy_vertices"],
            proxy_faces=piece.arrays["proxy_faces"],
            proxy_corner_uv=piece.uv.astype(np.float32),
            proxy_chart_id=(piece.face_island + chart_offset).astype(np.int32),
            proxy_component_id=piece.arrays["proxy_component_id"],
            proxy_piece_component_id=piece.arrays["proxy_piece_component_id"],
            proxy_component_role=piece.arrays["proxy_component_role"],
            proxy_piece_id=piece.arrays["proxy_piece_id"],
            proxy_corner_normals_local=piece.arrays["proxy_corner_normals_local"],
            proxy_corner_normals_world=piece.arrays["proxy_corner_normals_world"],
            **transfer,
        )
        object_files.append({"path": str(path), "sha256": sha256_file(path)})
        chart_offset += len(piece.island_sizes) + degenerate_count

    core.HERE = output
    core.LAYOUT_SIZE = preview_size
    layout = layout_dir / f"{page_id}_layout_{preview_size}.png"
    core.write_layout_png(layout, page_report["page_faces"], page_report["page_uv"], page_report["page_chart"])
    floor = piece_floor_audit(pieces, resolution, inventory_by_id)
    return {
        "page_id": page_id,
        "resolution": [resolution, resolution],
        "objects": [piece.object_id for piece in pieces],
        "render_piece_floor_audit": floor,
        "object_sidecars": object_files,
        "layout": str(layout),
        "layout_sha256": sha256_file(layout),
        "charts_including_degenerate_placeholders": chart_offset,
        "degenerate_placeholders": degenerate_slot,
        **{key: value for key, value in page_report.items() if key not in {"page_uv", "page_faces", "page_chart"}},
    }


def reopen_normal_audit(blend_path: Path, expected: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    exact_minima = {}
    decoded_minima = {}
    for name, targets in expected.items():
        obj = bpy.data.objects.get(name)
        require(obj is not None, f"Reopened blend lost object: {name}")
        local_target = np.asarray(targets["local"], dtype=np.float64).reshape(-1, 3)
        local_target /= np.linalg.norm(local_target, axis=1, keepdims=True)
        actual = np.asarray([value.vector[:] for value in obj.data.corner_normals], dtype=np.float64)
        require(actual.shape == local_target.shape, f"{name}: reopened normal shape drift")
        decoded_minima[name] = float(np.einsum("ij,ij->i", actual, local_target).min())
        require(decoded_minima[name] >= 0.999995, f"{name}: reopened decoded custom normals drift")
        for space in ("local", "world"):
            target = np.asarray(targets[space], dtype=np.float32).reshape(-1, 3)
            attribute = obj.data.attributes.get(f"v8_source_corner_normal_{space}")
            require(attribute is not None and attribute.domain == "CORNER", f"{name}: exact {space} normal attribute missing")
            stored = np.empty(target.size, dtype=np.float32)
            attribute.data.foreach_get("vector", stored)
            stored = stored.reshape(-1, 3)
            target64 = target.astype(np.float64)
            stored64 = stored.astype(np.float64)
            target64 /= np.linalg.norm(target64, axis=1, keepdims=True)
            stored64 /= np.linalg.norm(stored64, axis=1, keepdims=True)
            dot = float(np.einsum("ij,ij->i", target64, stored64).min())
            exact_minima[f"{name}:{space}"] = dot
            require(dot >= 0.999999, f"{name}: reopened exact {space} corner normal drift")
    return {
        "exact_attribute_minimum_dot_by_object_space": exact_minima,
        "exact_attribute_global_minimum_dot": min(exact_minima.values()),
        "decoded_custom_minimum_dot_by_object": decoded_minima,
        "decoded_custom_global_minimum_dot": min(decoded_minima.values()),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-manifest", type=Path, default=ROOT / "artifacts/proxy_manifest.json")
    parser.add_argument("--inventory", type=Path, default=ROOT / "inventory/nonbody_inventory_v8.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/unwrap_pack")
    parser.add_argument("--pages", default=",".join(TARGET_PAGE_IDS))
    parser.add_argument("--object-ids", help="Optional comma-separated diagnostic subset; pages still constrain membership")
    parser.add_argument("--smoke", action="store_true", help="Use only one source parent per selected page")
    parser.add_argument("--preview-size", type=int, default=2048)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Read/write unwrap checkpoints here; defaults to OUTPUT/checkpoints",
    )
    parser.add_argument(
        "--repack-from-checkpoints",
        action="store_true",
        help="Require every selected object checkpoint and perform no unwrap",
    )
    parser.add_argument("--overwrite-report", action="store_true")
    args = parser.parse_args(blender_args())
    started = time.perf_counter()
    require(bpy.app.background, "Run with Blender --background")
    require(UNWRAP_METHOD == "ANGLE_BASED", "V8 no longer requests Angle Based unwrap")
    pages = tuple(value.strip() for value in args.pages.split(",") if value.strip())
    require(pages and not (set(pages) - set(TARGET_PAGE_IDS)), f"Unknown pages: {pages}")
    output = safe_output(args.output_dir)
    if output.exists():
        require(args.overwrite_report or not any(output.iterdir()), f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_ready_manifest(args.proxy_manifest.resolve())
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    require(inventory["status"] == "passed", "Inventory is not passed")
    inventory_by_id = {int(row["object_id"]): row for row in inventory["objects"]}
    manifest_by_id = {int(row["object_id"]): row for row in manifest["objects"]}
    core.clear_scene()

    if args.object_ids:
        selected_ids = [int(value) for value in args.object_ids.split(",") if value.strip()]
        allowed = {object_id for page_id in pages for object_id in TARGET_PAGE_SPECS[page_id].object_ids}
        require(selected_ids and not (set(selected_ids) - allowed), "Diagnostic object IDs fall outside selected pages")
    else:
        selected_ids = []
        for page_id in pages:
            ids = list(TARGET_PAGE_SPECS[page_id].object_ids)
            selected_ids.extend(ids[:1] if args.smoke else ids)
    pieces = []
    checkpoint_dir = (
        safe_output(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else output / "checkpoints"
    )
    for object_id in selected_ids:
        piece = unwrap_one(
            manifest_by_id[object_id],
            inventory_by_id[object_id],
            checkpoint_dir,
            require_checkpoint=args.repack_from_checkpoints,
        )
        pieces.append(piece)
        stage = "CHECKPOINT" if piece.report.get("loaded_from_checkpoint") else "ABF"
        print(f"V8_{stage} {object_id:02d} {piece.name}: {len(piece.solver_faces)} faces, {len(piece.island_sizes)} charts", flush=True)

    page_outputs = []
    for page_id in pages:
        page_pieces = [piece for piece in pieces if piece.page_id == page_id]
        if not page_pieces:
            continue
        resolution = 2048 if args.smoke else TARGET_PAGE_SPECS[page_id].base_size
        floor_iterations = []
        for attempt in range(5):
            packed = pack_page(page_pieces, resolution)
            floor = piece_floor_audit(page_pieces, resolution, inventory_by_id)
            floor_iterations.append(
                {
                    "attempt": attempt + 1,
                    "resolution": resolution,
                    "failures": floor["failures"],
                }
            )
            if args.smoke or floor["passed"]:
                break
            applied = boost_failed_piece_groups(page_pieces, floor, inventory_by_id)
            floor_iterations[-1]["applied_linear_boosts"] = applied
        require(args.smoke or floor["passed"], f"{page_id}: semantic piece floors failed")
        page_output = save_outputs(output, page_id, page_pieces, resolution, packed, inventory_by_id, args.preview_size)
        page_output["resolution_policy"] = (
            "fixed_8k_hero" if page_id in HERO_8K_PAGE_IDS else "fixed_4k"
        )
        page_output["floor_iterations"] = floor_iterations
        page_outputs.append(page_output)
        print(f"V8_PAGE {page_id}: {packed['charts']} charts, {packed['occupancy_fraction']:.4f} occupancy", flush=True)

    blend = output / ("v8_smoke_uv_proxy.blend" if args.smoke else "v8_nonbody_uv_proxy.blend")
    expected_normals = {
        piece.obj.name: {
            "local": piece.arrays["proxy_corner_normals_local"].copy(),
            "world": piece.arrays["proxy_corner_normals_world"].copy(),
        }
        for piece in pieces
    }
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    blend_sha = sha256_file(blend)
    normal_reopen = reopen_normal_audit(blend, expected_normals)
    report = {
        "schema": "car-city.ferrari-nonbody-unwrap-pack/v8",
        "status": "passed_smoke" if args.smoke else "passed",
        "operation": (
            "isolated_checkpoint_only_hero_8k_semantic_six_page_repack"
            if args.repack_from_checkpoints
            else "isolated_angle_based_targeted_repair_semantic_six_page_pack"
        ),
        "blender_version": bpy.app.version_string,
        "pages": page_outputs,
        "objects": [piece.report for piece in pieces],
        "blend": str(blend),
        "blend_sha256": blend_sha,
        "reopened_corner_normal_audit": normal_reopen,
        "source_parent_objects": len(pieces),
        "proxy_faces": sum(len(piece.arrays["proxy_faces"]) for piece in pieces),
        "fal_called": False,
        "texture_bake_performed": False,
        "production_asset_modified": False,
        "runtime_seconds": float(time.perf_counter() - started),
        "checkpoint_source": str(checkpoint_dir),
        "unwrap_performed_this_run": not args.repack_from_checkpoints,
        "hard_gates": {
            "body_excluded": BODY_OBJECT_ID not in selected_ids,
            "all_local_folds_repaired_bounded": True,
            "all_positive_area_overlap_zero": True,
            "all_representative_loop_normals_survive_reopen": normal_reopen["passed"],
            "no_texture_or_fal": True,
            "exact_fixed_page_resolution_policy": all(
                int(page["resolution"]) == int(TARGET_PAGE_SPECS[page["page_id"]].base_size)
                for page in page_outputs
            ),
            "checkpoint_only_repack_honored": (
                not args.repack_from_checkpoints
                or all(piece.report.get("loaded_from_checkpoint") is True for piece in pieces)
            ),
        },
    }
    report_path = output / "unwrap_pack_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("V8_UNWRAP_PACK=" + json.dumps({"status": report["status"], "output": str(output), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
