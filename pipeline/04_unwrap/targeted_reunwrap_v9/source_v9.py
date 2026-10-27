#!/usr/bin/env python3
"""Read-only source adapters for BODY and v8 non-BODY targeted pieces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract_v9 import (
    BODY_PROXY,
    BODY_PROXY_SHA256,
    TargetSpec,
    V8_HERO_REPORT,
    V8_PROXY_MANIFEST,
    require,
    sha256_file,
    verify_authorities,
)
from metrics_v9 import edge_uses, face_components, hash_array


@dataclass
class TargetArrays:
    spec: TargetSpec
    object_name: str
    source_path: Path
    source_sha256: str
    proxy_path: Path
    proxy_sha256: str
    vertices: np.ndarray
    faces: np.ndarray
    baseline_uv: np.ndarray
    corner_normals: np.ndarray
    selected_face_indices: np.ndarray
    selected_vertex_indices: np.ndarray
    compact_vertices: np.ndarray
    compact_faces: np.ndarray
    compact_baseline_uv: np.ndarray
    compact_corner_normals: np.ndarray
    v8_seams: np.ndarray
    baseline_seams: np.ndarray
    topology_component_id: np.ndarray
    face_smooth: np.ndarray
    immutable_hashes: dict[str, str]


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _load_indices() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    hero = json.loads(V8_HERO_REPORT.read_text(encoding="utf-8"))
    require(hero.get("status") == "passed", "V8 hero unwrap authority is not passed")
    sidecars: dict[int, dict[str, Any]] = {}
    for page in hero["pages"]:
        for entry in page["object_sidecars"]:
            path = Path(entry["path"]).resolve()
            with np.load(path, allow_pickle=False) as payload:
                object_id = int(payload["source_object_id"])
            sidecars[object_id] = {
                "path": path,
                "sha256": entry["sha256"],
                "page_id": page["page_id"],
                "resolution": int(page["resolution"]),
            }
    proxy_manifest = json.loads(V8_PROXY_MANIFEST.read_text(encoding="utf-8"))
    require(proxy_manifest.get("status") in {"passed", "passed_existing_artifacts_verified"}, "V8 proxy manifest is not passed")
    proxies = {int(row["object_id"]): row for row in proxy_manifest["objects"]}
    return sidecars, proxies


def cut_edges_from_uv(faces: np.ndarray, uv: np.ndarray, tolerance: float = 2.0e-7) -> np.ndarray:
    result: list[tuple[int, int]] = []
    for edge, linked in edge_uses(faces).items():
        if len(linked) != 2:
            continue
        endpoint_rows = []
        for face_index, c0, c1 in linked:
            mapping = {
                int(faces[face_index, c0]): uv[face_index, c0],
                int(faces[face_index, c1]): uv[face_index, c1],
            }
            endpoint_rows.append((mapping[edge[0]], mapping[edge[1]]))
        left, right = endpoint_rows
        if max(float(np.max(np.abs(left[0] - right[0]))), float(np.max(np.abs(left[1] - right[1])))) > tolerance:
            result.append(edge)
    return np.asarray(result, dtype=np.int32).reshape(-1, 2)


def _compact(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    normals: np.ndarray,
    selected: np.ndarray,
    raw_v8_seams: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen_faces = np.asarray(faces[selected], dtype=np.int32)
    selected_vertices, inverse = np.unique(chosen_faces.reshape(-1), return_inverse=True)
    selected_vertices = selected_vertices.astype(np.int32)
    compact_faces = inverse.reshape(-1, 3).astype(np.int32)
    old_to_new = np.full(len(vertices), -1, dtype=np.int32)
    old_to_new[selected_vertices] = np.arange(len(selected_vertices), dtype=np.int32)
    raw_v8_seams = np.asarray(raw_v8_seams, dtype=np.int32).reshape(-1, 2)
    if len(raw_v8_seams):
        mask = np.all(old_to_new[raw_v8_seams] >= 0, axis=1)
        v8_seams = np.sort(old_to_new[raw_v8_seams[mask]], axis=1).astype(np.int32)
        v8_seams = np.unique(v8_seams, axis=0)
    else:
        v8_seams = np.empty((0, 2), dtype=np.int32)
    compact_uv = np.asarray(uv[selected], dtype=np.float64)
    return (
        selected_vertices,
        np.asarray(vertices[selected_vertices], dtype=np.float64),
        compact_faces,
        compact_uv,
        np.asarray(normals[selected], dtype=np.float32),
        v8_seams,
        cut_edges_from_uv(compact_faces, compact_uv),
    )


def load_target(spec: TargetSpec) -> TargetArrays:
    verify_authorities()
    if spec.source_kind == "body_proxy":
        arrays = read_npz(BODY_PROXY)
        vertices = np.asarray(arrays["render_vertices"], dtype=np.float32)
        faces = np.asarray(arrays["render_faces"], dtype=np.int32)
        uv = np.asarray(arrays["render_corner_uv"], dtype=np.float32)
        normals = np.asarray(arrays["render_corner_normals"], dtype=np.float32)
        components, sizes = face_components(faces)
        require(len(sizes) == 24 and int(sizes[17]) == 3525 and int(sizes[18]) == 3525, "BODY component contract drift")
        selected = np.flatnonzero(components == spec.component_id).astype(np.int32)
        raw_seams = np.empty((0, 2), dtype=np.int32)
        source_path = proxy_path = BODY_PROXY
        source_sha = proxy_sha = BODY_PROXY_SHA256
        object_name = "body"
        smooth = np.asarray(arrays["render_face_smooth"], dtype=np.bool_)
        topology = components
    else:
        sidecars, proxies = _load_indices()
        require(spec.object_id in sidecars and spec.object_id in proxies, f"Missing v8 authority for object {spec.object_id}")
        sidecar_record = sidecars[spec.object_id]
        require(sidecar_record["page_id"] == spec.page_id, f"Page drift for {spec.selector}")
        require(sidecar_record["resolution"] == spec.resolution, f"Resolution drift for {spec.selector}")
        source_path = sidecar_record["path"]
        source_sha = sidecar_record["sha256"]
        require(source_path.is_file() and sha256_file(source_path) == source_sha, f"Sidecar hash drift: {source_path}")
        proxy_record = proxies[spec.object_id]
        proxy_path = Path(proxy_record["artifact"]["path"]).resolve()
        proxy_sha = proxy_record["artifact"]["sha256"]
        require(proxy_path.is_file() and sha256_file(proxy_path) == proxy_sha, f"Proxy hash drift: {proxy_path}")
        arrays = read_npz(source_path)
        proxy = read_npz(proxy_path)
        vertices = np.asarray(arrays["proxy_vertices_local"], dtype=np.float32)
        faces = np.asarray(arrays["proxy_faces"], dtype=np.int32)
        uv = np.asarray(arrays["proxy_corner_uv"], dtype=np.float32)
        normals = np.asarray(arrays["proxy_corner_normals_local"], dtype=np.float32)
        topology = np.asarray(arrays["proxy_component_id"], dtype=np.int32)
        require(np.array_equal(faces, proxy["proxy_faces"]), f"Proxy face drift: {spec.selector}")
        require(np.array_equal(vertices, proxy["proxy_vertices_local"]), f"Proxy vertex drift: {spec.selector}")
        require(np.array_equal(normals, proxy["proxy_corner_normals_local"]), f"Proxy normal drift: {spec.selector}")
        if spec.component_id == -1:
            selected = np.arange(len(faces), dtype=np.int32)
        else:
            piece_component = np.asarray(arrays["proxy_piece_component_id"], dtype=np.int32)
            selected = np.flatnonzero(piece_component == spec.component_id).astype(np.int32)
        raw_seams = np.asarray(proxy["proxy_seam_edges"], dtype=np.int32).reshape(-1, 2)
        object_name = str(np.asarray(arrays["object_name"]).reshape(-1)[0])
        smooth = np.ones(len(faces), dtype=np.bool_)

    require(len(selected) > 0, f"Empty target selection: {spec.selector}")
    compacted = _compact(vertices, faces, uv, normals, selected, raw_seams)
    selected_vertices, compact_vertices, compact_faces, compact_uv, compact_normals, v8_seams, baseline_seams = compacted
    immutable_hashes = {
        "source_vertices": hash_array(vertices),
        "source_faces": hash_array(faces),
        "source_corner_normals": hash_array(normals),
        "selected_face_indices": hash_array(selected),
        "selected_vertices": hash_array(compact_vertices),
        "selected_faces": hash_array(compact_faces),
        "selected_corner_normals": hash_array(compact_normals),
        "baseline_selected_uv": hash_array(compact_uv.astype(np.float32)),
    }
    return TargetArrays(
        spec=spec,
        object_name=object_name,
        source_path=source_path,
        source_sha256=source_sha,
        proxy_path=proxy_path,
        proxy_sha256=proxy_sha,
        vertices=vertices,
        faces=faces,
        baseline_uv=uv,
        corner_normals=normals,
        selected_face_indices=selected,
        selected_vertex_indices=selected_vertices,
        compact_vertices=compact_vertices,
        compact_faces=compact_faces,
        compact_baseline_uv=compact_uv,
        compact_corner_normals=compact_normals,
        v8_seams=v8_seams,
        baseline_seams=baseline_seams,
        topology_component_id=topology,
        face_smooth=smooth,
        immutable_hashes=immutable_hashes,
    )


def orient_faces_solver_only(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Two-colour winding constraints without editing authority geometry.

    ``corner_order`` maps solver corners to compact/source corners.  UV output
    must be mapped back with the exact inverse assignment before it is saved.
    Contradictory/non-orientable edges remain explicit solver splits.
    """

    faces = np.asarray(faces, dtype=np.int32)
    owners = edge_uses(faces)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(faces))]
    initial_same = 0
    for edge, linked in owners.items():
        if len(linked) != 2:
            continue
        first, second = linked
        left_direction = (int(faces[first[0], first[1]]), int(faces[first[0], first[2]]))
        right_direction = (int(faces[second[0], second[1]]), int(faces[second[0], second[2]]))
        same = int(left_direction == right_direction)
        initial_same += same
        adjacency[first[0]].append((second[0], same))
        adjacency[second[0]].append((first[0], same))
    flips = np.full(len(faces), -1, dtype=np.int8)
    contradictions = 0
    for seed in range(len(faces)):
        if flips[seed] >= 0:
            continue
        flips[seed] = 0
        stack = [seed]
        while stack:
            face = stack.pop()
            for other, parity in adjacency[face]:
                expected = int(flips[face]) ^ parity
                if flips[other] < 0:
                    flips[other] = expected
                    stack.append(other)
                elif int(flips[other]) != expected:
                    contradictions += 1
    corner_order = np.tile(np.arange(3, dtype=np.int8), (len(faces), 1))
    corner_order[flips == 1] = np.asarray((0, 2, 1), dtype=np.int8)
    rows = np.arange(len(faces), dtype=np.int32)[:, None]
    oriented = faces[rows, corner_order]
    unresolved: list[tuple[int, int]] = []
    for edge, linked in edge_uses(oriented).items():
        if len(linked) != 2:
            continue
        first, second = linked
        left = (int(oriented[first[0], first[1]]), int(oriented[first[0], first[2]]))
        right = (int(oriented[second[0], second[1]]), int(oriented[second[0], second[2]]))
        if left == right:
            unresolved.append(edge)
    return oriented, corner_order, {
        "solver_only_face_reorientations": int(np.count_nonzero(flips)),
        "same_direction_two_owner_edges_before": initial_same,
        "same_direction_two_owner_edges_after": len(unresolved),
        "orientation_constraint_contradictions": contradictions // 2,
        "unresolved_same_direction_edges": [list(edge) for edge in unresolved[:40]],
        "canonical_geometry_edited": False,
        "canonical_topology_edited": False,
        "canonical_normals_edited": False,
    }
