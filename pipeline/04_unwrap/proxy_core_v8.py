#!/usr/bin/env python3
"""Exact per-object weld/dedup and transfer primitives for non-BODY v8."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract_v8 import (
    UVGAMI_MIN_PROXY_FACES,
    piece_component_id,
    piece_component_role,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def csr_from_group(
    group: np.ndarray, item: np.ndarray, group_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(group, kind="stable")
    counts = np.bincount(group, minlength=group_count)
    offsets = np.empty(group_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, item[order], order


def face_edge_owners(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            owners[(u, v) if u < v else (v, u)].append(face_index)
    return owners


def same_direction_two_owner_edges(faces: np.ndarray) -> np.ndarray:
    """Return manifold edges whose two oriented face traversals agree.

    Such an edge cannot be traversed by one orientable solver chart.  It is a
    mandatory UV split independently of the feature-seam policy.
    """

    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    directions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            owners[edge].append(face_index)
            directions[edge].append(1 if (u, v) == edge else -1)
    conflicts = [
        edge
        for edge, linked in owners.items()
        if len(linked) == 2 and directions[edge][0] == directions[edge][1]
    ]
    return np.asarray(sorted(conflicts), dtype=np.int32).reshape(-1, 2)


def face_components(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(len(faces), dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[left] = right

    for linked in face_edge_owners(faces).values():
        for other in linked[1:]:
            union(linked[0], other)
    roots = np.asarray([find(index) for index in range(len(faces))], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    sizes = np.bincount(labels).astype(np.int32)
    return labels.astype(np.int32), sizes


def chart_labels(faces: np.ndarray, seams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    owners = face_edge_owners(faces)
    seam_set = {tuple(map(int, edge)) for edge in seams.tolist()}
    require(not (seam_set - set(owners)), "Seam set contains a non-proxy edge")
    parent = np.arange(len(faces), dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    for edge, linked in owners.items():
        if len(linked) != 2 or edge in seam_set:
            continue
        left, right = find(linked[0]), find(linked[1])
        if left != right:
            parent[left] = right
    roots = np.asarray([find(index) for index in range(len(faces))], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), np.bincount(labels).astype(np.int32)


@dataclass
class ProxyResult:
    arrays: dict[str, np.ndarray]
    report: dict[str, Any]


def build_object_proxy(geometry: dict[str, np.ndarray], object_id: int) -> ProxyResult:
    names = [str(value) for value in geometry["object_names"]]
    name = names[object_id]
    vertex_start = int(geometry["object_vertex_start"][object_id])
    vertex_count = int(geometry["object_vertex_count"][object_id])
    face_start = int(geometry["object_face_start"][object_id])
    face_count = int(geometry["object_face_count"][object_id])
    original_face_indices = np.arange(
        face_start, face_start + face_count, dtype=np.int32
    )
    original_global_faces = np.asarray(
        geometry["faces"][original_face_indices], dtype=np.int32
    )
    require(
        int(original_global_faces.min()) >= vertex_start
        and int(original_global_faces.max()) < vertex_start + vertex_count,
        f"Object face references a vertex outside its source block: {name}",
    )
    all_object_vertices = np.asarray(
        geometry["vertices"][vertex_start : vertex_start + vertex_count],
        dtype=np.float32,
    )
    all_welded_vertices, all_source_to_welded = np.unique(
        all_object_vertices, axis=0, return_inverse=True
    )
    all_welded_faces = all_source_to_welded[
        original_global_faces - vertex_start
    ]
    all_geometric_unique_faces = np.unique(
        np.sort(all_welded_faces, axis=1), axis=0
    )
    degenerate = np.asarray(
        geometry["geometry_degenerate_face"][original_face_indices], dtype=bool
    )
    renderable_rows = np.flatnonzero(~degenerate).astype(np.int32)
    renderable_global_faces = original_global_faces[renderable_rows]
    source_vertex_indices, local_flat = np.unique(
        renderable_global_faces.reshape(-1), return_inverse=True
    )
    source_vertex_indices = source_vertex_indices.astype(np.int32)
    source_local_faces = local_flat.reshape(-1, 3).astype(np.int32)
    source_vertices = np.asarray(
        geometry["vertices"][source_vertex_indices], dtype=np.float32
    )

    proxy_vertices, first_source_row, source_to_proxy_vertex = np.unique(
        source_vertices,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    first_source_row = first_source_row.astype(np.int32)
    source_to_proxy_vertex = source_to_proxy_vertex.astype(np.int32)
    welded_faces = source_to_proxy_vertex[source_local_faces].astype(np.int32)
    collapsed = np.any(np.diff(np.sort(welded_faces, axis=1), axis=1) == 0, axis=1)
    require(not bool(collapsed.any()), f"Renderable face collapsed after exact weld: {name}")
    require(
        np.array_equal(source_vertices, proxy_vertices[source_to_proxy_vertex]),
        f"Exact weld displaced vertices: {name}",
    )

    canonical = np.sort(welded_faces, axis=1)
    _, unique_first, canonical_group = np.unique(
        canonical, axis=0, return_index=True, return_inverse=True
    )
    proxy_source_renderable_rows = np.sort(unique_first).astype(np.int32)
    proxy_faces = welded_faces[proxy_source_renderable_rows].astype(np.int32)
    canonical_to_proxy = np.empty(len(unique_first), dtype=np.int32)
    canonical_to_proxy[canonical_group[proxy_source_renderable_rows]] = np.arange(
        len(proxy_faces), dtype=np.int32
    )
    renderable_to_proxy = canonical_to_proxy[canonical_group].astype(np.int32)
    require(
        np.array_equal(
            np.sort(welded_faces, axis=1),
            np.sort(proxy_faces[renderable_to_proxy], axis=1),
        ),
        f"Proxy face mapping changed geometry: {name}",
    )
    corner_matches = (
        welded_faces[:, :, None]
        == proxy_faces[renderable_to_proxy][:, None, :]
    )
    require(
        bool(np.all(corner_matches.sum(axis=2) == 1))
        and bool(np.all(corner_matches.sum(axis=1) == 1)),
        f"Proxy corner map is not a permutation: {name}",
    )
    renderable_corner_to_proxy = np.argmax(corner_matches, axis=2).astype(np.int8)

    original_to_proxy = np.full(face_count, -1, dtype=np.int32)
    original_to_proxy[renderable_rows] = renderable_to_proxy
    original_corner_to_proxy = np.full((face_count, 3), -1, dtype=np.int8)
    original_corner_to_proxy[renderable_rows] = renderable_corner_to_proxy
    renderable_face_rows = np.arange(len(renderable_rows), dtype=np.int32)
    proxy_to_original_offsets, proxy_to_original_rows, _ = csr_from_group(
        renderable_to_proxy,
        renderable_rows,
        len(proxy_faces),
    )
    flat_face_rows = np.repeat(renderable_rows, 3)
    flat_corner_rows = np.tile(np.arange(3, dtype=np.int8), len(renderable_rows))
    proxy_corner_group = (
        np.repeat(renderable_to_proxy, 3) * 3
        + renderable_corner_to_proxy.reshape(-1)
    ).astype(np.int32)
    (
        proxy_corner_to_original_offsets,
        proxy_corner_to_original_face_rows,
        corner_order,
    ) = csr_from_group(
        proxy_corner_group,
        flat_face_rows,
        len(proxy_faces) * 3,
    )
    proxy_corner_to_original_corner = flat_corner_rows[corner_order]
    mapped_corner_ids = (
        proxy_corner_to_original_face_rows.astype(np.int64) * 3
        + proxy_corner_to_original_corner.astype(np.int64)
    )
    expected_corner_ids = (
        renderable_rows[:, None].astype(np.int64) * 3
        + np.arange(3, dtype=np.int64)[None]
    ).reshape(-1)
    require(
        np.array_equal(np.sort(mapped_corner_ids), np.sort(expected_corner_ids)),
        f"Renderable corner coverage drift: {name}",
    )
    require(
        np.array_equal(np.sort(proxy_to_original_rows), renderable_rows),
        f"Renderable face coverage drift: {name}",
    )

    owners = face_edge_owners(proxy_faces)
    components, component_sizes = face_components(proxy_faces)
    winding_conflict_edges = same_direction_two_owner_edges(proxy_faces)
    topology = {
        "unique_edges": len(owners),
        "boundary_edges": sum(len(linked) == 1 for linked in owners.values()),
        "manifold_edges": sum(len(linked) == 2 for linked in owners.values()),
        "nonmanifold_edges": sum(len(linked) > 2 for linked in owners.values()),
        "same_direction_two_owner_edges": int(len(winding_conflict_edges)),
        "connected_components": int(len(component_sizes)),
        "component_faces_min": int(component_sizes.min()),
        "component_faces_median": float(np.median(component_sizes)),
        "component_faces_max": int(component_sizes.max()),
    }
    if len(proxy_faces) < UVGAMI_MIN_PROXY_FACES:
        seam_policy = "boundaries_nonmanifold_only_micro_mesh"
        seam_reason = f"proxy has fewer than {UVGAMI_MIN_PROXY_FACES} faces"
    elif topology["manifold_edges"] == 0:
        seam_policy = "boundaries_nonmanifold_only_no_manifold_adjacency"
        seam_reason = "proxy has no 2-owner manifold edge"
    else:
        seam_policy = "uvgami_hard_surface_plus_solver_boundaries_nonmanifold"
        seam_reason = "sufficient manifold topology for UVgami feature/disk cuts"

    representative_global_faces = original_face_indices[
        renderable_rows[proxy_source_renderable_rows]
    ]
    proxy_corner_normals_world = np.asarray(
        geometry["corner_normals_world"][representative_global_faces],
        dtype=np.float64,
    )
    normal_lengths = np.linalg.norm(proxy_corner_normals_world, axis=2, keepdims=True)
    require(bool(np.all(normal_lengths > 1e-12)), f"Zero representative corner normal: {name}")
    proxy_corner_normals_world /= normal_lengths
    matrix_world = np.asarray(geometry["object_matrix_world"][object_id], dtype=np.float64)
    linear = matrix_world[:3, :3]
    proxy_corner_normals_local = proxy_corner_normals_world @ linear
    local_lengths = np.linalg.norm(proxy_corner_normals_local, axis=2, keepdims=True)
    require(bool(np.all(local_lengths > 1e-12)), f"Singular local corner normal: {name}")
    proxy_corner_normals_local /= local_lengths
    reconstructed_world = proxy_corner_normals_local @ np.linalg.inv(linear)
    reconstructed_world /= np.linalg.norm(reconstructed_world, axis=2, keepdims=True)
    normal_dot = np.einsum("fci,fci->fc", reconstructed_world, proxy_corner_normals_world)
    require(float(normal_dot.min()) >= 0.999999, f"Local/world corner normal transform drift: {name}")

    proxy_vertex_source_global = source_vertex_indices[first_source_row]
    proxy_vertices_local = np.asarray(
        geometry["vertices_local"][proxy_vertex_source_global], dtype=np.float32
    )
    reconstructed_positions = (
        np.concatenate(
            [proxy_vertices_local.astype(np.float64), np.ones((len(proxy_vertices_local), 1))],
            axis=1,
        )
        @ matrix_world.T
    )[:, :3]
    require(
        bool(np.array_equal(reconstructed_positions.astype(np.float32), proxy_vertices.astype(np.float32))),
        f"Local/world representative position drift: {name}",
    )
    proxy_piece_component = np.asarray(
        [piece_component_id(object_id, int(value)) for value in components], dtype=np.int16
    )
    proxy_component_role = np.asarray(
        [piece_component_role(object_id, int(value)) for value in components]
    )
    proxy_piece_id = np.asarray(
        [f"{object_id}:{int(value)}" for value in proxy_piece_component]
    )

    arrays = {
        "schema": np.asarray("car-city.ferrari-nonbody-object-proxy/v8"),
        "object_id": np.asarray(object_id, dtype=np.int16),
        "object_name": np.asarray(name),
        "object_matrix_world": np.asarray(
            geometry["object_matrix_world"][object_id], dtype=np.float32
        ),
        "source_object_id": np.asarray(object_id, dtype=np.int16),
        "original_face_indices_global": original_face_indices,
        "original_global_faces": original_global_faces,
        "original_geometry_degenerate": degenerate.astype(np.uint8),
        "renderable_original_face_rows": renderable_rows,
        "renderable_source_vertex_indices_global": source_vertex_indices,
        "renderable_source_vertices": source_vertices,
        "renderable_source_local_faces": source_local_faces,
        "renderable_source_to_proxy_vertex": source_to_proxy_vertex,
        "renderable_welded_faces": welded_faces,
        "proxy_vertices": proxy_vertices.astype(np.float32),
        "proxy_vertices_local": proxy_vertices_local,
        "proxy_faces": proxy_faces,
        "proxy_vertex_first_source_global_index": proxy_vertex_source_global,
        "proxy_face_source_original_row": renderable_rows[proxy_source_renderable_rows],
        "proxy_face_source_global_index": original_face_indices[
            renderable_rows[proxy_source_renderable_rows]
        ],
        "original_to_proxy_face": original_to_proxy,
        "original_corner_to_proxy_corner": original_corner_to_proxy,
        "proxy_to_original_offsets": proxy_to_original_offsets,
        "proxy_to_original_face_rows": proxy_to_original_rows,
        "proxy_corner_to_original_offsets": proxy_corner_to_original_offsets,
        "proxy_corner_to_original_face_rows": proxy_corner_to_original_face_rows,
        "proxy_corner_to_original_corner": proxy_corner_to_original_corner,
        "proxy_component_id": components,
        "proxy_component_face_counts": component_sizes,
        "proxy_piece_component_id": proxy_piece_component,
        "proxy_component_role": proxy_component_role,
        "proxy_piece_id": proxy_piece_id,
        "proxy_same_direction_two_owner_edges": winding_conflict_edges,
        "proxy_corner_normals_world": proxy_corner_normals_world.astype(np.float32),
        "proxy_corner_normals_local": proxy_corner_normals_local.astype(np.float32),
    }
    multiplicity = np.diff(proxy_to_original_offsets)
    multiplicities, multiplicity_counts = np.unique(multiplicity, return_counts=True)
    report = {
        "object_id": object_id,
        "name": name,
        "source_material": str(geometry["object_source_material"][object_id]),
        "source_mesh": str(geometry["object_source_mesh"][object_id]),
        "source_vertices": vertex_count,
        "source_faces": face_count,
        "geometric_unique_faces_including_degenerate": int(
            len(all_geometric_unique_faces)
        ),
        "geometry_degenerate_faces": int(degenerate.sum()),
        "renderable_source_faces": int(len(renderable_rows)),
        "referenced_renderable_source_vertices": int(len(source_vertices)),
        "proxy_vertices": int(len(proxy_vertices)),
        "proxy_faces": int(len(proxy_faces)),
        "removed_geometric_face_copies": int(len(renderable_rows) - len(proxy_faces)),
        "maximum_vertex_displacement": 0.0,
        "minimum_corner_normal_local_world_dot": float(normal_dot.min()),
        "representative_corner_normals_not_averaged": True,
        "exact_coordinate_reconstruction": True,
        "renderable_faces_covered_exactly_once": True,
        "renderable_corners_covered_exactly_once": True,
        "degenerate_faces_quarantined_for_placeholder_strip": int(degenerate.sum()),
        "duplicate_group_multiplicity": {
            str(int(value)): int(count)
            for value, count in zip(multiplicities, multiplicity_counts, strict=True)
        },
        "topology": topology,
        "seam_policy": seam_policy,
        "seam_policy_reason": seam_reason,
        "proxy_fingerprint": array_fingerprint(proxy_vertices, proxy_faces),
    }
    return ProxyResult(arrays=arrays, report=report)
