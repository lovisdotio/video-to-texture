#!/usr/bin/env python3
"""Build isolated per-object non-BODY proxies and optional UVgami seams.

No Blender, texture, Fal or bake input is opened.  Production paths are read
only and every public artifact is constrained below ``smart_uv_v8_full_unwrap``.
Use ``--check-only`` to execute all topology/UVgami gates without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from contract_v8 import (
    DEFAULT_UVGAMI_ROOT,
    ALLOW_INTER_OBJECT_UV_STACKING,
    EXPECTED_NONBODY_GEOMETRY_DEGENERATES,
    EXPECTED_NONBODY_GEOMETRIC_UNIQUE_TRIANGLES,
    EXPECTED_NONBODY_PROXY_TRIANGLES,
    EXPECTED_NONBODY_SOURCE_TRIANGLES,
    EXPECTED_V8_SOURCE_PARENT_OBJECTS,
    EXPECTED_V8_NONBODY_RENDER_OBJECTS,
    EXPECTED_FINAL_SCENE_OBJECTS,
    EXPECTED_FINAL_SCENE_TRIANGLES,
    EXPECTED_V8_GEOMETRY_TRIANGLES,
    GEOMETRY,
    GEOMETRY_DELIVERY_MODE,
    GEOMETRY_DELIVERY_RECOMMENDED,
    GEOMETRY_SHA256,
    NONBODY_OBJECT_IDS,
    OBJECT_SEMANTIC_SPECS,
    PIECE_ID_EQUALS_OBJECT_ID,
    PIECE_ID_FIELD,
    PIECE_ID_KIND,
    PIECE_MIN_EFFECTIVE_TEXELS_OVERRIDES,
    PRESERVE_SOURCE_MATERIAL_ASSIGNMENT,
    PRESERVE_SOURCE_OBJECT_BOUNDARIES,
    ROOT,
    SOURCE_PBR_REUSE_ALLOWED,
    SOURCE_MATERIAL_TABLE_SHA256,
    SPECIAL_COMPONENT_GATES,
    TARGET_PAGE_SPECS,
    UVGAMI_COMMIT,
    UVGAMI_LICENSE,
    UVGAMI_REPOSITORY,
    UVGAMI_REVISION,
    UVGAMI_SEAM_ANGLE_DEGREES,
    V8_MATERIAL_BACKFACE_CULLING,
    WRITE_SOURCE_TRANSFER_SIDECAR,
    safe_object_stem,
    render_piece_ids_for_source_object,
    source_atlas_for_object,
    target_page_for_object,
)
from proxy_core_v8 import (
    build_object_proxy,
    chart_labels,
    face_edge_owners,
    require,
    sha256_file,
)


def git_text(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(f"Cannot inspect UVgami checkout: {detail.strip()}") from error
    return result.stdout.strip()


def load_uvgami(root: Path):
    root = root.expanduser().resolve()
    package_dir = root / "src/seams"
    license_path = root / "LICENSE"
    require((package_dir / "__init__.py").is_file(), f"Not a UVgami checkout: {root}")
    require(license_path.is_file(), f"UVgami license missing: {root}")
    commit = git_text(root, "rev-parse", "HEAD")
    require(commit == UVGAMI_COMMIT, f"UVgami commit drift: {commit}")
    require(not git_text(root, "status", "--porcelain", "--untracked-files=no"), "UVgami tracked tree is dirty")
    module_name = "_car_city_uvgami_seams_v8_8774be0"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    require(spec is not None and spec.loader is not None, "Cannot load UVgami seams package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    provenance = {
        "path": str(root),
        "repository": UVGAMI_REPOSITORY,
        "revision": UVGAMI_REVISION,
        "commit": commit,
        "tracked_tree_clean": True,
        "license": UVGAMI_LICENSE,
        "license_sha256": sha256_file(license_path),
        "entrypoint": "src.seams.seam_edges",
    }
    return module.seam_edges, provenance


def uvgami_seams(result, seam_edges) -> dict[str, Any]:
    vertices = result.arrays["proxy_vertices"]
    faces = result.arrays["proxy_faces"]
    object_id = int(result.report["object_id"])
    owners = face_edge_owners(faces)
    if object_id == 15:
        seams = np.empty((0, 2), dtype=np.int32)
        implementation = "smooth_glass_boundaries_components_only_no_UVgami"
    elif not result.report["seam_policy"].startswith("uvgami"):
        seams = np.empty((0, 2), dtype=np.int32)
        implementation = "not_run_micro_mesh; solver still splits boundaries/nonmanifold"
    else:
        raw = seam_edges(
            vertices.tolist(),
            faces.tolist(),
            angle=UVGAMI_SEAM_ANGLE_DEGREES,
        )
        seams = np.asarray(sorted(raw), dtype=np.int32).reshape(-1, 2)
        implementation = "UVgami hard-surface seams"
    raw_uvgami_count = int(len(seams))
    protected_feature_seams_removed = 0
    if object_id == 14 and len(seams):
        components = result.arrays["proxy_component_id"]
        protected = {26, 27}
        keep = []
        for edge in seams.tolist():
            linked = owners[tuple(map(int, edge))]
            linked_components = {int(components[face]) for face in linked}
            if len(linked_components) == 1 and next(iter(linked_components)) in protected:
                protected_feature_seams_removed += 1
            else:
                keep.append(edge)
        seams = np.asarray(keep, dtype=np.int32).reshape(-1, 2)
    if len(seams):
        require(int(seams.min()) >= 0 and int(seams.max()) < len(vertices), "UVgami seam vertex out of range")
        require(bool(np.all(seams[:, 0] < seams[:, 1])), "UVgami seams are not canonical")
        require(len(np.unique(seams, axis=0)) == len(seams), "UVgami seams repeat")
    winding_conflicts = np.asarray(
        result.arrays["proxy_same_direction_two_owner_edges"], dtype=np.int32
    ).reshape(-1, 2)
    solver_splits = np.asarray(
        sorted(
            {tuple(map(int, edge)) for edge in seams.tolist()}
            | {tuple(map(int, edge)) for edge in winding_conflicts.tolist()}
        ),
        dtype=np.int32,
    ).reshape(-1, 2)
    labels, sizes = chart_labels(faces, solver_splits)
    result.arrays["proxy_seam_edges"] = seams
    result.arrays["proxy_solver_split_edges"] = solver_splits
    result.arrays["proxy_preunwrap_chart_id"] = labels
    result.arrays["proxy_preunwrap_chart_face_counts"] = sizes
    conflict_set = {tuple(map(int, edge)) for edge in winding_conflicts.tolist()}
    split_set = {tuple(map(int, edge)) for edge in solver_splits.tolist()}
    require(not (conflict_set - split_set), f"Winding-conflict split missing for object {object_id}")
    protected_chart_evidence = None
    if object_id == 14:
        component = result.arrays["proxy_component_id"]
        protected_chart_evidence = {
            str(component_id): int(np.unique(labels[component == component_id]).size)
            for component_id in (26, 27)
        }
    return {
        "implementation": implementation,
        "angle_degrees": UVGAMI_SEAM_ANGLE_DEGREES if len(seams) else None,
        "raw_uvgami_seam_edges": raw_uvgami_count,
        "uvgami_seam_edges": int(len(seams)),
        "protected_feature_seams_removed": protected_feature_seams_removed,
        "same_direction_two_owner_solver_splits": int(len(winding_conflicts)),
        "solver_split_edges": int(len(solver_splits)),
        "winding_conflict_edges_missing_from_solver_split": 0,
        "protected_component_chart_counts": protected_chart_evidence,
        "preunwrap_charts": int(len(sizes)),
        "chart_faces_min": int(sizes.min()),
        "chart_faces_median": float(np.median(sizes)),
        "chart_faces_max": int(sizes.max()),
        "solver_mandatory_boundary_edges": sum(len(linked) == 1 for linked in owners.values()),
        "solver_mandatory_nonmanifold_edges": sum(len(linked) > 2 for linked in owners.values()),
        "solver_split_policy": "split every UVgami seam and same-direction two-owner edge; boundaries/nonmanifold are always solver boundaries",
    }


def safe_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    require(ROOT == path or ROOT in path.parents, f"Output escaped isolated v8 root: {path}")
    return path


def verify_existing_npz(path: Path, expected: dict[str, np.ndarray]) -> None:
    """Prove an existing proxy artifact is exactly what this builder emits."""

    require(path.is_file(), f"Existing proxy artifact missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        require(set(payload.files) == set(expected), f"Proxy array schema drift: {path}")
        for key, expected_array in expected.items():
            actual = payload[key]
            require(actual.dtype == expected_array.dtype, f"Proxy dtype drift: {path}:{key}")
            require(actual.shape == expected_array.shape, f"Proxy shape drift: {path}:{key}")
            require(np.array_equal(actual, expected_array), f"Proxy value drift: {path}:{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uvgami-root", type=Path, default=DEFAULT_UVGAMI_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/proxies")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify every existing NPZ exactly and refresh only its manifest.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional isolated JSON report, including in check-only mode.",
    )
    args = parser.parse_args()
    started = time.monotonic()
    require(sha256_file(GEOMETRY) == GEOMETRY_SHA256, "Geometry hash drift")
    with np.load(GEOMETRY, allow_pickle=False) as payload:
        geometry = {key: payload[key].copy() for key in payload.files}
    seam_edges, uvgami_provenance = load_uvgami(args.uvgami_root)
    output_dir = safe_output(args.output_dir)
    if args.verify_existing:
        require(output_dir.is_dir(), f"Existing proxy directory missing: {output_dir}")
    elif not args.check_only:
        require(not output_dir.exists(), f"Refusing to overwrite proxy directory: {output_dir}")
        output_dir.mkdir(parents=True)
    rows = []
    for object_id in NONBODY_OBJECT_IDS:
        result = build_object_proxy(geometry, object_id)
        seam_report = uvgami_seams(result, seam_edges)
        semantic = OBJECT_SEMANTIC_SPECS[object_id]
        require(
            result.report["source_material"] == semantic.source_material,
            f"Source material drift for object {object_id}",
        )
        result.report["source_atlas_id"] = source_atlas_for_object(object_id)
        result.report["target_page_id"] = target_page_for_object(object_id)
        result.report["render_piece_ids"] = [
            [source_object_id, component_id]
            for source_object_id, component_id in render_piece_ids_for_source_object(object_id)
        ]
        result.report["piece_minimum_effective_texel_overrides"] = {
            str(component_id): int(PIECE_MIN_EFFECTIVE_TEXELS_OVERRIDES[(source_object_id, component_id)])
            for source_object_id, component_id in render_piece_ids_for_source_object(object_id)
            if (source_object_id, component_id) in PIECE_MIN_EFFECTIVE_TEXELS_OVERRIDES
        }
        result.report["semantic_material"] = asdict(semantic)
        if object_id in SPECIAL_COMPONENT_GATES:
            gate = SPECIAL_COMPONENT_GATES[object_id]
            component_sizes = np.bincount(result.arrays["proxy_component_id"])
            require(len(component_sizes) == gate["expected_components"], f"Component count drift for {object_id}")
            for component_id, expected_faces in gate["required_proxy_component_faces"].items():
                require(int(component_sizes[component_id]) == expected_faces, f"Component face count drift for {object_id}:{component_id}")
            result.report["special_component_evidence"] = gate
        result.report["hard_surface_seams"] = seam_report
        if not args.check_only:
            filename = safe_object_stem(object_id, result.report["name"]) + ".npz"
            path = output_dir / filename
            if args.verify_existing:
                verify_existing_npz(path, result.arrays)
            else:
                np.savez_compressed(path, **result.arrays)
            result.report["artifact"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        rows.append(result.report)
        print(
            f"[v8] {object_id:02d} {result.report['name']}: "
            f"{result.report['source_faces']} -> {result.report['proxy_faces']} faces; "
            f"{seam_report['uvgami_seam_edges']} UVgami seams; "
            f"{seam_report['preunwrap_charts']} pre-unwrap charts",
            flush=True,
        )
    totals = {
        "objects": len(rows),
        "source_faces": sum(row["source_faces"] for row in rows),
        "geometry_degenerate_faces": sum(row["geometry_degenerate_faces"] for row in rows),
        "geometric_unique_faces_including_degenerate": sum(
            row["geometric_unique_faces_including_degenerate"] for row in rows
        ),
        "proxy_faces": sum(row["proxy_faces"] for row in rows),
        "removed_geometric_face_copies": sum(row["removed_geometric_face_copies"] for row in rows),
        "uvgami_seam_edges": sum(row["hard_surface_seams"]["uvgami_seam_edges"] for row in rows),
        "same_direction_two_owner_solver_splits": sum(
            row["hard_surface_seams"]["same_direction_two_owner_solver_splits"]
            for row in rows
        ),
        "solver_split_edges": sum(
            row["hard_surface_seams"]["solver_split_edges"] for row in rows
        ),
        "preunwrap_charts": sum(row["hard_surface_seams"]["preunwrap_charts"] for row in rows),
    }
    require(totals["objects"] == EXPECTED_V8_SOURCE_PARENT_OBJECTS, "Object total drift")
    require(totals["same_direction_two_owner_solver_splits"] == 3014, "Winding-conflict total drift")
    require(totals["source_faces"] == EXPECTED_NONBODY_SOURCE_TRIANGLES, "Source face total drift")
    require(totals["geometry_degenerate_faces"] == EXPECTED_NONBODY_GEOMETRY_DEGENERATES, "Degenerate total drift")
    require(
        totals["geometric_unique_faces_including_degenerate"]
        == EXPECTED_NONBODY_GEOMETRIC_UNIQUE_TRIANGLES,
        "All-face geometric unique total drift",
    )
    require(totals["proxy_faces"] == EXPECTED_NONBODY_PROXY_TRIANGLES, "Proxy face total drift")
    report = {
        "schema": "car-city.ferrari-nonbody-object-proxies/v8",
        "status": (
            "passed_check_only"
            if args.check_only
            else "passed_existing_artifacts_verified"
            if args.verify_existing
            else "passed"
        ),
        "operation": "exact_object_proxy_and_hard_surface_seams_only",
        "blender_run": False,
        "unwrap_performed": False,
        "packing_performed": False,
        "bake_performed": False,
        "fal_called": False,
        "source": {"geometry": str(GEOMETRY), "sha256": GEOMETRY_SHA256},
        "authoritative_source_material_table_sha256": SOURCE_MATERIAL_TABLE_SHA256,
        "uvgami": uvgami_provenance,
        "geometry_delivery": {
            "mode": GEOMETRY_DELIVERY_MODE,
            "recommended": GEOMETRY_DELIVERY_RECOMMENDED,
            "source_parent_object_count": EXPECTED_V8_SOURCE_PARENT_OBJECTS,
            "nonbody_render_object_count": EXPECTED_V8_NONBODY_RENDER_OBJECTS,
            "final_scene_object_count_with_BODY": EXPECTED_FINAL_SCENE_OBJECTS,
            "triangle_count": EXPECTED_V8_GEOMETRY_TRIANGLES,
            "final_scene_triangle_count_with_BODY": EXPECTED_FINAL_SCENE_TRIANGLES,
            "preserve_source_object_boundaries": PRESERVE_SOURCE_OBJECT_BOUNDARIES,
            "preserve_source_material_identity": PRESERVE_SOURCE_MATERIAL_ASSIGNMENT,
            "material_backface_culling": V8_MATERIAL_BACKFACE_CULLING,
            "source_pbr_reuse_allowed": SOURCE_PBR_REUSE_ALLOWED,
            "piece_id_field": PIECE_ID_FIELD,
            "piece_id_kind": PIECE_ID_KIND,
            "piece_id_equals_object_id": PIECE_ID_EQUALS_OBJECT_ID,
            "allow_inter_object_uv_stacking": ALLOW_INTER_OBJECT_UV_STACKING,
            "write_source_transfer_sidecar": WRITE_SOURCE_TRANSFER_SIDECAR,
        },
        "totals": totals,
        "target_pages": [
            {
                "page_id": slug,
                "object_ids": list(spec.object_ids),
                "base_size": [spec.base_size, spec.base_size],
                "allowed_sizes": [[size, size] for size in spec.allowed_sizes],
                "shader_class": spec.shader_class,
                "source_faces": sum(
                    row["source_faces"] for row in rows if row["target_page_id"] == slug
                ),
                "proxy_faces": sum(
                    row["proxy_faces"] for row in rows if row["target_page_id"] == slug
                ),
                "uvgami_seam_edges": sum(
                    row["hard_surface_seams"]["uvgami_seam_edges"]
                    for row in rows
                    if row["target_page_id"] == slug
                ),
                "preunwrap_charts": sum(
                    row["hard_surface_seams"]["preunwrap_charts"]
                    for row in rows
                    if row["target_page_id"] == slug
                ),
            }
            for slug, spec in TARGET_PAGE_SPECS.items()
        ],
        "objects": rows,
        "hard_gates": {
            "body_excluded": True,
            "source_assets_unchanged": True,
            "exact_coordinate_weld": True,
            "all_renderable_faces_and_corners_mapped": True,
            "all_degenerate_faces_quarantined": True,
            "boundaries_and_nonmanifold_edges_forced_solver_splits": True,
            "all_same_direction_two_owner_edges_forced_solver_splits": totals["same_direction_two_owner_solver_splits"] == 3014,
            "zero_solver_chart_winding_conflict_crossings": all(
                row["hard_surface_seams"]["winding_conflict_edges_missing_from_solver_split"] == 0
                for row in rows
            ),
            "representative_corner_normals_local_world_stored": all(
                row["minimum_corner_normal_local_world_dot"] >= 0.999999
                and row["representative_corner_normals_not_averaged"]
                for row in rows
            ),
            "semantic_metadata_complete": True,
            "inter_object_uv_stacking_forbidden": not ALLOW_INTER_OBJECT_UV_STACKING,
            "new_pbr_materials_required": not SOURCE_PBR_REUSE_ALLOWED,
            "unique_proxy_geometry_contract_declared": GEOMETRY_DELIVERY_RECOMMENDED,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }
    if not args.check_only:
        report_path = output_dir.parent / "proxy_manifest.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.report is not None:
        explicit_report = safe_output(args.report)
        explicit_report.parent.mkdir(parents=True, exist_ok=True)
        explicit_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("V8_PROXY_BUILD=" + json.dumps({
        "status": report["status"],
        "totals": totals,
        "output_dir": None if args.check_only else str(output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
