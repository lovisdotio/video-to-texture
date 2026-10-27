#!/usr/bin/env python3
"""Additive handoff for the Ferrari targeted UV/rebake v9 pass.

This module deliberately does not launch Blender, Fal, Seedream or the GPU
baker.  It defines the exact target registry and validates local re-unwrap
patches.  Two additive handoffs are available: ``strict_patch`` keeps the
original atlas positions and recomposes target-only RGB, while ``page_repack``
allows complete repacking and RGB rebaking of the five touched non-BODY pages.

The strict mode's preservation invariant is pixel based.  The page-repack
mode instead preserves each KEEP chart's topology and local UV shape up to an
orientation-preserving similarity transform; placement and scale in the page
may change.  Tires remain byte-identical in both modes.  Gutters are always
regenerated from final piece/chart owners without cross-domain averaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
V8_ROOT = HERE.parent
BAKE_V8 = V8_ROOT / "bake_v8"
if str(BAKE_V8) not in sys.path:
    sys.path.insert(0, str(BAKE_V8))

import contract as bake_contract


SCHEMA = "car-city.ferrari-targeted-rebake/v9"
PATCH_SCHEMA = "car-city.ferrari-targeted-reunwrap-patch/v9"
SCENE_MESH_COUNT = 66
TARGETED_SCENE_MESH_COUNT = 13
PRESERVED_SCENE_MESH_COUNT = 53
DEFAULT_PADDING = 16
MODE_STRICT_PATCH = "strict_patch"
MODE_PAGE_REPACK = "page_repack"
SUPPORTED_MODES = (MODE_STRICT_PATCH, MODE_PAGE_REPACK)

BASE_TEXTURE_ROOT = V8_ROOT / "seedream_full_texture_v5" / "textures"
BASE_DIAGNOSTIC_ROOT = (
    V8_ROOT
    / "bake_v8"
    / "diagnostics"
    / "production_seedvr_rgb_v1"
    / "extracted"
    / "diagnostics"
)


@dataclass(frozen=True)
class TargetSelector:
    selector: str
    page_id: str
    resolution: int
    source_object_id: int
    piece_component_id: int
    nonbody_piece_key: str | None


TARGETS: tuple[TargetSelector, ...] = (
    TargetSelector("1:17", "body_hero_8k", 8192, 1, 17, None),
    TargetSelector("49:-1", "exterior_opaque", 4096, 49, -1, "49:-1"),
    TargetSelector("15:3", "glass", 4096, 15, 3, "15:3"),
    TargetSelector("6:-1", "lights_translucent", 4096, 6, -1, "6:-1"),
    TargetSelector("26:-1", "running_metal", 8192, 26, -1, "26:-1"),
    TargetSelector("27:-1", "running_metal", 8192, 27, -1, "27:-1"),
    TargetSelector("7:-1", "cockpit", 8192, 7, -1, "7:-1"),
    TargetSelector("17:-1", "cockpit", 8192, 17, -1, "17:-1"),
    TargetSelector("18:-1", "cockpit", 8192, 18, -1, "18:-1"),
    TargetSelector("19:-1", "cockpit", 8192, 19, -1, "19:-1"),
    TargetSelector("23:-1", "cockpit", 8192, 23, -1, "23:-1"),
    TargetSelector("28:-1", "cockpit", 8192, 28, -1, "28:-1"),
    TargetSelector("36:-1", "cockpit", 8192, 36, -1, "36:-1"),
)

TARGET_BY_SELECTOR = {row.selector: row for row in TARGETS}
if len(TARGET_BY_SELECTOR) != TARGETED_SCENE_MESH_COUNT:
    raise RuntimeError("The targeted v9 registry must contain exactly 13 selectors")
if SCENE_MESH_COUNT - TARGETED_SCENE_MESH_COUNT != PRESERVED_SCENE_MESH_COUNT:
    raise RuntimeError("The scene preservation count is inconsistent")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(payload: Mapping[str, np.ndarray], name: str) -> Any:
    require(name in payload, f"Patch is missing {name}")
    value = np.asarray(payload[name])
    require(value.size == 1, f"Patch field {name} must be scalar")
    return value.reshape(-1)[0].item()


def nonbody_piece_index(piece_key: str) -> int:
    lookup = {
        bake_contract.piece_key(piece): index
        for index, piece in enumerate(bake_contract.EXPECTED_PIECE_IDS)
    }
    require(piece_key in lookup, f"Unknown canonical non-BODY piece {piece_key}")
    return int(lookup[piece_key])


def targeted_nonbody_piece_indices(page_id: str | None = None) -> tuple[int, ...]:
    values = []
    for target in TARGETS:
        if target.nonbody_piece_key is None:
            continue
        if page_id is not None and target.page_id != page_id:
            continue
        values.append(nonbody_piece_index(target.nonbody_piece_key))
    return tuple(sorted(values))


def targets_by_page() -> dict[str, tuple[TargetSelector, ...]]:
    output: dict[str, list[TargetSelector]] = {}
    for target in TARGETS:
        output.setdefault(target.page_id, []).append(target)
    return {page_id: tuple(rows) for page_id, rows in output.items()}


def _base_texture_for_page(page_id: str) -> Path:
    if page_id == "body_hero_8k":
        return bake_contract.BODY_V6_TEXTURE
    return BASE_TEXTURE_ROOT / f"{page_id}_seedream_v5_final.png"


def build_plan(
    reunwrap_run: Path | None = None,
    *,
    mode: str = MODE_STRICT_PATCH,
) -> dict[str, Any]:
    require(mode in SUPPORTED_MODES, f"Unsupported rebake mode {mode!r}")
    strict = mode == MODE_STRICT_PATCH
    page_rows = []
    for page_id, targets in targets_by_page().items():
        resolution = targets[0].resolution
        nonbody_keys = [
            row.nonbody_piece_key
            for row in targets
            if row.nonbody_piece_key is not None
        ]
        is_body = page_id == "body_hero_8k"
        if is_body:
            mode_fields = {
                "layout_strategy": "separate_body_component_repair_not_nonbody_page_repack",
                "bake_scope": "body_component_1:17_only",
                "composition_scope": "separate_body_texture_handoff",
                "preserve_untargeted_uv": "body_specific_gate",
                "preserve_untargeted_occupied_rgb": "body_specific_gate",
                "base_texture_role": "approved_body_reference_and_composition_source",
            }
        elif strict:
            mode_fields = {
                "layout_strategy": "keep_existing_page_positions_exact",
                "bake_scope": "target_faces_only",
                "composition_scope": "replace_target_occupied_texels_only",
                "preserve_untargeted_uv": "exact_array_identity",
                "preserve_untargeted_occupied_rgb": "byte_exact",
                "base_texture_role": "untargeted_rgb_source",
            }
        else:
            mode_fields = {
                "layout_strategy": "full_page_repack",
                "bake_scope": "full_page_all_pieces",
                "composition_scope": "replace_full_page_texture",
                "preserve_untargeted_uv": (
                    "same_face_chart_partition_and_local_shape_up_to_"
                    "orientation_preserving_similarity_per_keep_chart"
                ),
                "preserve_untargeted_occupied_rgb": "not_required_full_page_rebake",
                "base_texture_role": "visual_and_material_reference_only",
            }
        row = {
                "page_id": page_id,
                "resolution": [resolution, resolution],
                "padding_pixels": DEFAULT_PADDING,
                "target_selectors": [row.selector for row in targets],
                "target_nonbody_piece_keys": nonbody_keys,
                "target_nonbody_piece_indices": [
                    nonbody_piece_index(value) for value in nonbody_keys
                ],
                "base_texture": str(_base_texture_for_page(page_id)),
                "base_piece_owner": (
                    None
                    if is_body
                    else str(BASE_DIAGNOSTIC_ROOT / page_id / "piece_owner_id.png")
                ),
                "base_chart_owner": (
                    None
                    if is_body
                    else str(BASE_DIAGNOSTIC_ROOT / page_id / "chart_owner_id.png")
                ),
                "layout_input": (
                    "new_full_body_layout_with_component_owner_1:17"
                    if is_body
                    else "new_full_v9_page_layout_owner_maps"
                    if strict
                    else "new_full_v9_page_repack_with_similarity_proof"
                ),
                "owner_map_policy": "regenerate_from_new_full_layout",
                "gutter_policy": (
                    "regenerate_16px_nearest_owned_piece_chart_no_average"
                ),
            }
        row.update(mode_fields)
        page_rows.append(row)
    if strict:
        scene_uv_policy = "exact_array_identity"
        scene_texture_policy = "exact_occupied_rgb_identity"
        handoff = [
            "validate the 13 local re-unwrap patches",
            "allocate final atlas positions while keeping untargeted UV arrays exact",
            "block bake while atlas_coordinates_final is false",
            "raster new full-layout piece/chart owner maps",
            "slice each runtime page to target global piece IDs",
            "rebake and texture-edit target occupied domains only",
            "recompose over approved textures and regenerate owned gutters",
        ]
        forbidden = [
            "overwrite canonical v8 unwrap or approved v5 textures",
            "move an untargeted UV corner",
            "sample target RGB into an untargeted occupied texel",
            "average colours across piece/chart domains",
            "treat BODY 1:17 as a canonical non-BODY piece index",
        ]
        status = "local_patch_contract_prepared_final_atlas_placement_required"
    else:
        scene_uv_policy = (
            "keep_chart_topology_and_local_shape_preserved_up_to_"
            "orientation_preserving_similarity"
        )
        scene_texture_policy = (
            "full_rgb_rebake_on_five_touched_nonbody_pages;tires_byte_exact;"
            "body_1:17_separate"
        )
        handoff = [
            "validate the 13 local re-unwrap patches",
            "repack every touched non-BODY page with all charts movable",
            "prove every KEEP chart retains face partition and local UV shape up to similarity",
            "block bake until overlap, bounds, degeneracy, margin and similarity gates pass",
            "rebake RGB for every piece on each touched non-BODY page",
            "regenerate full-page piece/chart owners and 16px gutters",
            "pass tires through byte-identically and process BODY 1:17 separately",
        ]
        forbidden = [
            "overwrite canonical v8 or approved v5 inputs in place",
            "split, merge, shear or non-uniformly scale a KEEP chart",
            "reuse stale owner maps after page repack",
            "perform target-only RGB composition on a repacked page",
            "change tires UV, owners or texture bytes",
            "treat BODY 1:17 as a canonical non-BODY piece index",
        ]
        status = "page_repack_contract_prepared_final_page_allocation_required"
    return {
        "schema": SCHEMA,
        "mode": mode,
        "status": status,
        "source_v8_root": str(V8_ROOT),
        "targeted_reunwrap_run": str(reunwrap_run) if reunwrap_run else None,
        "scene_scope": {
            "scene_meshes": SCENE_MESH_COUNT,
            "targeted_meshes": TARGETED_SCENE_MESH_COUNT,
            "preserved_meshes": PRESERVED_SCENE_MESH_COUNT,
            "preserved_mesh_uv_policy": scene_uv_policy,
            "preserved_mesh_texture_policy": scene_texture_policy,
        },
        "piece_scope": {
            "canonical_nonbody_piece_ids": len(bake_contract.EXPECTED_PIECE_IDS),
            "targeted_nonbody_piece_ids": len(targeted_nonbody_piece_indices()),
            "preserved_nonbody_piece_ids": (
                len(bake_contract.EXPECTED_PIECE_IDS)
                - len(targeted_nonbody_piece_indices())
            ),
            "body_component_targets": ["1:17"],
        },
        "pages": page_rows,
        "untouched_pages": ["tires"],
        "untouched_page_contract": {
            "tires": {
                "uv": "byte_exact",
                "piece_chart_owner_maps": "byte_exact",
                "texture": "byte_exact",
                "rebake": False,
            }
        },
        "body_contract": {
            "selector": "1:17",
            "routing": "separate_body_workflow",
            "included_in_nonbody_page_repack": False,
        },
        "final_atlas_gate": {
            "atlas_coordinates_final": True,
            "atlas_placement_required": False,
            "overlap_pairs": 0,
            "uv_degenerate_triangles": 0,
            "out_of_bounds_triangles": 0,
            "minimum_margin_pixels": DEFAULT_PADDING,
            "keep_similarity_required": not strict,
        },
        "handoff": handoff,
        "forbidden": forbidden,
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name].copy() for name in handle.files}


def validate_patch_payload(payload: Mapping[str, np.ndarray], path: Path | None = None) -> dict[str, Any]:
    object_id = int(_scalar(payload, "source_object_id"))
    component_id = int(_scalar(payload, "piece_component_id"))
    selector = f"{object_id}:{component_id}"
    require(selector in TARGET_BY_SELECTOR, f"Unexpected re-unwrap selector {selector}")
    target = TARGET_BY_SELECTOR[selector]

    faces = np.asarray(payload.get("selected_proxy_face_indices"))
    require(faces.ndim == 1 and len(faces) > 0, f"{selector}: empty face selection")
    require(np.issubdtype(faces.dtype, np.integer), f"{selector}: face indices are not integers")
    require(bool(np.all(faces >= 0)), f"{selector}: negative face index")
    require(len(np.unique(faces)) == len(faces), f"{selector}: duplicate face index")

    uv = np.asarray(payload.get("replacement_proxy_corner_uv"), dtype=np.float64)
    if uv.ndim == 2:
        require(uv.shape == (len(faces) * 3, 2), f"{selector}: bad flat corner UV shape")
        triangles = uv.reshape(-1, 3, 2)
    else:
        require(uv.shape == (len(faces), 3, 2), f"{selector}: bad corner UV shape")
        triangles = uv
    require(bool(np.all(np.isfinite(triangles))), f"{selector}: non-finite UV")
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    signed_area2 = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    require(bool(np.all(np.abs(signed_area2) > 1e-12)), f"{selector}: degenerate UV triangle")

    chart = np.asarray(payload.get("local_chart_id"))
    require(chart.shape == (len(faces),), f"{selector}: local_chart_id shape mismatch")
    require(np.issubdtype(chart.dtype, np.integer), f"{selector}: chart IDs are not integers")
    require(bool(np.all(chart >= 0)), f"{selector}: negative chart ID")

    if "target_page_id" in payload:
        page_id = str(_scalar(payload, "target_page_id"))
        require(page_id == target.page_id, f"{selector}: page {page_id} != {target.page_id}")
    if "page_resolution" in payload:
        resolution = int(_scalar(payload, "page_resolution"))
        require(resolution == target.resolution, f"{selector}: page resolution drift")

    require(
        bool(_scalar(payload, "atlas_coordinates_final")) is False,
        f"{selector}: local patch was incorrectly labelled atlas-final",
    )
    require(
        bool(_scalar(payload, "atlas_placement_required")) is True,
        f"{selector}: atlas placement requirement is missing",
    )
    require(
        bool(_scalar(payload, "keep_faces_may_move")) is False,
        f"{selector}: patch authorizes movement of preserved faces",
    )

    for required_hash in ("source_sha256", "proxy_sha256"):
        require(required_hash in payload, f"{selector}: missing {required_hash}")
    hash_fields = []
    for name, raw_value in payload.items():
        if not name.endswith("sha256"):
            continue
        value = str(np.asarray(raw_value).reshape(-1)[0])
        require(bool(re.fullmatch(r"[0-9a-f]{64}", value)), f"{selector}: malformed {name}")
        hash_fields.append(name)
    require(hash_fields, f"{selector}: no immutable SHA-256 proof")

    return {
        "selector": selector,
        "page_id": target.page_id,
        "faces": int(len(faces)),
        "charts": int(len(np.unique(chart))),
        "uv_min": [float(value) for value in triangles.min(axis=(0, 1))],
        "uv_max": [float(value) for value in triangles.max(axis=(0, 1))],
        "hash_fields": sorted(hash_fields),
        "path": str(path) if path else None,
        "sha256": sha256_file(path) if path else None,
    }


def validate_reunwrap_run(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root).expanduser().resolve()
    patch_root = run_root / "patches"
    require(patch_root.is_dir(), f"Missing patch folder: {patch_root}")
    rows = [validate_patch_payload(load_npz(path), path) for path in sorted(patch_root.glob("*.npz"))]
    selectors = [row["selector"] for row in rows]
    require(len(selectors) == len(set(selectors)), "Duplicate selector patch")
    expected = set(TARGET_BY_SELECTOR)
    actual = set(selectors)
    require(actual == expected, f"Patch registry mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return {
        "schema": PATCH_SCHEMA,
        "status": "passed",
        "run_root": str(run_root),
        "patch_count": len(rows),
        "selectors": sorted(selectors),
        "patches": rows,
    }


def validate_final_atlas_readiness(
    report: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Reject local-only UVs and incomplete page allocations before any bake."""

    require(mode in SUPPORTED_MODES, f"Unsupported rebake mode {mode!r}")
    require(
        bool(report.get("atlas_coordinates_final")) is True,
        "Atlas coordinates are still local/non-final; rebake is blocked",
    )
    require(
        bool(report.get("atlas_placement_required")) is False,
        "Atlas placement is still required; rebake is blocked",
    )
    integer_zero_gates = (
        "overlap_pairs",
        "uv_degenerate_triangles",
        "out_of_bounds_triangles",
    )
    for name in integer_zero_gates:
        require(int(report.get(name, -1)) == 0, f"Final atlas gate failed: {name}")
    margin = int(report.get("minimum_margin_pixels", -1))
    require(margin >= DEFAULT_PADDING, "Final atlas margin is below 16 pixels")
    if mode == MODE_PAGE_REPACK:
        require(
            bool(report.get("keep_similarity_passed")) is True,
            "KEEP chart similarity proof is missing or failed",
        )
    return {
        "status": "passed",
        "mode": mode,
        "atlas_coordinates_final": True,
        "minimum_margin_pixels": margin,
        "overlap_pairs": 0,
        "uv_degenerate_triangles": 0,
        "out_of_bounds_triangles": 0,
        "keep_similarity_passed": (
            bool(report.get("keep_similarity_passed"))
            if mode == MODE_PAGE_REPACK
            else None
        ),
    }


def _fit_orientation_preserving_similarity(
    old_points: np.ndarray,
    new_points: np.ndarray,
    *,
    relative_tolerance: float,
) -> dict[str, Any]:
    old = np.asarray(old_points, dtype=np.float64)
    new = np.asarray(new_points, dtype=np.float64)
    require(old.shape == new.shape and old.ndim == 2 and old.shape[1] == 2, "Similarity points must be matching Nx2 arrays")
    require(len(old) >= 3, "A KEEP chart needs at least three UV corners")
    require(bool(np.all(np.isfinite(old)) and np.all(np.isfinite(new))), "Similarity input contains non-finite UV")
    old_center = old.mean(axis=0)
    new_center = new.mean(axis=0)
    old_zero = old - old_center
    new_zero = new - new_center
    denominator = float(np.square(old_zero).sum())
    require(denominator > 1e-20, "A KEEP chart has no measurable UV extent")

    left, _singular, right_t = np.linalg.svd(old_zero.T @ new_zero)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        right_t[-1] *= -1.0
        rotation = left @ right_t
    scale = float(np.sum((old_zero @ rotation) * new_zero) / denominator)
    require(scale > 0.0 and np.isfinite(scale), "KEEP chart similarity has invalid scale")
    translation = new_center - scale * (old_center @ rotation)
    predicted = scale * (old @ rotation) + translation
    errors = np.linalg.norm(predicted - new, axis=1)
    new_extent = max(float(np.linalg.norm(new_zero, axis=1).max()), 1e-12)
    relative_error = float(errors.max()) / new_extent
    require(
        relative_error <= float(relative_tolerance),
        f"KEEP chart local shape changed (relative similarity residual {relative_error:.3e})",
    )
    return {
        "scale": scale,
        "rotation_degrees": float(
            np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0]))
        ),
        "translation": [float(value) for value in translation],
        "determinant": float(np.linalg.det(rotation)),
        "maximum_absolute_error": float(errors.max()),
        "maximum_relative_error": relative_error,
    }


def validate_keep_chart_similarity(
    *,
    old_corner_uv: np.ndarray,
    new_corner_uv: np.ndarray,
    old_face_chart: np.ndarray,
    new_face_chart: np.ndarray,
    keep_face_mask: np.ndarray,
    relative_tolerance: float = 2e-5,
) -> dict[str, Any]:
    """Prove KEEP chart partition and local UV shape survive a page repack.

    Face and corner order are assumed to be the immutable proxy order already
    checked by the geometry contract. Chart numeric IDs may change, but a KEEP
    chart cannot split, merge, shear, reflect or receive non-uniform scaling.
    """

    old_uv = np.asarray(old_corner_uv, dtype=np.float64)
    new_uv = np.asarray(new_corner_uv, dtype=np.float64)
    old_chart = np.asarray(old_face_chart)
    new_chart = np.asarray(new_face_chart)
    keep = np.asarray(keep_face_mask, dtype=bool)
    require(old_uv.shape == new_uv.shape and old_uv.ndim == 3 and old_uv.shape[1:] == (3, 2), "Corner UV arrays must be matching Fx3x2")
    require(old_chart.shape == new_chart.shape == keep.shape == (len(old_uv),), "Face chart/mask shape mismatch")
    require(bool(np.any(keep)), "Similarity audit contains no KEEP faces")

    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    rows = []
    for old_id in sorted(map(int, np.unique(old_chart[keep]))):
        face_mask = keep & (old_chart == old_id)
        mapped = np.unique(new_chart[face_mask])
        require(len(mapped) == 1, f"KEEP chart {old_id} was split during page repack")
        new_id = int(mapped[0])
        require(new_id not in reverse, f"KEEP charts {reverse.get(new_id)} and {old_id} were merged")
        mapping[old_id] = new_id
        reverse[new_id] = old_id
        fit = _fit_orientation_preserving_similarity(
            old_uv[face_mask].reshape(-1, 2),
            new_uv[face_mask].reshape(-1, 2),
            relative_tolerance=relative_tolerance,
        )
        rows.append(
            {
                "old_chart_id": old_id,
                "new_chart_id": new_id,
                "faces": int(face_mask.sum()),
                "corners": int(face_mask.sum() * 3),
                "similarity": fit,
            }
        )
    return {
        "status": "passed",
        "keep_faces": int(keep.sum()),
        "keep_charts": len(rows),
        "chart_partition_preserved": True,
        "orientation_preserving_similarity_per_chart": True,
        "relative_tolerance": float(relative_tolerance),
        "maximum_relative_error": max(
            row["similarity"]["maximum_relative_error"] for row in rows
        ),
        "charts": rows,
    }


def slice_page_for_piece_indices(page: Any, selected_piece_indices: Sequence[int]) -> Any:
    """Return a target-only PageRuntimeContract while retaining global piece IDs.

    Whole-car visibility stays on the parent RuntimeV8Contract; only the UV
    surface raster and sample accumulation page are narrowed here.
    """

    selected = np.asarray(sorted(set(map(int, selected_piece_indices))), dtype=np.int32)
    require(len(selected) > 0, "A target page slice cannot be empty")
    keep = np.isin(np.asarray(page.face_piece), selected)
    require(bool(np.any(keep)), "No requested piece appears on the page")
    actual = np.unique(np.asarray(page.face_piece)[keep]).astype(np.int32)
    require(np.array_equal(actual, selected), "A requested piece is absent from the page")

    face_indices = np.flatnonzero(keep)
    corner_indices = np.asarray(page.uv_faces)[face_indices].reshape(-1)
    uv = np.asarray(page.uv)[corner_indices].copy()
    world = np.asarray(page.world_by_uv_vertex)[corner_indices].copy()
    uv_faces = np.arange(len(uv), dtype=np.int32).reshape(-1, 3)
    chart_original = np.asarray(page.face_chart_original)[keep].copy()
    _unique_chart, chart_dense = np.unique(chart_original, return_inverse=True)
    chart_dense = chart_dense.astype(np.int32)
    face_object = np.asarray(page.face_object)[keep].copy()

    source_count_by_piece = {
        int(piece): int(count)
        for piece, count in zip(page.piece_indices, page.piece_source_renderable_face_counts)
    }
    key_by_piece = {
        int(piece): str(key) for piece, key in zip(page.piece_indices, page.piece_keys)
    }
    proxy_counts = np.asarray(
        [int(np.count_nonzero(np.asarray(page.face_piece)[keep] == piece)) for piece in selected],
        dtype=np.int32,
    )
    source_counts = np.asarray([source_count_by_piece[int(piece)] for piece in selected], dtype=np.int32)

    selected_objects = np.unique(face_object).astype(np.int32)
    object_name_by_id = {
        int(object_id): str(name) for object_id, name in zip(page.object_ids, page.object_names)
    }
    object_positions = {
        int(object_id): index for index, object_id in enumerate(np.asarray(page.object_ids))
    }
    sidecar_positions = [object_positions[int(object_id)] for object_id in selected_objects]

    return replace(
        page,
        object_ids=selected_objects,
        object_names=tuple(object_name_by_id[int(value)] for value in selected_objects),
        piece_indices=selected,
        piece_keys=tuple(key_by_piece[int(value)] for value in selected),
        piece_proxy_face_counts=proxy_counts,
        piece_source_renderable_face_counts=source_counts,
        uv=uv,
        uv_faces=uv_faces,
        world_by_uv_vertex=world,
        face_object=face_object,
        face_piece=np.asarray(page.face_piece)[keep].copy(),
        face_chart=chart_dense,
        face_chart_original=chart_original,
        face_normals_world=np.asarray(page.face_normals_world)[keep].copy(),
        visibility_face_indices=np.asarray(page.visibility_face_indices)[keep].copy(),
        sidecar_paths=tuple(page.sidecar_paths[index] for index in sidecar_positions),
        sidecar_sha256=tuple(page.sidecar_sha256[index] for index in sidecar_positions),
        source_face_count=int(source_counts.sum()),
        source_degenerate_count=0,
        source_chart_slots=int(len(_unique_chart)),
    )


def _validate_owner_map(piece_owner: np.ndarray, chart_owner: np.ndarray, label: str) -> None:
    require(piece_owner.ndim == 2, f"{label}: piece owner must be HxW")
    require(chart_owner.shape == piece_owner.shape, f"{label}: chart owner shape mismatch")
    require(np.issubdtype(piece_owner.dtype, np.integer), f"{label}: piece owner must be integer")
    require(np.issubdtype(chart_owner.dtype, np.integer), f"{label}: chart owner must be integer")
    require(bool(np.all(piece_owner >= 0) and np.all(chart_owner >= 0)), f"{label}: negative owner")
    require(np.array_equal(piece_owner > 0, chart_owner > 0), f"{label}: piece/chart occupancy mismatch")


def regenerate_owned_gutters(
    occupied_rgb: np.ndarray,
    piece_owner_plus1: np.ndarray,
    chart_owner_plus1: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grow deterministic nearest occupied donors for a finite clean padding.

    OpenCV's labelled chessboard distance transform gives one discrete donor
    pixel, never an average.  The donor's exact piece/chart IDs are propagated
    with its RGB.  Chessboard distance matches the 8-neighbour padding footprint
    used by the v8 baker.
    """

    import cv2

    rgb = np.asarray(occupied_rgb)
    piece = np.asarray(piece_owner_plus1)
    chart = np.asarray(chart_owner_plus1)
    _validate_owner_map(piece, chart, "new layout")
    require(rgb.shape == piece.shape + (3,), "RGB/owner shape mismatch")
    require(rgb.dtype == np.uint8, "RGB must be uint8")
    require(0 <= int(padding) <= 256, "Padding is outside the supported range")
    occupied = piece > 0
    require(bool(np.any(occupied)), "Cannot grow gutters without occupied texels")

    source = np.where(occupied, 0, 255).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_C,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    seed_labels = labels[occupied]
    require(bool(np.all(seed_labels > 0)), "Distance transform did not label every seed")
    require(len(np.unique(seed_labels)) == len(seed_labels), "Pixel-label transform merged occupied donors")
    slots = int(labels.max()) + 1
    donor_rgb = np.zeros((slots, 3), dtype=np.uint8)
    donor_piece = np.zeros(slots, dtype=np.uint16)
    donor_chart = np.zeros(slots, dtype=np.uint16)
    donor_rgb[seed_labels] = rgb[occupied]
    donor_piece[seed_labels] = piece[occupied].astype(np.uint16, copy=False)
    donor_chart[seed_labels] = chart[occupied].astype(np.uint16, copy=False)

    padded = (labels > 0) & (distance <= float(padding))
    output = np.zeros_like(rgb)
    gutter_piece = np.zeros(piece.shape, dtype=np.uint16)
    gutter_chart = np.zeros(chart.shape, dtype=np.uint16)
    selected_labels = labels[padded]
    output[padded] = donor_rgb[selected_labels]
    gutter_piece[padded] = donor_piece[selected_labels]
    gutter_chart[padded] = donor_chart[selected_labels]
    require(np.array_equal(output[occupied], rgb[occupied]), "Occupied RGB changed during gutter growth")
    require(np.array_equal(gutter_piece[occupied], piece[occupied]), "Occupied piece owner changed")
    require(np.array_equal(gutter_chart[occupied], chart[occupied]), "Occupied chart owner changed")
    return output, padded, gutter_piece, gutter_chart


def compose_targeted_arrays(
    *,
    base_rgb: np.ndarray,
    base_piece_owner_plus1: np.ndarray,
    base_chart_owner_plus1: np.ndarray,
    new_piece_owner_plus1: np.ndarray,
    new_chart_owner_plus1: np.ndarray,
    target_rgb: np.ndarray,
    target_piece_owner_plus1: np.ndarray,
    target_chart_owner_plus1: np.ndarray,
    target_piece_indices: Sequence[int],
    padding: int = DEFAULT_PADDING,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Compose target-only RGB while proving all untargeted occupancy unchanged."""

    base_rgb = np.asarray(base_rgb)
    target_rgb = np.asarray(target_rgb)
    base_piece = np.asarray(base_piece_owner_plus1)
    base_chart = np.asarray(base_chart_owner_plus1)
    new_piece = np.asarray(new_piece_owner_plus1)
    new_chart = np.asarray(new_chart_owner_plus1)
    target_piece = np.asarray(target_piece_owner_plus1)
    target_chart = np.asarray(target_chart_owner_plus1)
    _validate_owner_map(base_piece, base_chart, "base")
    _validate_owner_map(new_piece, new_chart, "new full layout")
    _validate_owner_map(target_piece, target_chart, "target bake")
    require(base_piece.shape == new_piece.shape == target_piece.shape, "Owner-map resolution mismatch")
    require(base_rgb.shape == target_rgb.shape == base_piece.shape + (3,), "Texture resolution mismatch")
    require(base_rgb.dtype == target_rgb.dtype == np.uint8, "Textures must be uint8 RGB")

    target_values = np.asarray(sorted({int(value) + 1 for value in target_piece_indices}), dtype=np.int64)
    require(len(target_values) > 0, "No target piece indices supplied")
    old_target = np.isin(base_piece, target_values)
    new_target = np.isin(new_piece, target_values)
    baked_target = np.isin(target_piece, target_values)
    require(bool(np.any(new_target)), "New layout contains no target occupancy")
    require(np.array_equal(new_target, baked_target), "Target-only bake occupancy differs from new layout")
    require(np.array_equal(np.where(new_target, new_piece, 0), np.where(baked_target, target_piece, 0)), "Target-only bake piece owners differ from new layout")
    require(not bool(np.any((target_piece > 0) & ~baked_target)), "Target bake contains an untargeted owner")

    old_preserved_owner = np.where((base_piece > 0) & ~old_target, base_piece, 0)
    new_preserved_owner = np.where((new_piece > 0) & ~new_target, new_piece, 0)
    require(np.array_equal(old_preserved_owner, new_preserved_owner), "An untargeted UV occupancy/owner moved in the V9 layout")
    preserved = new_preserved_owner > 0

    occupied_rgb = np.zeros_like(base_rgb)
    occupied_rgb[preserved] = base_rgb[preserved]
    occupied_rgb[new_target] = target_rgb[new_target]
    output, padded, gutter_piece, gutter_chart = regenerate_owned_gutters(
        occupied_rgb, new_piece, new_chart, padding
    )
    require(np.array_equal(output[preserved], base_rgb[preserved]), "Untargeted occupied RGB changed")
    require(np.array_equal(output[new_target], target_rgb[new_target]), "Target occupied RGB changed")

    report = {
        "schema": SCHEMA,
        "status": "passed",
        "size": [int(base_piece.shape[1]), int(base_piece.shape[0])],
        "target_piece_indices": [int(value) for value in sorted(set(target_piece_indices))],
        "old_target_occupied_texels": int(old_target.sum()),
        "new_target_occupied_texels": int(new_target.sum()),
        "preserved_occupied_texels": int(preserved.sum()),
        "padded_texels": int((padded & (new_piece == 0)).sum()),
        "untargeted_owner_map_exact": True,
        "untargeted_occupied_rgb_byte_exact": True,
        "target_bake_did_not_own_untargeted_texels": True,
        "owner_maps_regenerated_from_new_layout": True,
        "gutter_algorithm": "opencv_labelled_chessboard_nearest_donor_no_average",
        "gutter_padding_pixels": int(padding),
        "cross_piece_or_chart_average": False,
    }
    diagnostics = {
        "piece_owner_id": new_piece.astype(np.uint16, copy=False),
        "chart_owner_id": new_chart.astype(np.uint16, copy=False),
        "occupied_mask": (new_piece > 0).astype(np.uint8) * 255,
        "gutter_piece_owner_id": gutter_piece,
        "gutter_chart_owner_id": gutter_chart,
        "padded_mask": padded.astype(np.uint8) * 255,
    }
    return output, report, diagnostics


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=MODE_STRICT_PATCH,
        help="Keep the original strict target-only path or prepare full touched-page repacks.",
    )
    parser.add_argument("--reunwrap-run", type=Path)
    parser.add_argument("--validate-patches", action="store_true")
    parser.add_argument("--write-plan", type=Path)
    args = parser.parse_args()

    patch_report = None
    if args.validate_patches:
        require(args.reunwrap_run is not None, "--validate-patches requires --reunwrap-run")
        patch_report = validate_reunwrap_run(args.reunwrap_run)
    plan = build_plan(args.reunwrap_run, mode=args.mode)
    if patch_report is not None:
        plan["reunwrap_validation"] = patch_report
        plan["status"] = (
            "local_patches_validated_strict_final_placement_still_required"
            if args.mode == MODE_STRICT_PATCH
            else "local_patches_validated_full_page_repack_still_required"
        )
    if args.write_plan:
        _write_json_exclusive(args.write_plan.expanduser().resolve(), plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
