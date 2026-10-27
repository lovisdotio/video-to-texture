#!/usr/bin/env python3
"""Strict local contract helpers for the targeted Seedream V10 run.

This module performs no network access.  It verifies every input byte and
decoded pixel hash before a caller is allowed to construct a Fal request.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "inputs_2k"
DEFAULT_RUNS_ROOT = SCRIPT_DIR / "runs"

INPUT_SCHEMA = "car-city.ferrari-targeted-seedream-inputs/v10"
RUN_CONTRACT_SCHEMA = "car-city.ferrari-targeted-seedream-run-contract/v10"
RUN_STATE_SCHEMA = "car-city.ferrari-targeted-seedream-run-state/v10"
RESULT_MANIFEST_SCHEMA = "car-city.ferrari-targeted-seedream-results/v10"
MODEL = "bytedance/seedream/v5/pro/edit"
TARGET_SIZE = 2048
NETWORK_GO_TOKEN = "SEEDREAM_V10_NETWORK_GO"

EXPECTED_PAGE_ORDER = (
    "body_hero_8k",
    "exterior_opaque",
    "glass",
    "lights_translucent",
    "running_metal",
    "tires",
    "cockpit",
)
EXPECTED_CANDIDATE_ORDER = (
    "body_hero_8k",
    "exterior_opaque",
    "glass",
    "lights_translucent",
    "running_metal",
    "tires",
    "cockpit_leather",
    "cockpit_plastics",
    "cockpit_carbon",
)
COCKPIT_SOURCE_IDS = {
    "cockpit_leather": (19, 36, 39),
    "cockpit_plastics": (17, 18, 28, 35),
    "cockpit_carbon": (7, 8, 33),
}
COCKPIT_EXCLUDED_SOURCE_IDS = (9, 23, 34, 37)
COCKPIT_TRUE_METAL_SOURCE_IDS = (23, 37)

ARTIFACT_KEYS = {
    "path",
    "semantic",
    "file_sha256",
    "decoded_pixel_sha256",
    "shape",
    "dtype",
    "bytes",
}
CANDIDATE_KEYS = {
    "candidate_id",
    "page",
    "working_texture",
    "sparse_layout",
    "occupancy_mask",
    "chart_owner",
    "material_mask",
    "source_object_ids",
    "excluded_source_object_ids",
    "gates",
    "material_group",
}


@dataclass(frozen=True)
class ValidatedArtifact:
    label: str
    path: Path
    record: dict[str, Any]

    @property
    def file_sha256(self) -> str:
        return str(self.record["file_sha256"])


@dataclass(frozen=True)
class ValidatedInputs:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    pages: dict[str, dict[str, Any]]
    candidates: dict[str, dict[str, Any]]
    artifacts: dict[str, ValidatedArtifact]
    audit: dict[str, Any]


@dataclass(frozen=True)
class ReferenceSpec:
    key: str
    path: Path
    file_sha256: str
    role: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(image is not None, f"PNG decode failed: {path}")
    return image


def _safe_artifact_path(root: Path, relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative != "", f"{label}: invalid path")
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"{label}: path must be relative to inputs root: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    require(resolved.is_relative_to(resolved_root), f"{label}: path escapes inputs root: {relative}")
    return resolved


def validate_artifact(root: Path, record: Mapping[str, Any], label: str) -> ValidatedArtifact:
    require(isinstance(record, Mapping), f"{label}: artifact must be an object")
    require(set(record) == ARTIFACT_KEYS, f"{label}: artifact keys drift: {sorted(set(record) ^ ARTIFACT_KEYS)}")
    normalized = dict(record)
    path = _safe_artifact_path(root, normalized["path"], label)
    require(path.is_file(), f"{label}: missing artifact: {path}")
    require(path.stat().st_size == normalized["bytes"], f"{label}: byte-size drift: {path}")
    actual_file_hash = sha256_file(path)
    require(actual_file_hash == normalized["file_sha256"], f"{label}: file SHA256 drift: {path}")
    image = read_image(path)
    require(list(image.shape) == normalized["shape"], f"{label}: decoded shape drift: {path}")
    require(str(image.dtype) == normalized["dtype"], f"{label}: decoded dtype drift: {path}")
    actual_pixel_hash = sha256_array(image)
    require(actual_pixel_hash == normalized["decoded_pixel_sha256"], f"{label}: decoded pixel SHA256 drift: {path}")
    require(isinstance(normalized["semantic"], str) and normalized["semantic"], f"{label}: missing semantic")
    return ValidatedArtifact(label=label, path=path, record=normalized)


def artifact_fingerprint(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[key] if key != "shape" else tuple(record[key]) for key in sorted(ARTIFACT_KEYS))


def _binary_mask(path: Path, label: str) -> np.ndarray:
    mask = read_image(path)
    require(mask.shape == (TARGET_SIZE, TARGET_SIZE), f"{label}: mask must be {TARGET_SIZE}x{TARGET_SIZE}, got {mask.shape}")
    require(mask.dtype == np.uint8, f"{label}: mask must be uint8, got {mask.dtype}")
    values = np.unique(mask)
    require(bool(np.all(np.isin(values, (0, 255)))), f"{label}: mask is not binary 0/255: {values.tolist()}")
    return mask > 0


def _owner_support(path: Path, label: str) -> np.ndarray:
    owner = read_image(path)
    require(owner.shape == (TARGET_SIZE, TARGET_SIZE), f"{label}: owner map must be {TARGET_SIZE}x{TARGET_SIZE}")
    require(owner.dtype == np.uint16, f"{label}: owner map must be uint16, got {owner.dtype}")
    return owner > 0


def validate_inputs(input_root: Path) -> ValidatedInputs:
    root = input_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == INPUT_SCHEMA, f"Input schema mismatch: {manifest.get('schema')!r}")
    require(manifest.get("status") == "passed", f"Input manifest is not passed: {manifest.get('status')!r}")
    require(tuple(manifest.get("page_order", ())) == EXPECTED_PAGE_ORDER, "V10 page_order drift")
    require(tuple(manifest.get("candidate_order", ())) == EXPECTED_CANDIDATE_ORDER, "V10 candidate_order drift")

    page_rows = manifest.get("pages")
    require(isinstance(page_rows, list), "Input manifest pages must be a list")
    require(tuple(row.get("page") for row in page_rows) == EXPECTED_PAGE_ORDER, "V10 pages order/content drift")
    pages = {row["page"]: row for row in page_rows}

    candidate_rows = manifest.get("candidates")
    require(isinstance(candidate_rows, list), "Input manifest candidates must be a list")
    require(tuple(row.get("candidate_id") for row in candidate_rows) == EXPECTED_CANDIDATE_ORDER, "V10 candidate order/content drift")
    candidates = {row["candidate_id"]: row for row in candidate_rows}

    artifacts: dict[str, ValidatedArtifact] = {}
    page_material_fingerprints: dict[str, set[tuple[Any, ...]]] = {}
    for page_id in EXPECTED_PAGE_ORDER:
        page = pages[page_id]
        require(isinstance(page, dict), f"Page row is not an object: {page_id}")
        for key in ("native_resolution", "resolution", "scale_factor", "source", "counts", "outputs", "material_masks", "gates"):
            require(key in page, f"Page {page_id}: missing {key}")
        require(list(page["resolution"]) == [TARGET_SIZE, TARGET_SIZE], f"Page {page_id}: resolution is not 2K")
        outputs = page["outputs"]
        require(isinstance(outputs, dict), f"Page {page_id}: outputs must be an object")
        required_outputs = {"working_texture", "sparse_layout", "occupancy_mask", "chart_owner"}
        required_outputs.add("object_owner")
        require(required_outputs.issubset(outputs), f"Page {page_id}: missing outputs {sorted(required_outputs - set(outputs))}")
        for key, record in outputs.items():
            label = f"pages.{page_id}.outputs.{key}"
            artifacts[label] = validate_artifact(root, record, label)
        material_masks = page["material_masks"]
        require(isinstance(material_masks, dict) and material_masks, f"Page {page_id}: material_masks must be non-empty")
        fingerprints: set[tuple[Any, ...]] = set()
        for key, group in material_masks.items():
            label = f"pages.{page_id}.material_masks.{key}"
            require(isinstance(group, dict), f"{label}: group must be an object")
            require(
                set(group) == {"artifact", "eligible_for_generation", "owner_values", "source_object_ids"},
                f"{label}: group keys drift",
            )
            require(isinstance(group["eligible_for_generation"], bool), f"{label}: invalid eligibility")
            require(isinstance(group["owner_values"], list), f"{label}: owner_values must be a list")
            require(isinstance(group["source_object_ids"], list), f"{label}: source_object_ids must be a list")
            artifacts[label] = validate_artifact(root, group["artifact"], label)
            fingerprints.add(artifact_fingerprint(group["artifact"]))
        page_material_fingerprints[page_id] = fingerprints

        occupancy_artifact = artifacts[f"pages.{page_id}.outputs.occupancy_mask"]
        occupancy = _binary_mask(occupancy_artifact.path, f"Page {page_id} occupancy")
        chart_support = _owner_support(artifacts[f"pages.{page_id}.outputs.chart_owner"].path, f"Page {page_id} chart_owner")
        require(np.array_equal(occupancy, chart_support), f"Page {page_id}: chart-owner support differs from occupancy")
        object_support = _owner_support(artifacts[f"pages.{page_id}.outputs.object_owner"].path, f"Page {page_id} object_owner")
        require(np.array_equal(occupancy, object_support), f"Page {page_id}: object-owner support differs from occupancy")
        sparse = read_image(artifacts[f"pages.{page_id}.outputs.sparse_layout"].path)
        working = read_image(artifacts[f"pages.{page_id}.outputs.working_texture"].path)
        require(sparse.shape == working.shape == (TARGET_SIZE, TARGET_SIZE, 3), f"Page {page_id}: RGB context is not 2048x2048x3")
        require(sparse.dtype == working.dtype == np.uint8, f"Page {page_id}: RGB context must be uint8")
        require(not bool(np.any(sparse[~occupancy])), f"Page {page_id}: sparse layout is nonzero outside occupancy")

    for candidate_id in EXPECTED_CANDIDATE_ORDER:
        row = candidates[candidate_id]
        require(isinstance(row, dict), f"Candidate row is not an object: {candidate_id}")
        require(set(row) == CANDIDATE_KEYS, f"Candidate {candidate_id}: keys drift: {sorted(set(row) ^ CANDIDATE_KEYS)}")
        page_id = row["page"]
        require(page_id in pages, f"Candidate {candidate_id}: unknown page {page_id!r}")
        expected_page = "cockpit" if candidate_id.startswith("cockpit_") else candidate_id
        require(page_id == expected_page, f"Candidate {candidate_id}: expected page {expected_page}, got {page_id}")
        require(isinstance(row["source_object_ids"], list), f"Candidate {candidate_id}: source_object_ids must be a list")
        require(isinstance(row["excluded_source_object_ids"], list), f"Candidate {candidate_id}: excluded_source_object_ids must be a list")
        require(isinstance(row["gates"], dict), f"Candidate {candidate_id}: gates must be an object")
        require(isinstance(row["material_group"], str) and row["material_group"], f"Candidate {candidate_id}: material_group is invalid")
        for key in ("working_texture", "sparse_layout", "occupancy_mask", "chart_owner", "material_mask"):
            label = f"candidates.{candidate_id}.{key}"
            artifacts[label] = validate_artifact(root, row[key], label)
        page_outputs = pages[page_id]["outputs"]
        for key in ("working_texture", "sparse_layout", "occupancy_mask", "chart_owner"):
            require(
                artifact_fingerprint(row[key]) == artifact_fingerprint(page_outputs[key]),
                f"Candidate {candidate_id}: {key} differs from page context",
            )
        require(
            artifact_fingerprint(row["material_mask"]) in page_material_fingerprints[page_id],
            f"Candidate {candidate_id}: material_mask is not declared by page {page_id}",
        )

    for candidate_id, expected_ids in COCKPIT_SOURCE_IDS.items():
        row = candidates[candidate_id]
        require(tuple(row["source_object_ids"]) == expected_ids, f"Candidate {candidate_id}: source object IDs drift")
        require(
            tuple(row["excluded_source_object_ids"]) == COCKPIT_TRUE_METAL_SOURCE_IDS,
            f"Candidate {candidate_id}: excluded true-metal cockpit IDs drift",
        )

    mask_counts: dict[str, int] = {}
    for candidate_id in EXPECTED_CANDIDATE_ORDER[:6]:
        row = candidates[candidate_id]
        occupancy = _binary_mask(artifacts[f"candidates.{candidate_id}.occupancy_mask"].path, f"{candidate_id} occupancy")
        target = _binary_mask(artifacts[f"candidates.{candidate_id}.material_mask"].path, f"{candidate_id} material mask")
        require(np.array_equal(target, occupancy), f"Candidate {candidate_id}: full-page material mask differs from occupancy")
        mask_counts[candidate_id] = int(target.sum())

    cockpit_occupancy = _binary_mask(
        artifacts["pages.cockpit.outputs.occupancy_mask"].path,
        "cockpit occupancy",
    )
    cockpit_masks: dict[str, np.ndarray] = {}
    for candidate_id in COCKPIT_SOURCE_IDS:
        mask = _binary_mask(artifacts[f"candidates.{candidate_id}.material_mask"].path, f"{candidate_id} material mask")
        require(not bool(np.any(mask & ~cockpit_occupancy)), f"Candidate {candidate_id}: mask escapes cockpit occupancy")
        cockpit_masks[candidate_id] = mask
        mask_counts[candidate_id] = int(mask.sum())
    cockpit_ids = tuple(COCKPIT_SOURCE_IDS)
    for index, left_id in enumerate(cockpit_ids):
        for right_id in cockpit_ids[index + 1 :]:
            require(not bool(np.any(cockpit_masks[left_id] & cockpit_masks[right_id])), f"Cockpit masks overlap: {left_id}/{right_id}")

    cockpit_materials = pages["cockpit"]["material_masks"]
    require("excluded" in cockpit_materials, "Cockpit page lacks excluded mask")
    require(cockpit_materials["excluded"]["eligible_for_generation"] is False, "Cockpit excluded mask became eligible")
    require(
        tuple(cockpit_materials["excluded"]["source_object_ids"]) == COCKPIT_EXCLUDED_SOURCE_IDS,
        "Cockpit excluded mask source IDs drift",
    )
    excluded_artifact = artifacts["pages.cockpit.material_masks.excluded"]
    cockpit_excluded = _binary_mask(excluded_artifact.path, "cockpit excluded mask")
    require(not bool(np.any(cockpit_excluded & ~cockpit_occupancy)), "Cockpit exclusion escapes occupancy")
    target_union = np.zeros_like(cockpit_occupancy)
    for mask in cockpit_masks.values():
        target_union |= mask
    require(not bool(np.any(target_union & cockpit_excluded)), "Cockpit target masks overlap exclusion")
    require(np.array_equal(target_union | cockpit_excluded, cockpit_occupancy), "Cockpit candidates plus exclusion do not partition occupancy")
    mask_counts["cockpit_excluded"] = int(cockpit_excluded.sum())

    audit = {
        "schema": "car-city.ferrari-targeted-seedream-input-audit/v10",
        "status": "passed",
        "manifest_sha256": sha256_file(manifest_path),
        "page_order": list(EXPECTED_PAGE_ORDER),
        "candidate_order": list(EXPECTED_CANDIDATE_ORDER),
        "validated_artifact_records": len(artifacts),
        "mask_texels": mask_counts,
        "cockpit_partition": {
            "candidate_masks_pairwise_disjoint": True,
            "candidate_union_plus_exclusion_equals_occupancy": True,
            "true_metal_source_ids_excluded": [23, 37],
        },
    }
    return ValidatedInputs(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=audit["manifest_sha256"],
        manifest=manifest,
        pages=pages,
        candidates=candidates,
        artifacts=artifacts,
        audit=audit,
    )


REFERENCE_SPECS: dict[str, ReferenceSpec] = {
    "around_front": ReferenceSpec(
        "around_front",
        V8_ROOT / "targeted_final_v9/outputs/renders/ferrari_targeted_reunwrap_seedream_v9_around5/front.png",
        "57afe643307f69458df086ae0ae62636a4365680c523332a8e7be197a8608e14",
        "targeted V9 front render; component identity only",
    ),
    "around_front_left": ReferenceSpec(
        "around_front_left",
        V8_ROOT / "targeted_final_v9/outputs/renders/ferrari_targeted_reunwrap_seedream_v9_around5/front_left.png",
        "6bf053026006aec26856a2fa075600564002c88563c9fcd3339a3b12d2153f1d",
        "targeted V9 front-left render; component identity only",
    ),
    "around_front_right": ReferenceSpec(
        "around_front_right",
        V8_ROOT / "targeted_final_v9/outputs/renders/ferrari_targeted_reunwrap_seedream_v9_around5/front_right.png",
        "930733b48265c2d915126dd71794179f8b228d9a7a180eea4ad8d2f5a726dca9",
        "targeted V9 front-right render; component identity only",
    ),
    "around_rear_left": ReferenceSpec(
        "around_rear_left",
        V8_ROOT / "targeted_final_v9/outputs/renders/ferrari_targeted_reunwrap_seedream_v9_around5/rear_left.png",
        "0de02cce2beca66ea43c2b6e6e45655282e2e7d6a17f8a61493ce40acf8847ca",
        "targeted V9 rear-left render; component identity only",
    ),
    "around_rear_right": ReferenceSpec(
        "around_rear_right",
        V8_ROOT / "targeted_final_v9/outputs/renders/ferrari_targeted_reunwrap_seedream_v9_around5/rear_right.png",
        "1960f3a0165c28d6ce2a1688a71cd164dfc47b30bb0d002414163fd98ed94e79",
        "targeted V9 rear-right render; component identity only",
    ),
    "original_front": ReferenceSpec(
        "original_front",
        V8_ROOT / "integration_v8/final_assembler_v4/outputs/renders/current_v4_preview_visual_review_v1/front.png",
        "2450e976abca8fde624dab51780d9b2f34271c44bebd97990104f9bfc05b116f",
        "original front render; geometry/artwork identity only",
    ),
    "original_rear": ReferenceSpec(
        "original_rear",
        V8_ROOT / "integration_v8/final_assembler_v4/outputs/renders/current_v4_preview_visual_review_v1/rear.png",
        "9402769d7e20ef0265585a89aa6d3dcf75a700a5b805e938c6313a340a6ff5f2",
        "original rear render; geometry/artwork identity only",
    ),
    "original_side": ReferenceSpec(
        "original_side",
        V8_ROOT / "integration_v8/final_assembler_v4/outputs/renders/current_v4_preview_visual_review_v1/side.png",
        "b650690e9115846435cf386b70580dd804b44b8589cca6f8c0309aaa6a292041",
        "original side render; geometry/artwork identity only",
    ),
    "original_hero": ReferenceSpec(
        "original_hero",
        V8_ROOT / "integration_v8/final_assembler_v4/outputs/renders/current_v4_preview_visual_review_v1/hero.png",
        "2eb845961cfef58a03a1b29d24250e76d42e3aeff0c5a957d46c5cf1b0f50795",
        "original hero render; geometry/material identity only, contaminated lighting",
    ),
    "wheel_close": ReferenceSpec(
        "wheel_close",
        V8_ROOT / "integration_v8/final_assembler_v4/outputs/renders/seedream_texture_cleanup_wheel_v1/seedream_wheel_cleanup_v1.png",
        "666ad3211e603eaad8bd78cf6ea41cb6bb2dcbf86e68d566ccbd6a095c188daf",
        "approved five-spoke wheel identity reference; lighting is not color evidence",
    ),
}

REFERENCE_POLICY: dict[str, tuple[str, ...]] = {
    "body_hero_8k": (
        "around_front",
        "around_front_left",
        "around_front_right",
        "around_rear_left",
        "around_rear_right",
        "original_front",
        "original_rear",
    ),
    "exterior_opaque": (
        "around_front",
        "around_front_left",
        "around_front_right",
        "around_rear_left",
        "around_rear_right",
        "original_front",
        "original_rear",
    ),
    "glass": ("around_front_left", "around_front_right", "around_rear_left", "around_rear_right", "original_hero"),
    "lights_translucent": (
        "around_front",
        "around_front_left",
        "around_front_right",
        "around_rear_left",
        "around_rear_right",
        "original_front",
        "original_rear",
    ),
    "running_metal": (
        "around_front_left",
        "around_front_right",
        "around_rear_left",
        "around_rear_right",
        "wheel_close",
        "original_side",
    ),
    "tires": ("around_front_left", "around_front_right", "around_rear_left", "around_rear_right", "wheel_close"),
    "cockpit_leather": ("around_front", "around_front_left", "around_front_right", "original_front", "original_hero"),
    "cockpit_plastics": ("around_front", "around_front_left", "around_front_right", "original_front", "original_hero"),
    "cockpit_carbon": ("around_front", "around_front_left", "around_front_right", "original_front", "original_hero"),
}


def validate_references(candidate_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    keys: list[str] = []
    for candidate_id in candidate_ids:
        require(candidate_id in REFERENCE_POLICY, f"Missing reference policy: {candidate_id}")
        for key in REFERENCE_POLICY[candidate_id]:
            if key not in keys:
                keys.append(key)
    records: dict[str, dict[str, Any]] = {}
    for key in keys:
        spec = REFERENCE_SPECS[key]
        require(spec.path.is_file(), f"Missing pinned reference: {spec.path}")
        actual_hash = sha256_file(spec.path)
        require(actual_hash == spec.file_sha256, f"Pinned reference SHA256 drift: {spec.path}")
        image = read_image(spec.path)
        require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] in (3, 4), f"Bad reference raster: {spec.path}")
        records[key] = {
            "path": str(spec.path),
            "role": spec.role,
            "file_sha256": actual_hash,
            "decoded_pixel_sha256": sha256_array(image),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "bytes": spec.path.stat().st_size,
        }
    return records


CANDIDATE_GUIDANCE = {
    "body_hero_8k": (
        "Preserve the exact existing ukiyo-e/Hokusai identity, wave and illustration contours, blue, cream and restrained gold color regions, and their placement. "
        "Clean only the paint base color; do not redraw, modernize, simplify or replace the artwork."
    ),
    "exterior_opaque": (
        "Preserve every panel/material identity and existing artwork fragment: painted detail, black grilles, dark trim, wipers and restrained chrome. "
        "Produce clean intrinsic paint/trim/grille albedo with no lighting baked into it."
    ),
    "glass": (
        "Every target texel is a real glass or optical-cover surface. Produce coherent neutral clear-to-smoked glass tint albedo with subtle intrinsic tone. "
        "Do not paint opaque white glare, mirrored scenery, sky, lamps, reflections, or empty black wedges."
    ),
    "lights_translucent": (
        "Preserve the exact lamp islands and their existing identities: saturated red rear lens base color, clean cream-white LED material, smoked housings and restrained clear optical tint. "
        "No emitted glow, bloom, hotspot, specular streak, scenery, new lamp graphics or hardware."
    ),
    "running_metal": (
        "Preserve the exact five-spoke wheel and every existing wheel/brake/fastener shape. Use intrinsic gunmetal and steel albedo, with the approved restrained champagne-bronze only where the source is already alloy. "
        "Only these true metal target islands may look metallic; do not invent spokes, holes, bolts, badges, geometry or reflections."
    ),
    "tires": (
        "Every target island is tire rubber, including rings and long strips. Use coherent matte dark-charcoal rubber albedo with restrained intrinsic rubber microtexture. "
        "Never produce pure-black void, pale/white rubber, metal, body livery, blue or cream paint, lettering, logos, wheel parts or invented tread graphics."
    ),
    "cockpit_leather": (
        "Every white target-mask texel is leather. Produce matte warm cognac-brown leather albedo with restrained intrinsic grain and preserve existing seam/panel cues. "
        "It must never be grey, silver, chrome, pearlescent, glossy or metallic. True metal, carpet, badge and other cockpit materials are excluded by the mask."
    ),
    "cockpit_plastics": (
        "Every white target-mask texel is interior plastic or Alcantara. Produce matte anthracite plastic/Alcantara albedo with restrained intrinsic microtexture. "
        "It must never be silver, chrome, glossy, reflective or metallic. True metals are excluded by the mask."
    ),
    "cockpit_carbon": (
        "Every white target-mask texel is carbon trim. Produce deep dark satin carbon albedo with a restrained, fine, consistently scaled weave. "
        "Do not turn the weave into bright silver, chrome, mirror reflection, glare or broad grey gradients. True metals are excluded by the mask."
    ),
}


def build_prompt(candidate_id: str, reference_keys: Sequence[str]) -> str:
    require(candidate_id in CANDIDATE_GUIDANCE, f"Unknown prompt candidate: {candidate_id}")
    reference_start = 4
    reference_end = 3 + len(reference_keys)
    reference_listing = "; ".join(
        f"Figure {reference_start + index} is {REFERENCE_SPECS[key].role}"
        for index, key in enumerate(reference_keys)
    )
    return " ".join(
        (
            "Create one lighting-neutral RGB base-color/albedo UV texture atlas, exactly 2048x2048 pixels.",
            "Figure 1 is the authoritative preguttered working atlas; preserve its existing material identity and artwork.",
            "Figure 2 is the authoritative sparse UV layout; its non-black support defines real UV surface texels and the exact island contours.",
            "Figure 3 is the authoritative binary target mask: edit only white pixels. Black pixels are excluded context or padding and are not a black material or empty hole.",
            "Keep every target UV island in exactly the same count, position, orientation, scale, edge contour and internal partition. Do not move, rotate, merge, split, shrink, expand or redesign islands.",
            "Inside the white target mask, every real surface must remain valid non-void albedo; remove black wedges, missing fills, projection smears, color bleeding, jagged seams and stretched artifacts.",
            "This is base color only: bake no highlights, specular reflections, glare, bloom, emitted glow, ambient occlusion, cast or contact shadows, directional light gradients, scenery or environment colors.",
            "Preserve intrinsic pigment, artwork, leather grain, rubber texture or weave only where appropriate. Add no text, logos, labels, geometry, controls, holes or decorative graphics.",
            CANDIDATE_GUIDANCE[candidate_id],
            (
                f"{reference_listing}. These rendered references are component-identity and material-family evidence only. "
                "Their illumination is contaminated: copy none of their silver or pink color casts, highlights, shadows, reflections, glare, sky, road or scenery."
                if reference_keys
                else "There are no rendered references."
            ),
            "If a full square canvas must be returned, keep non-target pixels close to Figure 1; downstream compositing will accept only Figure 3 white pixels.",
            "Return exactly one lossless 2048x2048 RGB PNG with no alpha channel and no presentation border.",
        )
    )


def candidate_local_inputs(validated: ValidatedInputs, candidate_id: str) -> list[dict[str, Any]]:
    row = validated.candidates[candidate_id]
    records: list[dict[str, Any]] = []
    for figure, key in enumerate(("working_texture", "sparse_layout", "material_mask"), start=1):
        artifact = validated.artifacts[f"candidates.{candidate_id}.{key}"]
        records.append(
            {
                "figure": figure,
                "kind": key,
                "path": str(artifact.path),
                "file_sha256": artifact.record["file_sha256"],
                "decoded_pixel_sha256": artifact.record["decoded_pixel_sha256"],
                "shape": artifact.record["shape"],
                "dtype": artifact.record["dtype"],
                "bytes": artifact.record["bytes"],
            }
        )
    return records


def select_candidates(requested: Iterable[str] | None) -> tuple[str, ...]:
    requested_list = list(requested or ())
    if not requested_list:
        return EXPECTED_CANDIDATE_ORDER
    require(len(requested_list) == len(set(requested_list)), "Candidate selection contains duplicates")
    unknown = set(requested_list) - set(EXPECTED_CANDIDATE_ORDER)
    require(not unknown, f"Unknown candidates: {sorted(unknown)}")
    selected = set(requested_list)
    return tuple(candidate_id for candidate_id in EXPECTED_CANDIDATE_ORDER if candidate_id in selected)
