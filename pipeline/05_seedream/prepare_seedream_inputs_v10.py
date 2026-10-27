#!/usr/bin/env python3
"""Build the local, deterministic V10 2K finishing-input package.

The package is derived only from the seven textures bound by the current V9
final manifest.  UV support and chart/object ownership always come from the
validated native diagnostics; generated RGB is never allowed to define UVs.

Exactly nine generation candidates are emitted:

* one full-occupancy candidate for BODY, exterior, glass, lights, running and
  tires;
* three cockpit candidates (leather, plastics and carbon) sharing byte-exact
  working/sparse/occupancy/chart context.

The cockpit candidates and a non-generation exclusion mask form an exact,
disjoint partition of cockpit occupancy.  In particular source objects 23 and
37 (the two real metals) are excluded from generation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inputs_2k"
FINAL_MANIFEST_PATH = (
    V8_ROOT
    / "targeted_final_v9"
    / "manifests"
    / "ferrari_targeted_reunwrap_seedream_v9_final.manifest.json"
)
V5_INPUT_MANIFEST_PATH = V8_ROOT / "seedream_full_texture_v5" / "inputs_2k" / "manifest.json"
V8_BAKE_ROOT = V8_ROOT / "bake_v8" / "diagnostics" / "production_seedvr_rgb_v1" / "extracted"

TARGET_SIZE = 2048
PADDING_PIXELS = 16
PNG_COMPRESSION = 6
ROW_CHUNK = 256
REPORT_SCHEMA = "car-city.ferrari-page-texture-reprojection/v9"
OUTPUT_SCHEMA = "car-city.ferrari-targeted-seedream-inputs/v10"
FINAL_SCHEMA = "car-city.ferrari-targeted-final-manifest/v9"

PAGE_ORDER = (
    "body_hero_8k",
    "exterior_opaque",
    "glass",
    "lights_translucent",
    "running_metal",
    "tires",
    "cockpit",
)

V9_NATIVE_RUNS = {
    "body_hero_8k": "reproject_body_native_v1",
    "exterior_opaque": "reproject_exterior_native_v1",
    "glass": "reproject_glass_native_v1",
    "lights_translucent": "reproject_lights_native_v1",
    "running_metal": "reproject_running_native_v1",
    "cockpit": "reproject_cockpit_native_v2",
}

# Source object IDs, not encoded owner values.  Native object-owner pixels are
# source_object_id + 1; zero remains background.
COCKPIT_PARTITION = {
    "leather": {
        "source_object_ids": (19, 36, 39),
        "eligible_for_generation": True,
        "semantic": "cockpit_leather_and_steering_leather_including_trim_39",
    },
    "plastics": {
        "source_object_ids": (17, 18, 28, 35),
        "eligible_for_generation": True,
        "semantic": "cockpit_plastics_dashboard_panels_and_steering_column",
    },
    "carbon": {
        "source_object_ids": (7, 8, 33),
        "eligible_for_generation": True,
        "semantic": "cockpit_carbon_fibre_carbon_trim_and_steering_carbon",
    },
    "excluded": {
        "source_object_ids": (9, 23, 34, 37),
        "eligible_for_generation": False,
        "semantic": "cockpit_carpet_badge_and_true_metals_excluded_from_generation",
    },
}

CANDIDATE_ORDER = (
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def path_for_manifest(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(V8_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(image is not None, f"PNG decode failed: {path}")
    return image


def write_png_atomic(path: Path, image: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(image)
    ok, encoded = cv2.imencode(
        ".png",
        contiguous,
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
    )
    require(bool(ok), f"PNG encode failed: {path}")
    atomic_write_bytes(path, encoded.tobytes())


def artifact_record(path: Path, expected: np.ndarray, *, root: Path, semantic: str) -> dict[str, Any]:
    reopened = read_image(path)
    require(reopened.shape == expected.shape, f"PNG shape changed after reopen: {path}")
    require(reopened.dtype == expected.dtype, f"PNG dtype changed after reopen: {path}")
    expected_hash = sha256_array(expected)
    decoded_hash = sha256_array(reopened)
    require(decoded_hash == expected_hash, f"PNG pixels changed after reopen: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "semantic": semantic,
        "file_sha256": sha256_file(path),
        "decoded_pixel_sha256": decoded_hash,
        "shape": list(reopened.shape),
        "dtype": str(reopened.dtype),
        "bytes": int(path.stat().st_size),
    }


def emit_png(
    path: Path,
    image: np.ndarray,
    *,
    root: Path,
    semantic: str,
) -> dict[str, Any]:
    write_png_atomic(path, image)
    return artifact_record(path, image, root=root, semantic=semantic)


def parse_hash_inventory(path: Path) -> dict[str, str]:
    require(path.is_file(), f"Missing hash inventory: {path}")
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        parts = raw.split("  ", 1)
        require(len(parts) == 2, f"Malformed hash inventory line {line_number}: {path}")
        digest, relative = parts
        require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), f"Bad SHA-256 on line {line_number}: {path}")
        candidate = Path(relative)
        require(not candidate.is_absolute() and ".." not in candidate.parts, f"Unsafe inventory path: {relative}")
        require(relative not in rows, f"Duplicate inventory path: {relative}")
        rows[relative] = digest
    require(bool(rows), f"Empty hash inventory: {path}")
    return rows


def verify_complete_hash_inventory(root: Path) -> dict[str, Any]:
    inventory_path = root / "hashes.sha256"
    entries = parse_hash_inventory(inventory_path)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "hashes.sha256"
    }
    require(set(entries) == actual, f"Hash inventory file set drift: {root}")
    for relative, expected in sorted(entries.items()):
        require(sha256_file(root / relative) == expected, f"Hash drift: {root / relative}")
    return {
        "path": path_for_manifest(inventory_path),
        "sha256": sha256_file(inventory_path),
        "entries": len(entries),
    }


def nearest_exact(image: np.ndarray) -> np.ndarray:
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(image, (TARGET_SIZE, TARGET_SIZE), interpolation=interpolation)


def resize_rgb(image: np.ndarray) -> np.ndarray:
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, "RGB source must be uint8 HxWx3")
    if image.shape[:2] == (TARGET_SIZE, TARGET_SIZE):
        return image.copy()
    return cv2.resize(image, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


def grow_owned_gutters(
    sparse_bgr: np.ndarray,
    surface_object_owner: np.ndarray,
    surface_chart_owner: np.ndarray,
    *,
    padding: int = PADDING_PIXELS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Copy one deterministic occupied donor into each 16px gutter texel."""

    require(sparse_bgr.shape == (TARGET_SIZE, TARGET_SIZE, 3) and sparse_bgr.dtype == np.uint8, "Bad sparse RGB")
    require(surface_object_owner.shape == surface_chart_owner.shape == (TARGET_SIZE, TARGET_SIZE), "Bad owner shapes")
    require(surface_object_owner.dtype == surface_chart_owner.dtype == np.uint16, "Owners must be uint16")
    occupied = surface_object_owner > 0
    require(np.array_equal(occupied, surface_chart_owner > 0), "Surface owner supports differ")
    require(bool(np.any(occupied)), "Cannot grow gutters from empty occupancy")
    require(not bool(np.any(sparse_bgr[~occupied] != 0)), "Sparse RGB is nonblack outside occupancy")

    source = np.where(occupied, 0, 255).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_C,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    del source
    seed_labels = labels[occupied]
    require(bool(np.all(seed_labels > 0)), "Occupied donors lack distance labels")
    slots = int(labels.max()) + 1
    donor_bgr = np.zeros((slots, 3), dtype=np.uint8)
    donor_object = np.zeros(slots, dtype=np.uint16)
    donor_chart = np.zeros(slots, dtype=np.uint16)
    donor_bgr[seed_labels] = sparse_bgr[occupied]
    donor_object[seed_labels] = surface_object_owner[occupied]
    donor_chart[seed_labels] = surface_chart_owner[occupied]
    del seed_labels

    working = sparse_bgr.copy()
    padded_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    gutter_texels = 0
    copied_object = surface_object_owner.copy()
    copied_chart = surface_chart_owner.copy()
    for y0 in range(0, TARGET_SIZE, ROW_CHUNK):
        y1 = min(TARGET_SIZE, y0 + ROW_CHUNK)
        local_occupied = occupied[y0:y1]
        band = (labels[y0:y1] > 0) & (distance[y0:y1] <= float(padding))
        padded_mask[y0:y1][band] = 255
        gutter = band & ~local_occupied
        if not bool(np.any(gutter)):
            continue
        selected = labels[y0:y1][gutter]
        working[y0:y1][gutter] = donor_bgr[selected]
        copied_object[y0:y1][gutter] = donor_object[selected]
        copied_chart[y0:y1][gutter] = donor_chart[selected]
        gutter_texels += int(gutter.sum())

    del distance, labels, donor_bgr, donor_object, donor_chart
    gc.collect()
    padded = padded_mask > 0
    require(np.array_equal(working[occupied], sparse_bgr[occupied]), "Gutter build changed occupied RGB")
    require(not bool(np.any(working[~padded] != 0)), "Working RGB escaped finite gutter support")
    require(np.array_equal(copied_object[padded] > 0, copied_chart[padded] > 0), "Padded owner supports differ")
    return working, padded_mask, {
        "algorithm": "opencv_DIST_C_labelled_nearest_pixel_donor",
        "padding_pixels_2k": int(padding),
        "occupied_texels": int(occupied.sum()),
        "gutter_texels": int(gutter_texels),
        "padded_texels": int(padded.sum()),
        "cross_domain_averaging": False,
        "rgb_object_chart_copied_from_same_single_donor": True,
    }


def load_final_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    require(FINAL_MANIFEST_PATH.is_file(), f"Missing final manifest: {FINAL_MANIFEST_PATH}")
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    require(manifest.get("schema") == FINAL_SCHEMA, "Final manifest schema drift")
    require(manifest.get("status") == "ready", "Final manifest is not ready")
    textures = manifest.get("textures")
    require(isinstance(textures, dict), "Final manifest lacks textures")
    require(set(PAGE_ORDER) <= set(textures), "Final manifest lacks one or more V10 pages")
    return manifest, {
        "path": path_for_manifest(FINAL_MANIFEST_PATH),
        "sha256": sha256_file(FINAL_MANIFEST_PATH),
        "schema": FINAL_SCHEMA,
        "status": "ready",
    }


def validate_texture_source(final_manifest: Mapping[str, Any], page_id: str) -> tuple[Path, dict[str, Any]]:
    row = final_manifest["textures"][page_id]
    path = Path(str(row["path"])).expanduser().resolve()
    require(path.is_file(), f"Missing current final texture: {path}")
    digest = sha256_file(path)
    require(digest == row.get("sha256"), f"Current final texture SHA drift: {page_id}")
    size = row.get("size")
    require(isinstance(size, list) and len(size) == 2 and size[0] == size[1], f"Bad final texture size: {page_id}")
    return path, {
        "path": path_for_manifest(path),
        "sha256": digest,
        "role": row.get("role"),
        "native_resolution": list(size),
    }


def validate_report_artifact(run_dir: Path, report: Mapping[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    row = report.get("artifacts", {}).get(name)
    require(isinstance(row, dict), f"Missing report artifact {name}: {run_dir}")
    path = run_dir / str(row.get("path", ""))
    require(path.is_file(), f"Missing report artifact file: {path}")
    require(sha256_file(path) == row.get("file_sha256"), f"Report artifact SHA drift: {path}")
    return path, dict(row)


def load_verified_image(path: Path, record: Mapping[str, Any] | None = None) -> np.ndarray:
    image = read_image(path)
    if record is not None:
        require(list(image.shape) == record.get("shape"), f"Decoded shape drift: {path}")
        require(str(image.dtype) == record.get("dtype"), f"Decoded dtype drift: {path}")
        require(sha256_array(image) == record.get("decoded_pixel_sha256"), f"Decoded pixel SHA drift: {path}")
    return image


def load_v9_support(page_id: str, expected_native: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    run_name = V9_NATIVE_RUNS[page_id]
    run_dir = V8_ROOT / "targeted_rebake_v9" / "runs" / run_name
    require(run_dir.is_dir(), f"Missing native reprojection run: {run_dir}")
    inventory = verify_complete_hash_inventory(run_dir)
    report_path = run_dir / "reproject_page_texture_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("schema") == REPORT_SCHEMA, f"Native report schema drift: {page_id}")
    require(report.get("status") == "passed" and report.get("mode") == "native", f"Native report did not pass: {page_id}")
    require(report.get("page_id") == page_id, f"Native report page drift: {page_id}")
    require(report.get("output_resolution") == [expected_native, expected_native], f"Native report resolution drift: {page_id}")
    gates = report.get("gates")
    require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()), f"Native report gates failed: {page_id}")

    occupancy_path, occupancy_record = validate_report_artifact(run_dir, report, "occupancy_mask")
    chart_path, chart_record = validate_report_artifact(run_dir, report, "chart_owner")
    object_path, object_record = validate_report_artifact(run_dir, report, "object_owner")

    occupancy_native = load_verified_image(occupancy_path, occupancy_record)
    require(occupancy_native.shape == (expected_native, expected_native) and occupancy_native.dtype == np.uint8, f"Bad native occupancy: {page_id}")
    require(set(np.unique(occupancy_native).tolist()).issubset({0, 255}), f"Nonbinary native occupancy: {page_id}")
    occupancy_2k = nearest_exact(occupancy_native) > 0
    del occupancy_native

    chart_native = load_verified_image(chart_path, chart_record)
    require(chart_native.shape == (expected_native, expected_native) and chart_native.dtype == np.uint16, f"Bad native chart owner: {page_id}")
    native_chart_ids = np.unique(chart_native[chart_native > 0])
    chart_2k = nearest_exact(chart_native).astype(np.uint16, copy=False)
    del chart_native

    object_native = load_verified_image(object_path, object_record)
    require(object_native.shape == (expected_native, expected_native) and object_native.dtype == np.uint16, f"Bad native object owner: {page_id}")
    native_object_values = np.unique(object_native[object_native > 0])
    object_2k = nearest_exact(object_native).astype(np.uint16, copy=False)
    del object_native
    gc.collect()

    require(np.array_equal(occupancy_2k, chart_2k > 0), f"2K chart support differs from nearest occupancy: {page_id}")
    require(np.array_equal(occupancy_2k, object_2k > 0), f"2K object support differs from nearest occupancy: {page_id}")
    require(set(np.unique(chart_2k[chart_2k > 0]).tolist()) <= set(native_chart_ids.tolist()), f"2K chart owner invented IDs: {page_id}")
    require(set(np.unique(object_2k[object_2k > 0]).tolist()) <= set(native_object_values.tolist()), f"2K object owner invented IDs: {page_id}")

    return (
        occupancy_2k.astype(np.uint8) * 255,
        chart_2k,
        object_2k,
        {
            "kind": "validated_v9_native_reprojection_support",
            "run": path_for_manifest(run_dir),
            "report": {
                "path": path_for_manifest(report_path),
                "sha256": sha256_file(report_path),
            },
            "hash_inventory": inventory,
            "artifacts": {
                "occupancy_mask": {"path": path_for_manifest(occupancy_path), "sha256": sha256_file(occupancy_path)},
                "chart_owner": {"path": path_for_manifest(chart_path), "sha256": sha256_file(chart_path)},
                "object_owner": {"path": path_for_manifest(object_path), "sha256": sha256_file(object_path)},
            },
            "native_chart_ids": int(len(native_chart_ids)),
            "nearest_2k_chart_ids": int(len(np.unique(chart_2k[chart_2k > 0]))),
            "native_object_values": [int(value) for value in native_object_values],
            "nearest_2k_object_values": [int(value) for value in np.unique(object_2k[object_2k > 0])],
        },
    )


def load_tires_support(expected_native: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require(V5_INPUT_MANIFEST_PATH.is_file(), f"Missing V5 inputs manifest: {V5_INPUT_MANIFEST_PATH}")
    v5 = json.loads(V5_INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    require(v5.get("status") == "ready", "V5 input manifest is not ready")
    tire_rows = [row for row in v5.get("pages", []) if row.get("page") == "tires"]
    require(len(tire_rows) == 1, "V5 input manifest tire row drift")
    tire_row = tire_rows[0]
    require(tire_row.get("native_size") == expected_native, "V5 tire native size drift")
    occupancy_input = tire_row["inputs"]["occupied_mask"]
    chart_input = tire_row["inputs"]["chart_owner"]
    occupancy_path = Path(str(occupancy_input["path"])).resolve()
    chart_path = Path(str(chart_input["path"])).resolve()
    piece_path = V8_BAKE_ROOT / "diagnostics" / "tires" / "piece_owner_id.png"
    inventory_path = V8_BAKE_ROOT / "hashes.sha256"
    inventory = parse_hash_inventory(inventory_path)

    selected = {
        "diagnostics/tires/occupied_mask.png": (occupancy_path, str(occupancy_input["sha256"])),
        "diagnostics/tires/chart_owner_id.png": (chart_path, str(chart_input["sha256"])),
        "diagnostics/tires/piece_owner_id.png": (piece_path, inventory.get("diagnostics/tires/piece_owner_id.png", "")),
    }
    for relative, (path, expected) in selected.items():
        require(path.is_file(), f"Missing tire support artifact: {path}")
        require(expected == inventory.get(relative), f"Tire source inventory disagreement: {relative}")
        require(sha256_file(path) == expected, f"Tire support SHA drift: {path}")

    occupancy_native = read_image(occupancy_path)
    chart_native = read_image(chart_path)
    piece_native = read_image(piece_path)
    require(occupancy_native.shape == chart_native.shape == piece_native.shape == (expected_native, expected_native), "Tire support shape drift")
    require(occupancy_native.dtype == np.uint8 and chart_native.dtype == piece_native.dtype == np.uint16, "Tire support dtype drift")
    require(set(np.unique(occupancy_native).tolist()).issubset({0, 255}), "Tire occupancy is nonbinary")
    occupied_native = occupancy_native > 0
    require(np.array_equal(occupied_native, chart_native > 0), "Tire chart support differs")
    require(np.array_equal(occupied_native, piece_native > 0), "Tire piece support differs")
    native_chart_ids = np.unique(chart_native[chart_native > 0])

    occupancy_2k = nearest_exact(occupancy_native) > 0
    chart_2k = nearest_exact(chart_native).astype(np.uint16, copy=False)
    piece_2k = nearest_exact(piece_native).astype(np.uint16, copy=False)
    del occupancy_native, chart_native, piece_native, occupied_native

    # Production piece-owner values 53..56 correspond to tire source objects
    # 40..43.  Normalize explicitly to the V9 object-owner convention.
    piece_to_object_value = {53: 41, 54: 42, 55: 43, 56: 44}
    object_2k = np.zeros_like(piece_2k, dtype=np.uint16)
    for piece_value, object_value in piece_to_object_value.items():
        object_2k[piece_2k == piece_value] = object_value
    observed_piece_values = set(np.unique(piece_2k[piece_2k > 0]).tolist())
    require(observed_piece_values == set(piece_to_object_value), f"Tire piece-owner mapping drift: {sorted(observed_piece_values)}")
    del piece_2k
    require(np.array_equal(occupancy_2k, chart_2k > 0), "2K tire chart support differs")
    require(np.array_equal(occupancy_2k, object_2k > 0), "2K normalized tire object support differs")

    return (
        occupancy_2k.astype(np.uint8) * 255,
        chart_2k,
        object_2k,
        {
            "kind": "v8_production_support_preserved_by_v9_tire_passthrough",
            "v5_input_manifest": {
                "path": path_for_manifest(V5_INPUT_MANIFEST_PATH),
                "sha256": sha256_file(V5_INPUT_MANIFEST_PATH),
            },
            "hash_inventory": {
                "path": path_for_manifest(inventory_path),
                "sha256": sha256_file(inventory_path),
                "selected_entries_verified": len(selected),
            },
            "artifacts": {
                key.rsplit("/", 1)[-1].removesuffix(".png"): {
                    "path": path_for_manifest(path),
                    "sha256": expected,
                }
                for key, (path, expected) in selected.items()
            },
            "piece_owner_normalization": {
                str(piece): value for piece, value in sorted(piece_to_object_value.items())
            },
            "native_chart_ids": int(len(native_chart_ids)),
            "nearest_2k_chart_ids": int(len(np.unique(chart_2k[chart_2k > 0]))),
            "nearest_2k_object_values": [int(value) for value in np.unique(object_2k[object_2k > 0])],
        },
    )


def make_binary_owner_mask(object_owner: np.ndarray, source_object_ids: tuple[int, ...]) -> np.ndarray:
    owner_values = np.asarray([object_id + 1 for object_id in source_object_ids], dtype=np.uint16)
    return np.isin(object_owner, owner_values).astype(np.uint8) * 255


def prepare_page(
    *,
    page_id: str,
    final_manifest: Mapping[str, Any],
    staging_root: Path,
) -> dict[str, Any]:
    texture_path, texture_provenance = validate_texture_source(final_manifest, page_id)
    native_resolution = int(texture_provenance["native_resolution"][0])

    texture_native = read_image(texture_path)
    require(texture_native.shape == (native_resolution, native_resolution, 3) and texture_native.dtype == np.uint8, f"Current final texture decode drift: {page_id}")
    final_2k = resize_rgb(texture_native)
    del texture_native
    gc.collect()

    if page_id == "tires":
        occupancy, chart_owner, object_owner, support_provenance = load_tires_support(native_resolution)
    else:
        occupancy, chart_owner, object_owner, support_provenance = load_v9_support(page_id, native_resolution)

    occupied = occupancy > 0
    require(np.array_equal(occupied, chart_owner > 0), f"Chart support mismatch: {page_id}")
    require(np.array_equal(occupied, object_owner > 0), f"Object support mismatch: {page_id}")
    sparse = np.zeros_like(final_2k)
    np.copyto(sparse, final_2k, where=occupied[:, :, None])
    del final_2k
    working, padded_mask, gutter_report = grow_owned_gutters(sparse, object_owner, chart_owner)
    padded = padded_mask > 0

    page_dir = staging_root / page_id
    outputs = {
        "working_texture": emit_png(
            page_dir / f"{page_id}_working_preguttered_2k.png",
            working,
            root=staging_root,
            semantic="current_final_v9_rgb_resampled_to_2k_with_regenerated_16px_single_donor_gutters",
        ),
        "sparse_layout": emit_png(
            page_dir / f"{page_id}_sparse_layout_2k.png",
            sparse,
            root=staging_root,
            semantic="current_final_v9_rgb_on_authoritative_uv_occupancy_black_elsewhere",
        ),
        "occupancy_mask": emit_png(
            page_dir / f"{page_id}_occupancy_mask_2k.png",
            occupancy,
            root=staging_root,
            semantic="authoritative_nearest_2k_binary_uv_occupancy_255_surface_0_background",
        ),
        "chart_owner": emit_png(
            page_dir / f"{page_id}_chart_owner_2k.png",
            chart_owner,
            root=staging_root,
            semantic="nearest_2k_occupied_only_existing_chart_id_zero_background_never_redefines_uvs",
        ),
        "object_owner": emit_png(
            page_dir / f"{page_id}_object_owner_2k.png",
            object_owner,
            root=staging_root,
            semantic="nearest_2k_occupied_only_source_object_id_plus_one_zero_background",
        ),
    }

    material_masks: dict[str, Any] = {}
    if page_id != "cockpit":
        full_mask = occupancy.copy()
        artifact = emit_png(
            page_dir / f"{page_id}_mask_full_occupied_2k.png",
            full_mask,
            root=staging_root,
            semantic="full_authoritative_occupied_material_candidate_mask",
        )
        material_masks["full_occupied"] = {
            "source_object_ids": [int(value - 1) for value in np.unique(object_owner[occupied])],
            "owner_values": [int(value) for value in np.unique(object_owner[occupied])],
            "eligible_for_generation": True,
            "artifact": artifact,
        }
    else:
        observed_source_ids = {int(value) - 1 for value in np.unique(object_owner[occupied])}
        configured_source_ids = {
            object_id
            for config in COCKPIT_PARTITION.values()
            for object_id in config["source_object_ids"]
        }
        require(observed_source_ids == configured_source_ids, f"Cockpit object partition IDs drift: observed={sorted(observed_source_ids)} configured={sorted(configured_source_ids)}")
        masks: dict[str, np.ndarray] = {}
        for group_name, config in COCKPIT_PARTITION.items():
            mask = make_binary_owner_mask(object_owner, config["source_object_ids"])
            require(bool(np.any(mask)), f"Empty cockpit material group: {group_name}")
            require(not bool(np.any((mask > 0) & ~occupied)), f"Cockpit group escapes occupancy: {group_name}")
            masks[group_name] = mask
            artifact = emit_png(
                page_dir / f"cockpit_mask_{group_name}_2k.png",
                mask,
                root=staging_root,
                semantic=str(config["semantic"]),
            )
            material_masks[group_name] = {
                "source_object_ids": list(config["source_object_ids"]),
                "owner_values": [object_id + 1 for object_id in config["source_object_ids"]],
                "eligible_for_generation": bool(config["eligible_for_generation"]),
                "artifact": artifact,
            }

        names = tuple(masks)
        overlap = np.zeros_like(occupancy, dtype=np.uint8)
        union = np.zeros_like(occupied, dtype=bool)
        for name in names:
            selected = masks[name] > 0
            require(not bool(np.any(union & selected)), f"Cockpit material masks overlap at {name}")
            union |= selected
            overlap += selected.astype(np.uint8)
        require(np.array_equal(union, occupied), "Cockpit candidate+exclusion masks do not exactly partition occupancy")
        require(int(overlap.max()) == 1, "Cockpit partition overlap count exceeds one")
        true_metal = masks["excluded"] > 0
        for name in ("leather", "plastics", "carbon"):
            require(not bool(np.any((masks[name] > 0) & true_metal)), f"Cockpit candidate includes excluded true metal: {name}")

    row = {
        "page": page_id,
        "native_resolution": [native_resolution, native_resolution],
        "resolution": [TARGET_SIZE, TARGET_SIZE],
        "scale_factor": native_resolution // TARGET_SIZE,
        "source": {
            "current_final_texture": texture_provenance,
            "uv_support": support_provenance,
        },
        "counts": {
            "occupied_texels": int(occupied.sum()),
            "padded_texels": int(padded.sum()),
            "gutter_texels": int(np.count_nonzero(padded & ~occupied)),
            "chart_ids_2k": int(len(np.unique(chart_owner[occupied]))),
            "object_ids_2k": int(len(np.unique(object_owner[occupied]))),
        },
        "gutter": gutter_report,
        "outputs": outputs,
        "material_masks": material_masks,
        "gates": {
            "current_final_texture_hash_verified": True,
            "authoritative_uv_support_hashes_verified": True,
            "nearest_2k_occupancy_chart_object_support_exact": True,
            "no_chart_or_uv_redefinition": True,
            "sparse_equals_current_final_resample_on_occupancy": True,
            "sparse_black_outside_occupancy": True,
            "working_occupied_pixels_unchanged_by_gutter": True,
            "working_black_outside_regenerated_16px_support": True,
            "outputs_reopened_losslessly": True,
        },
    }
    if page_id == "cockpit":
        row["gates"].update(
            {
                "cockpit_material_masks_pairwise_disjoint": True,
                "cockpit_candidates_plus_exclusion_union_exact_occupancy": True,
                "cockpit_true_metals_23_37_excluded": True,
            }
        )
    return row


def artifact_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("path", "semantic", "file_sha256", "decoded_pixel_sha256", "shape", "dtype", "bytes")
    }


def build_candidates(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page = {row["page"]: row for row in pages}
    candidates: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_ORDER:
        if candidate_id.startswith("cockpit_"):
            page_id = "cockpit"
            group = candidate_id.removeprefix("cockpit_")
        else:
            page_id = candidate_id
            group = "full_occupied"
        page = by_page[page_id]
        mask_row = page["material_masks"][group]
        require(mask_row["eligible_for_generation"] is True, f"Candidate mask is not eligible: {candidate_id}")
        outputs = page["outputs"]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "page": page_id,
                "material_group": group,
                "source_object_ids": list(mask_row["source_object_ids"]),
                "excluded_source_object_ids": [23, 37] if page_id == "cockpit" else [],
                "working_texture": artifact_reference(outputs["working_texture"]),
                "sparse_layout": artifact_reference(outputs["sparse_layout"]),
                "occupancy_mask": artifact_reference(outputs["occupancy_mask"]),
                "chart_owner": artifact_reference(outputs["chart_owner"]),
                "material_mask": artifact_reference(mask_row["artifact"]),
                "gates": {
                    "material_mask_subset_of_authoritative_occupancy": True,
                    "working_context_from_current_final_texture": True,
                    "chart_owner_is_context_only_never_generated": True,
                    "uv_support_never_defined_by_generated_output": True,
                },
            }
        )
    require(tuple(row["candidate_id"] for row in candidates) == CANDIDATE_ORDER, "Candidate order drift")

    cockpit = [row for row in candidates if row["page"] == "cockpit"]
    require(len(cockpit) == 3, "Expected exactly three cockpit candidates")
    for key in ("working_texture", "sparse_layout", "occupancy_mask", "chart_owner"):
        hashes = {row[key]["file_sha256"] for row in cockpit}
        paths = {row[key]["path"] for row in cockpit}
        require(len(hashes) == len(paths) == 1, f"Cockpit candidate context differs: {key}")
    return candidates


def write_hash_inventory(root: Path) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "hashes.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    inventory_path = root / "hashes.sha256"
    atomic_write_text(inventory_path, "\n".join(lines) + "\n")
    return {
        "path": inventory_path.relative_to(root).as_posix(),
        "sha256": sha256_file(inventory_path),
        "entries": len(lines),
    }


def iter_artifact_records(manifest: Mapping[str, Any]):
    seen: set[tuple[str, str]] = set()
    for page in manifest["pages"]:
        for record in page["outputs"].values():
            key = (str(record["path"]), str(record["file_sha256"]))
            if key not in seen:
                seen.add(key)
                yield record
        for group in page["material_masks"].values():
            record = group["artifact"]
            key = (str(record["path"]), str(record["file_sha256"]))
            if key not in seen:
                seen.add(key)
                yield record


def verify_generated_tree(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"Missing generated manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == OUTPUT_SCHEMA, "Generated manifest schema drift")
    require(manifest.get("status") == "passed", "Generated manifest did not pass")
    require(tuple(manifest.get("page_order", [])) == PAGE_ORDER, "Generated page order drift")
    require(tuple(row.get("candidate_id") for row in manifest.get("candidates", [])) == CANDIDATE_ORDER, "Generated candidate order drift")
    require(all(value is True for value in manifest.get("gates", {}).values()), "Generated global gate failed")

    for record in iter_artifact_records(manifest):
        path = root / str(record["path"])
        require(path.is_file(), f"Missing generated artifact: {path}")
        require(sha256_file(path) == record["file_sha256"], f"Generated artifact SHA drift: {path}")
        image = read_image(path)
        require(list(image.shape) == record["shape"], f"Generated artifact shape drift: {path}")
        require(str(image.dtype) == record["dtype"], f"Generated artifact dtype drift: {path}")
        require(sha256_array(image) == record["decoded_pixel_sha256"], f"Generated decoded pixel drift: {path}")

    for page in manifest["pages"]:
        outputs = page["outputs"]
        working = read_image(root / outputs["working_texture"]["path"])
        sparse = read_image(root / outputs["sparse_layout"]["path"])
        occupancy = read_image(root / outputs["occupancy_mask"]["path"]) > 0
        chart = read_image(root / outputs["chart_owner"]["path"])
        owner = read_image(root / outputs["object_owner"]["path"])
        require(np.array_equal(occupancy, chart > 0), f"Generated chart support drift: {page['page']}")
        require(np.array_equal(occupancy, owner > 0), f"Generated object support drift: {page['page']}")
        require(np.array_equal(sparse[occupancy], working[occupancy]), f"Generated occupied working/sparse drift: {page['page']}")
        require(not bool(np.any(sparse[~occupancy] != 0)), f"Generated sparse background drift: {page['page']}")
        for group_name, group in page["material_masks"].items():
            mask = read_image(root / group["artifact"]["path"]) > 0
            require(not bool(np.any(mask & ~occupancy)), f"Generated material mask escapes occupancy: {page['page']}/{group_name}")

    cockpit_page = next(page for page in manifest["pages"] if page["page"] == "cockpit")
    cockpit_occupancy = read_image(root / cockpit_page["outputs"]["occupancy_mask"]["path"]) > 0
    union = np.zeros_like(cockpit_occupancy)
    for group_name in ("leather", "plastics", "carbon", "excluded"):
        mask = read_image(root / cockpit_page["material_masks"][group_name]["artifact"]["path"]) > 0
        require(not bool(np.any(union & mask)), f"Generated cockpit partition overlap: {group_name}")
        union |= mask
    require(np.array_equal(union, cockpit_occupancy), "Generated cockpit partition union drift")

    inventory = verify_complete_hash_inventory(root)
    require("manifest.json" in parse_hash_inventory(root / "hashes.sha256"), "Generated manifest is not hash-covered")
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "hash_inventory": inventory,
        "pages": list(PAGE_ORDER),
        "candidates": list(CANDIDATE_ORDER),
    }


def self_test() -> dict[str, Any]:
    sparse = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    object_owner = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint16)
    chart_owner = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint16)
    object_owner[100:104, 100:104] = 20
    chart_owner[100:104, 100:104] = 1
    sparse[100:104, 100:104] = (11, 22, 33)
    object_owner[160:164, 160:164] = 24
    chart_owner[160:164, 160:164] = 2
    sparse[160:164, 160:164] = (44, 55, 66)
    working, padded, report = grow_owned_gutters(sparse, object_owner, chart_owner)
    require(report["padding_pixels_2k"] == PADDING_PIXELS, "Self-test gutter padding drift")
    require(np.array_equal(working[object_owner > 0], sparse[object_owner > 0]), "Self-test surface drift")
    require(not bool(np.any(working[padded == 0] != 0)), "Self-test working escaped padding")
    synthetic = np.zeros_like(object_owner)
    synthetic[10:20, 10:20] = 20
    synthetic[20:30, 10:20] = 37
    synthetic[30:40, 10:20] = 40
    leather = make_binary_owner_mask(synthetic, (19, 36, 39))
    require(int(np.count_nonzero(leather)) == 300, "Self-test owner mask mapping drift")
    return {
        "status": "passed",
        "gutter_single_donor": True,
        "owner_value_is_source_object_id_plus_one": True,
    }


def build_inputs(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    require(not output_dir.exists(), f"Refusing to replace existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=str(output_dir.parent)))
    try:
        final_manifest, final_manifest_record = load_final_manifest()
        pages = [
            prepare_page(page_id=page_id, final_manifest=final_manifest, staging_root=stage)
            for page_id in PAGE_ORDER
        ]
        candidates = build_candidates(pages)
        cockpit = next(page for page in pages if page["page"] == "cockpit")
        cockpit_candidate_context = {
            key: cockpit["outputs"][key]["file_sha256"]
            for key in ("working_texture", "sparse_layout", "occupancy_mask", "chart_owner")
        }
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "status": "passed",
            "resolution": [TARGET_SIZE, TARGET_SIZE],
            "page_order": list(PAGE_ORDER),
            "candidate_order": list(CANDIDATE_ORDER),
            "source_final_manifest": final_manifest_record,
            "policy": {
                "working_texture": "current final atlas RGB resampled to 2K on fixed authoritative UV support, then deterministic 16px single-donor gutters regenerated",
                "sparse_layout": "same working RGB only on authoritative occupied texels, black elsewhere",
                "occupancy_and_owners": "nearest-exact reduction from authoritative native support; never inferred from RGB or generated output",
                "material_masks": "binary subsets of fixed occupancy selected by nearest-exact source_object_id_plus_one owner",
                "cockpit_partition": "leather, plastics, carbon candidates plus excluded remainder are pairwise disjoint and union exactly to cockpit occupancy",
                "cockpit_true_metals_excluded_source_object_ids": [23, 37],
                "generated_output_never_defines_uv_or_chart_support": True,
                "no_network_fal_blender_or_github_performed": True,
            },
            "cockpit_shared_context_sha256": cockpit_candidate_context,
            "pages": pages,
            "candidates": candidates,
            "gates": {
                "exact_seven_page_set": True,
                "exact_nine_candidate_set": True,
                "all_current_final_texture_hashes_verified": True,
                "all_authoritative_support_hashes_verified": True,
                "all_outputs_reopened_losslessly": True,
                "all_candidate_masks_subset_fixed_occupancy": True,
                "cockpit_three_candidates_share_identical_context": True,
                "cockpit_material_masks_pairwise_disjoint": True,
                "cockpit_candidates_plus_exclusion_union_exact_occupancy": True,
                "cockpit_true_metals_23_37_excluded": True,
                "charts_and_uv_support_never_redefined": True,
                "no_network_fal_blender_or_github": True,
            },
        }
        atomic_write_text(stage / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        write_hash_inventory(stage)
        verify_generated_tree(stage)
        require(not output_dir.exists(), f"Output appeared while staging: {output_dir}")
        os.replace(stage, output_dir)
        return verify_generated_tree(output_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = self_test()
    if not args.self_test_only:
        result = {"self_test": result, "build": build_inputs(args.output_dir)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
