#!/usr/bin/env python3
"""Shared fail-closed helpers for the additive V10 texture finishing stages."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
INPUT_ROOT = SCRIPT_DIR / "inputs_2k"

INPUT_SCHEMA = "car-city.ferrari-targeted-seedream-inputs/v10"
SEEDREAM_SCHEMA = "car-city.ferrari-targeted-seedream-results/v10"
SEEDREAM_RETRY_SCHEMA = "car-city.ferrari-targeted-seedream-nomask-retry-results/v10"
PREPARED_TOPAZ_SCHEMA = "car-city.ferrari-targeted-topaz-inputs/v10"
TOPAZ_SCHEMA = "car-city.ferrari-targeted-topaz-results/v10"
FINAL_TEXTURE_SCHEMA = "car-city.ferrari-targeted-final-textures/v10"

TARGET_SIZE = 2048
PADDING_PIXELS = 16
PNG_COMPRESSION = 6
ROW_CHUNK = 128

PAGE_ORDER = (
    "body_hero_8k",
    "exterior_opaque",
    "glass",
    "lights_translucent",
    "running_metal",
    "tires",
    "cockpit",
)
NATIVE_SIZES = {
    "body_hero_8k": 8192,
    "exterior_opaque": 4096,
    "glass": 4096,
    "lights_translucent": 4096,
    "running_metal": 8192,
    "tires": 4096,
    "cockpit": 8192,
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
COCKPIT_CANDIDATES = (
    "cockpit_leather",
    "cockpit_plastics",
    "cockpit_carbon",
)
RETRY_CANDIDATE_ORDER = (
    "exterior_opaque",
    "running_metal",
    "tires",
)


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
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    require(path.is_file(), f"Missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard-link publishes the complete inode atomically
        # and, unlike rename/replace, fails if the destination already exists.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"Refusing to overwrite: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def read_image(path: Path, *, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
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


def artifact_record(
    path: Path,
    expected: np.ndarray,
    *,
    root: Path,
    semantic: str,
) -> dict[str, Any]:
    reopened = read_image(path)
    require(reopened.shape == expected.shape, f"PNG shape changed after reopen: {path}")
    require(reopened.dtype == expected.dtype, f"PNG dtype changed after reopen: {path}")
    expected_pixels = sha256_array(expected)
    decoded_pixels = sha256_array(reopened)
    require(decoded_pixels == expected_pixels, f"PNG pixels changed after reopen: {path}")
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "semantic": semantic,
        "file_sha256": sha256_file(path),
        "decoded_pixel_sha256": decoded_pixels,
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


def resolve_record_path(
    record: Mapping[str, Any],
    *,
    root: Path,
    field: str,
) -> Path:
    value = record.get("path")
    require(isinstance(value, str) and bool(value), f"Missing artifact path: {field}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    path = candidate.resolve()
    require(path.is_file(), f"Missing artifact {field}: {path}")
    expected = record.get("file_sha256", record.get("sha256"))
    require(isinstance(expected, str) and len(expected) == 64, f"Missing SHA-256: {field}")
    require(sha256_file(path) == expected, f"Artifact hash drift: {field}: {path}")
    return path


def load_verified_png(
    record: Mapping[str, Any],
    *,
    root: Path,
    field: str,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: np.dtype[Any] | type[np.generic] | None = None,
    flags: int = cv2.IMREAD_UNCHANGED,
) -> tuple[Path, np.ndarray]:
    path = resolve_record_path(record, root=root, field=field)
    image = read_image(path, flags=flags)
    declared_shape = record.get("shape")
    if declared_shape is not None:
        require(list(image.shape) == declared_shape, f"Declared shape drift: {field}")
    declared_dtype = record.get("dtype")
    if declared_dtype is not None:
        require(str(image.dtype) == declared_dtype, f"Declared dtype drift: {field}")
    decoded_sha = record.get("decoded_pixel_sha256")
    if decoded_sha is not None:
        require(sha256_array(image) == decoded_sha, f"Decoded pixel hash drift: {field}")
    if expected_shape is not None:
        require(image.shape == expected_shape, f"Shape drift for {field}: {image.shape}")
    if expected_dtype is not None:
        require(image.dtype == np.dtype(expected_dtype), f"Dtype drift for {field}: {image.dtype}")
    return path, image


def write_hash_inventory(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "hashes.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    inventory = root / "hashes.sha256"
    atomic_write_bytes(inventory, ("\n".join(lines) + "\n").encode("utf-8"))
    return {
        "path": inventory.relative_to(root).as_posix(),
        "sha256": sha256_file(inventory),
        "entries": len(lines),
    }


def parse_hash_inventory(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        parts = raw.split("  ", 1)
        require(len(parts) == 2, f"Malformed inventory line {line_number}: {path}")
        digest, relative = parts
        require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), f"Bad inventory digest: {path}:{line_number}")
        candidate = Path(relative)
        require(not candidate.is_absolute() and ".." not in candidate.parts, f"Unsafe inventory path: {relative}")
        require(relative not in rows, f"Duplicate inventory path: {relative}")
        rows[relative] = digest
    return rows


def verify_hash_inventory(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    inventory = root / "hashes.sha256"
    require(inventory.is_file(), f"Missing hash inventory: {inventory}")
    expected = parse_hash_inventory(inventory)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "hashes.sha256"
    }
    require(set(expected) == actual, f"Hash inventory file-set drift: {root}")
    for relative, digest in expected.items():
        require(sha256_file(root / relative) == digest, f"Hash inventory drift: {root / relative}")
    return {
        "path": inventory.relative_to(root).as_posix(),
        "sha256": sha256_file(inventory),
        "entries": len(expected),
    }


def load_input_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    manifest = load_json(path)
    require(manifest.get("schema") == INPUT_SCHEMA, "V10 input manifest schema drift")
    require(manifest.get("status") == "passed", "V10 input manifest did not pass")
    require(tuple(manifest.get("page_order", ())) == PAGE_ORDER, "V10 page order drift")
    require(tuple(manifest.get("candidate_order", ())) == CANDIDATE_ORDER, "V10 candidate order drift")
    gates = manifest.get("gates")
    require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()), "V10 input manifest gate failed")
    pages = manifest.get("pages")
    candidates = manifest.get("candidates")
    require(isinstance(pages, list) and tuple(row.get("page") for row in pages) == PAGE_ORDER, "V10 input page rows drift")
    require(isinstance(candidates, list) and tuple(row.get("candidate_id") for row in candidates) == CANDIDATE_ORDER, "V10 input candidate rows drift")
    by_page = {str(row["page"]): row for row in pages}
    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    require(len(by_page) == len(PAGE_ORDER), "Duplicate V10 input page")
    require(len(by_candidate) == len(CANDIDATE_ORDER), "Duplicate V10 input candidate")
    return manifest, by_page, by_candidate


def grow_owned_gutters_in_place(
    surface_bgr: np.ndarray,
    object_owner: np.ndarray,
    chart_owner: np.ndarray,
    *,
    padding: int = PADDING_PIXELS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow finite gutters by copying one exact occupied RGB/owner donor."""

    require(surface_bgr.dtype == np.uint8 and surface_bgr.ndim == 3 and surface_bgr.shape[2] == 3, "Gutter RGB must be uint8 HxWx3")
    require(object_owner.dtype == chart_owner.dtype == np.uint16, "Gutter owners must be uint16")
    require(object_owner.shape == chart_owner.shape == surface_bgr.shape[:2], "Gutter shapes differ")
    occupied = object_owner > 0
    require(np.array_equal(occupied, chart_owner > 0), "Object/chart occupied support differs")
    occupied_count = int(occupied.sum())
    require(occupied_count > 0, "Cannot grow gutters from empty support")
    require(not bool(np.any(surface_bgr[~occupied] != 0)), "Surface must be black outside occupancy before gutter growth")

    source = np.where(occupied, 0, 255).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_C,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    del source
    seed_labels = labels[occupied]
    require(bool(np.all(seed_labels > 0)), "Occupied donors have missing labels")
    slots = int(labels.max()) + 1
    donor_bgr = np.zeros((slots, 3), dtype=np.uint8)
    donor_object = np.zeros(slots, dtype=np.uint16)
    donor_chart = np.zeros(slots, dtype=np.uint16)
    donor_bgr[seed_labels] = surface_bgr[occupied]
    donor_object[seed_labels] = object_owner[occupied]
    donor_chart[seed_labels] = chart_owner[occupied]
    del seed_labels

    padded_mask = np.zeros(object_owner.shape, dtype=np.uint8)
    gutter_count = 0
    for y0 in range(0, object_owner.shape[0], ROW_CHUNK):
        y1 = min(object_owner.shape[0], y0 + ROW_CHUNK)
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
        gutter_count += int(gutter.sum())

    del distance, labels, donor_bgr, donor_object, donor_chart
    gc.collect()
    padded = padded_mask > 0
    require(np.array_equal(object_owner[padded] > 0, chart_owner[padded] > 0), "Padded owner supports differ")
    require(not bool(np.any((object_owner > 0) & ~padded)), "Owner escaped finite gutter support")
    return padded_mask, {
        "algorithm": "opencv_DIST_C_labelled_nearest_pixel_donor",
        "padding_pixels": int(padding),
        "occupied_texels": occupied_count,
        "gutter_texels": int(gutter_count),
        "padded_texels": int(np.count_nonzero(padded_mask)),
        "cross_domain_averaging": False,
        "rgb_object_chart_copied_from_same_single_donor": True,
        "tie_resolution": "deterministic_opencv_pixel_label_order",
    }


def inject_candidates(
    baseline: np.ndarray,
    candidate_images: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
    occupied: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Insert candidates only under disjoint masks, falling back on new black."""

    require(baseline.dtype == np.uint8 and baseline.ndim == 3 and baseline.shape[2] == 3, "Baseline must be uint8 HxWx3")
    require(occupied.shape == baseline.shape[:2] and occupied.dtype == np.bool_, "Bad occupancy")
    require(set(candidate_images) == set(candidate_masks), "Candidate image/mask keys differ")
    surface = np.zeros_like(baseline)
    surface[occupied] = baseline[occupied]
    union = np.zeros_like(occupied)
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_images:
        donor = candidate_images[candidate_id]
        mask = candidate_masks[candidate_id]
        require(donor.shape == baseline.shape and donor.dtype == np.uint8, f"Bad donor: {candidate_id}")
        require(mask.shape == occupied.shape and mask.dtype == np.bool_, f"Bad mask: {candidate_id}")
        require(not bool(np.any(mask & ~occupied)), f"Candidate mask escapes occupancy: {candidate_id}")
        require(not bool(np.any(union & mask)), f"Candidate masks overlap: {candidate_id}")
        union |= mask
        donor_black = np.all(donor == 0, axis=2)
        baseline_nonblack = np.any(baseline != 0, axis=2)
        new_black = mask & donor_black & baseline_nonblack
        accepted = mask & ~new_black
        surface[accepted] = donor[accepted]
        rows.append(
            {
                "candidate_id": candidate_id,
                "mask_texels": int(mask.sum()),
                "introduced_exact_black_detected_and_replaced": int(new_black.sum()),
                "changed_masked_texels": int((accepted & np.any(donor != baseline, axis=2)).sum()),
            }
        )
        del donor_black, baseline_nonblack, new_black, accepted
    surface[~occupied] = 0
    remaining = occupied & union & np.all(surface == 0, axis=2) & np.any(baseline != 0, axis=2)
    require(not bool(np.any(remaining)), "New exact-black texels remain after fallback")
    return surface, {
        "candidates": rows,
        "candidate_union_texels": int(union.sum()),
        "baseline_preserved_occupied_texels": int((occupied & ~union).sum()),
        "remaining_new_exact_black_texels": int(remaining.sum()),
        "candidate_masks_pairwise_disjoint": True,
        "injection_inside_fixed_masks_only": True,
    }


def self_test_common() -> dict[str, Any]:
    baseline = np.full((32, 32, 3), 40, dtype=np.uint8)
    occupied = np.zeros((32, 32), dtype=bool)
    occupied[8:24, 8:24] = True
    mask_a = np.zeros_like(occupied)
    mask_b = np.zeros_like(occupied)
    mask_a[8:16, 8:24] = True
    mask_b[16:24, 8:16] = True
    donor_a = np.full_like(baseline, 90)
    donor_b = np.full_like(baseline, 120)
    donor_a[10, 10] = 0
    surface, metrics = inject_candidates(
        baseline,
        {"a": donor_a, "b": donor_b},
        {"a": mask_a, "b": mask_b},
        occupied,
    )
    require(np.array_equal(surface[10, 10], baseline[10, 10]), "Synthetic new-black fallback failed")
    require(bool(np.all(surface[20:24, 16:24] == baseline[20:24, 16:24])), "Synthetic exclusion preservation failed")
    owner = occupied.astype(np.uint16)
    chart = owner.copy()
    padded, gutter = grow_owned_gutters_in_place(surface, owner, chart, padding=3)
    require(gutter["cross_domain_averaging"] is False, "Synthetic gutter mixed domains")
    require(gutter["rgb_object_chart_copied_from_same_single_donor"] is True, "Synthetic gutter donor proof failed")
    require(int(np.count_nonzero(padded)) > int(occupied.sum()), "Synthetic gutter did not grow")
    return {
        "status": "passed",
        "new_black_fallback": metrics["candidates"][0]["introduced_exact_black_detected_and_replaced"],
        "excluded_baseline_preserved": True,
        "single_donor_gutter": True,
    }
