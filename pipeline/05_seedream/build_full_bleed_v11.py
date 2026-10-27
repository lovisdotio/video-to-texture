#!/usr/bin/env python3
"""Build seven deterministic full-bleed V11 atlases from final V10 textures.

The authoritative occupied support is taken from the hash-pinned V10 final
report (V9 native reprojection support for six pages and the preserved V8 tire
support).  Occupied RGB is never changed.  Every unoccupied texel is copied
from one nearest occupied, non-exact-black donor, so the atlas has no black
void while legitimate exact-black texels inside UV islands remain byte-exact.

This is a local post-process only: no network, Seedream, Topaz, Blender, or Git
operation is performed.  Outputs are staged beside the destination and
published by one no-overwrite directory rename.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from v10_postprocess_common import (
    PAGE_ORDER,
    atomic_write_json,
    emit_png,
    load_json,
    load_verified_png,
    read_image,
    require,
    resolve_record_path,
    sha256_array,
    sha256_file,
    verify_hash_inventory,
    write_hash_inventory,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_REPORT = SCRIPT_DIR / "textures_final_v10" / "report.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "textures_final_v11_full_bleed"

SOURCE_SCHEMA = "car-city.ferrari-targeted-final-textures/v10"
OUTPUT_SCHEMA = "car-city.ferrari-full-bleed-textures/v11"
PNG_COMPRESSION = 6
ROW_CHUNK = 128
PREVIEW_MAX_SIDE = 2048
CONTACT_COLUMNS = 4
CONTACT_CELL = 1024
CONTACT_LABEL_HEIGHT = 48


def _masked_pixel_sha256(image: np.ndarray, mask: np.ndarray) -> str:
    """Hash selected RGB in row-major order without one large gather."""

    import hashlib

    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, "Bad RGB for masked hash")
    require(mask.dtype == np.bool_ and mask.shape == image.shape[:2], "Bad mask for masked hash")
    digest = hashlib.sha256()
    digest.update(b"uint8\0masked-row-major-bgr\0")
    digest.update(np.asarray([int(mask.sum()), 3], dtype=np.int64).tobytes())
    for y0 in range(0, image.shape[0], ROW_CHUNK):
        y1 = min(image.shape[0], y0 + ROW_CHUNK)
        selected = np.ascontiguousarray(image[y0:y1][mask[y0:y1]])
        digest.update(selected.tobytes())
    return digest.hexdigest()


def _count_exact_black(image: np.ndarray, mask: np.ndarray) -> int:
    total = 0
    for y0 in range(0, image.shape[0], ROW_CHUNK):
        y1 = min(image.shape[0], y0 + ROW_CHUNK)
        local_mask = mask[y0:y1]
        local_black = np.all(image[y0:y1] == 0, axis=2)
        total += int(np.count_nonzero(local_mask & local_black))
    return total


def _record_native_artifact(record: Mapping[str, Any], *, field: str) -> tuple[Path, np.ndarray, dict[str, Any]]:
    path = resolve_record_path(record, root=SCRIPT_DIR.parent, field=field)
    image = read_image(path, flags=cv2.IMREAD_UNCHANGED)
    return path, image, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "decoded_pixel_sha256": sha256_array(image),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }


def _validate_support(
    page: str,
    row: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs = row.get("inputs")
    require(isinstance(inputs, dict), f"Missing V10 inputs: {page}")
    support = inputs.get("native_uv_support")
    require(isinstance(support, dict), f"Missing native support: {page}")
    support_hashes = inputs.get("native_support_sha256")
    require(isinstance(support_hashes, dict), f"Missing native decoded hashes: {page}")

    expected_kind = (
        "v8_production_support_preserved_by_v9_tire_passthrough"
        if page == "tires"
        else "validated_v9_native_reprojection_support"
    )
    require(support.get("kind") == expected_kind, f"Native support kind drift: {page}")

    occupancy_path, occupancy, occupancy_record = _record_native_artifact(
        support["occupancy_mask"], field=f"{page}.occupancy"
    )
    require(occupancy.shape == expected_shape and occupancy.dtype == np.uint8, f"Bad occupancy raster: {page}")
    require(set(np.unique(occupancy).tolist()).issubset({0, 255}), f"Non-binary occupancy: {page}")
    require(
        occupancy_record["decoded_pixel_sha256"] == support_hashes.get("occupancy_decoded"),
        f"Occupancy decoded hash drift: {page}",
    )
    occupied = occupancy > 0
    del occupancy

    owner_key = "piece_owner" if page == "tires" else "object_owner"
    owner_path, raw_owner, owner_record = _record_native_artifact(
        support[owner_key], field=f"{page}.{owner_key}"
    )
    require(raw_owner.shape == expected_shape and raw_owner.dtype == np.uint16, f"Bad owner raster: {page}")
    if page == "tires":
        mapping_raw = support.get("piece_owner_normalization")
        require(isinstance(mapping_raw, dict) and mapping_raw, "Missing tire owner normalization")
        mapping = {int(piece): int(owner) for piece, owner in mapping_raw.items()}
        observed = set(int(value) for value in np.unique(raw_owner[raw_owner > 0]))
        require(observed == set(mapping), f"Tire piece-owner set drift: {sorted(observed)}")
        normalized_owner = np.zeros_like(raw_owner, dtype=np.uint16)
        for piece, owner in sorted(mapping.items()):
            normalized_owner[raw_owner == piece] = owner
        owner_decoded_sha = sha256_array(normalized_owner)
        owner_support = normalized_owner > 0
        owner_record["normalization"] = {str(piece): owner for piece, owner in sorted(mapping.items())}
        owner_record["normalized_decoded_pixel_sha256"] = owner_decoded_sha
        del normalized_owner
    else:
        owner_decoded_sha = owner_record["decoded_pixel_sha256"]
        owner_support = raw_owner > 0
    require(np.array_equal(owner_support, occupied), f"Owner/occupancy support drift: {page}")
    require(owner_decoded_sha == support_hashes.get("object_owner_decoded"), f"Owner decoded hash drift: {page}")
    del raw_owner, owner_support
    gc.collect()

    chart_path, chart_owner, chart_record = _record_native_artifact(
        support["chart_owner"], field=f"{page}.chart_owner"
    )
    require(chart_owner.shape == expected_shape and chart_owner.dtype == np.uint16, f"Bad chart owner: {page}")
    require(np.array_equal(chart_owner > 0, occupied), f"Chart/occupancy support drift: {page}")
    require(
        chart_record["decoded_pixel_sha256"] == support_hashes.get("chart_owner_decoded"),
        f"Chart owner decoded hash drift: {page}",
    )
    del chart_owner
    gc.collect()

    declared_occupied = row.get("metrics", {}).get("occupied_texels")
    occupied_count = int(occupied.sum())
    require(declared_occupied == occupied_count, f"V10 occupied count drift: {page}")

    inventory_record = support.get("hash_inventory")
    require(isinstance(inventory_record, dict), f"Missing support hash inventory: {page}")
    inventory_path = resolve_record_path(
        inventory_record, root=SCRIPT_DIR.parent, field=f"{page}.support_hash_inventory"
    )

    if page != "tires":
        require("targeted_rebake_v9" in occupancy_path.parts, f"Expected V9 native support path: {page}")
    provenance = {
        "support_kind": expected_kind,
        "support_generation": "V8 preserved tire support" if page == "tires" else "V9 native reprojection support",
        "occupancy_mask": occupancy_record,
        "owner": {"kind": owner_key, **owner_record},
        "chart_owner": chart_record,
        "hash_inventory": {"path": str(inventory_path), "file_sha256": sha256_file(inventory_path)},
        "audit": {
            "binary_occupancy": True,
            "occupancy_equals_owner_support": True,
            "occupancy_equals_chart_support": True,
            "decoded_hashes_match_v10_report": True,
            "occupied_texels": occupied_count,
        },
    }
    return occupied, provenance


def _validate_source_report(report_path: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    report_path = report_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    require(not output_root.exists(), f"Refusing to overwrite V11 output: {output_root}")
    report = load_json(report_path)
    require(report.get("schema") == SOURCE_SCHEMA, "V10 final report schema drift")
    require(report.get("status") == "passed", "V10 final report did not pass")
    top_gates = report.get("hard_gates")
    require(isinstance(top_gates, dict) and top_gates and all(value is True for value in top_gates.values()), "V10 final hard gate failed")
    rows = report.get("pages")
    require(isinstance(rows, list), "V10 final pages missing")
    require(tuple(row.get("page") for row in rows) == PAGE_ORDER, "V10 final page order/set drift")
    by_page = {str(row["page"]): row for row in rows}
    require(len(by_page) == len(PAGE_ORDER), "Duplicate V10 final page")
    source_inventory = verify_hash_inventory(report_path.parent)
    source_record = {
        "path": str(report_path),
        "file_sha256": sha256_file(report_path),
        "schema": report["schema"],
        "status": report["status"],
        "hash_inventory": {
            "path": str((report_path.parent / source_inventory["path"]).resolve()),
            "file_sha256": source_inventory["sha256"],
            "entries": source_inventory["entries"],
        },
    }
    return report, by_page, source_record


def _load_page(
    page: str,
    row: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    resolution = row.get("native_resolution")
    require(
        isinstance(resolution, list)
        and len(resolution) == 2
        and resolution[0] == resolution[1]
        and int(resolution[0]) > 0,
        f"Bad V10 native resolution: {page}",
    )
    size = int(resolution[0])
    occupied, support_record = _validate_support(page, row, expected_shape=(size, size))
    texture_path, image = load_verified_png(
        row["output"],
        root=source_root,
        field=f"{page}.v10_final_texture",
        expected_shape=(size, size, 3),
        expected_dtype=np.uint8,
        flags=cv2.IMREAD_COLOR,
    )
    page_gates = row.get("hard_gates")
    require(isinstance(page_gates, dict) and page_gates and all(value is True for value in page_gates.values()), f"V10 page hard gate failed: {page}")
    input_record = {
        "v10_final_texture": {
            "path": str(texture_path),
            "file_sha256": sha256_file(texture_path),
            "decoded_pixel_sha256": sha256_array(image),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        },
        "native_support": support_record,
    }
    return image, occupied, input_record


def dry_run(*, source_report_path: Path, output_root: Path) -> dict[str, Any]:
    report, pages, source_record = _validate_source_report(source_report_path, output_root)
    rows: list[dict[str, Any]] = []
    for page in PAGE_ORDER:
        image, occupied, input_record = _load_page(page, pages[page], source_root=source_report_path.resolve().parent)
        occupied_black = _count_exact_black(image, occupied)
        nonblack_donors = int(occupied.sum()) - occupied_black
        require(nonblack_donors > 0, f"No non-black occupied donor: {page}")
        rows.append(
            {
                "page": page,
                "resolution": list(image.shape[:2]),
                "occupied_texels": int(occupied.sum()),
                "occupied_exact_black_texels_to_preserve": occupied_black,
                "eligible_occupied_nonblack_donors": nonblack_donors,
                "unoccupied_texels_to_fill": int(occupied.size - occupied.sum()),
                "source_texture": input_record["v10_final_texture"],
                "support_kind": input_record["native_support"]["support_kind"],
                "support_audit_passed": all(input_record["native_support"]["audit"].values()),
            }
        )
        del image, occupied
        gc.collect()
    return {
        "schema": "car-city.ferrari-full-bleed-plan/v11",
        "status": "passed",
        "source_final_v10": source_record,
        "output_root": str(output_root.expanduser().resolve()),
        "would_overwrite": False,
        "page_order": list(PAGE_ORDER),
        "algorithm": {
            "occupied_rgb": "byte_exact_unchanged",
            "unoccupied_rgb": "single nearest occupied non-exact-black donor",
            "distance_metric": "exact Chebyshev distance via OpenCV DIST_C labelled transform",
            "tie_resolution": "deterministic OpenCV pixel-label order",
            "interpolation_or_averaging": False,
            "publication": "fresh sibling staging directory then no-overwrite rename",
        },
        "pages": rows,
        "hard_gates": {
            "source_v10_hash_inventory_verified": True,
            "exact_seven_pages": len(rows) == 7,
            "all_v10_v9_masks_and_owners_verified": all(row["support_audit_passed"] for row in rows),
            "all_pages_have_nonblack_occupied_donors": all(row["eligible_occupied_nonblack_donors"] > 0 for row in rows),
            "output_root_absent": True,
            "network_required": False,
            "blender_required": False,
        },
    }


def _full_bleed_in_place(image: np.ndarray, occupied: np.ndarray) -> dict[str, Any]:
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, "Full-bleed RGB must be uint8 HxWx3")
    require(occupied.dtype == np.bool_ and occupied.shape == image.shape[:2], "Full-bleed occupancy mismatch")
    occupied_count = int(occupied.sum())
    require(occupied_count > 0, "Cannot full-bleed empty occupancy")
    occupied_hash_before = _masked_pixel_sha256(image, occupied)
    occupied_black_before = _count_exact_black(image, occupied)

    valid = occupied.copy()
    for y0 in range(0, image.shape[0], ROW_CHUNK):
        y1 = min(image.shape[0], y0 + ROW_CHUNK)
        valid[y0:y1] &= np.any(image[y0:y1] != 0, axis=2)
    valid_count = int(valid.sum())
    require(valid_count == occupied_count - occupied_black_before, "Valid donor count drift")
    require(valid_count > 0, "No occupied non-exact-black donor")

    distance_input = np.full(occupied.shape, 255, dtype=np.uint8)
    distance_input[valid] = 0
    distance, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_C,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    del distance_input, distance
    gc.collect()
    require(labels.dtype == np.int32 and labels.shape == occupied.shape, "Bad nearest-donor label raster")
    require(int(labels.min()) > 0, "Some atlas texels have no valid donor label")
    label_count = int(labels.max())
    require(label_count == valid_count, "OpenCV donor label count drift")

    donor_lut = np.zeros((label_count + 1, 3), dtype=np.uint8)
    for y0 in range(0, image.shape[0], ROW_CHUNK):
        y1 = min(image.shape[0], y0 + ROW_CHUNK)
        local_valid = valid[y0:y1]
        donor_lut[labels[y0:y1][local_valid]] = image[y0:y1][local_valid]
    require(bool(np.all(np.any(donor_lut[1:] != 0, axis=1))), "Missing or exact-black donor LUT entry")
    del valid
    gc.collect()

    unoccupied_count = int(occupied.size - occupied_count)
    unoccupied_black_before = _count_exact_black(image, ~occupied)
    changed_unoccupied = 0
    for y0 in range(0, image.shape[0], ROW_CHUNK):
        y1 = min(image.shape[0], y0 + ROW_CHUNK)
        local_background = ~occupied[y0:y1]
        if not bool(np.any(local_background)):
            continue
        replacement = donor_lut[labels[y0:y1][local_background]]
        previous = image[y0:y1][local_background]
        changed_unoccupied += int(np.count_nonzero(np.any(previous != replacement, axis=1)))
        image[y0:y1][local_background] = replacement

    occupied_hash_after = _masked_pixel_sha256(image, occupied)
    occupied_black_after = _count_exact_black(image, occupied)
    unoccupied_black_after = _count_exact_black(image, ~occupied)
    require(occupied_hash_after == occupied_hash_before, "Occupied RGB changed during full bleed")
    require(occupied_black_after == occupied_black_before, "Occupied exact-black texels changed")
    require(unoccupied_black_after == 0, "Exact-black atlas background remains")
    require(int(labels.min()) > 0, "Unlabelled atlas background remains")
    del donor_lut, labels
    gc.collect()
    return {
        "occupied_texels": occupied_count,
        "unoccupied_texels_filled": unoccupied_count,
        "eligible_occupied_nonblack_donor_texels": valid_count,
        "occupied_exact_black_texels_before": occupied_black_before,
        "occupied_exact_black_texels_after": occupied_black_after,
        "unoccupied_exact_black_texels_before": unoccupied_black_before,
        "unoccupied_exact_black_texels_after": unoccupied_black_after,
        "unoccupied_texels_value_changed": changed_unoccupied,
        "occupied_rgb_sha256_before": occupied_hash_before,
        "occupied_rgb_sha256_after": occupied_hash_after,
        "nearest_donor_labels": valid_count,
    }


def _preview(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, PREVIEW_MAX_SIDE / float(max(height, width)))
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if target == (width, height):
        return image.copy()
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def _contact_sheet(staging: Path, rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    contact_rows = (len(rows) + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS
    cell_height = CONTACT_LABEL_HEIGHT + CONTACT_CELL
    sheet = np.full((contact_rows * cell_height, CONTACT_COLUMNS * CONTACT_CELL, 3), 42, dtype=np.uint8)
    for index, row in enumerate(rows):
        preview_path = staging / row["preview"]["path"]
        preview = read_image(preview_path, flags=cv2.IMREAD_COLOR)
        thumb = cv2.resize(preview, (CONTACT_CELL, CONTACT_CELL), interpolation=cv2.INTER_AREA)
        grid_y, grid_x = divmod(index, CONTACT_COLUMNS)
        x0 = grid_x * CONTACT_CELL
        y0 = grid_y * cell_height
        sheet[y0 + CONTACT_LABEL_HEIGHT : y0 + cell_height, x0 : x0 + CONTACT_CELL] = thumb
        cv2.putText(
            sheet,
            str(row["page"]),
            (x0 + 16, y0 + 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    record = emit_png(
        staging / "V11_FULL_BLEED_SEVEN_PAGES_CONTACT.png",
        sheet,
        root=staging,
        semantic="Seven-page V11 full-bleed contact sheet; each source preview is at most 2048px per side",
    )
    return sheet, record


def _process_page(
    page: str,
    row: Mapping[str, Any],
    *,
    source_root: Path,
    staging: Path,
) -> dict[str, Any]:
    image, occupied, input_record = _load_page(page, row, source_root=source_root)
    metrics = _full_bleed_in_place(image, occupied)
    size = image.shape[0]
    output = emit_png(
        staging / f"{page}_seedream_v11_full_bleed.png",
        image,
        root=staging,
        semantic="V10 occupied RGB byte-exact; all unoccupied RGB copied from one nearest occupied non-exact-black donor",
    )
    preview_image = _preview(image)
    preview = emit_png(
        staging / "previews" / f"{page}_seedream_v11_full_bleed_preview_2k.png",
        preview_image,
        root=staging,
        semantic="Downsampled inspection preview of V11 full-bleed atlas; maximum side 2048 pixels",
    )
    gates = {
        "v10_v9_mask_owner_audit_passed": all(input_record["native_support"]["audit"].values()),
        "occupied_rgb_byte_exact": metrics["occupied_rgb_sha256_before"] == metrics["occupied_rgb_sha256_after"],
        "occupied_exact_black_preserved": metrics["occupied_exact_black_texels_before"] == metrics["occupied_exact_black_texels_after"],
        "all_unoccupied_texels_assigned_valid_donor": metrics["unoccupied_texels_filled"] == int(occupied.size - occupied.sum()),
        "zero_exact_black_unoccupied_texels": metrics["unoccupied_exact_black_texels_after"] == 0,
        "single_donor_no_interpolation": True,
        "native_dimensions_preserved": output["shape"] == [size, size, 3],
    }
    require(all(gates.values()), f"V11 full-bleed page gate failed: {page}")
    result = {
        "page": page,
        "native_resolution": [size, size],
        "input": input_record,
        "algorithm": {
            "name": "opencv_DIST_C_labelled_nearest_nonblack_occupied_donor",
            "distance_metric": "exact Chebyshev",
            "label_type": "DIST_LABEL_PIXEL",
            "tie_resolution": "deterministic OpenCV pixel-label order",
            "donor_eligibility": "authoritative occupancy AND RGB != [0,0,0]",
            "write_scope": "unoccupied texels only",
            "interpolation_or_averaging": False,
        },
        "metrics": metrics,
        "output": output,
        "preview": preview,
        "hard_gates": gates,
    }
    del image, occupied, preview_image
    gc.collect()
    return result


def build(*, source_report_path: Path, output_root: Path) -> dict[str, Any]:
    source_report_path = source_report_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    source_report, pages, source_record = _validate_source_report(source_report_path, output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=str(output_root.parent)))
    try:
        rows: list[dict[str, Any]] = []
        for page in PAGE_ORDER:
            print(f"V11_FULL_BLEED_START {page}", flush=True)
            result = _process_page(
                page,
                pages[page],
                source_root=source_report_path.parent,
                staging=staging,
            )
            rows.append(result)
            print(f"V11_FULL_BLEED_DONE {page} {result['output']['file_sha256']}", flush=True)
        _, contact = _contact_sheet(staging, rows)
        report = {
            "schema": OUTPUT_SCHEMA,
            "status": "passed",
            "page_order": list(PAGE_ORDER),
            "source_final_v10": source_record,
            "policy": {
                "uv_layout_changed": False,
                "occupied_support_changed": False,
                "occupied_rgb": "byte_exact_from_textures_final_v10_including_legitimate_exact_black",
                "unoccupied_rgb": "single_nearest_authoritative_occupied_non_exact_black_donor_to_every_atlas_edge",
                "background_exact_black": "forbidden; occupied exact black remains allowed and preserved",
                "interpolation_or_cross_domain_averaging": False,
                "output_publication": "fresh_sibling_staging_directory_then_atomic_rename_no_overwrite",
                "network_seedream_topaz_blender_git": "not_called",
            },
            "source_to_output_mapping": {
                row["page"]: {
                    "source": row["input"]["v10_final_texture"]["path"],
                    "output": row["output"]["path"],
                }
                for row in rows
            },
            "pages": rows,
            "preview_contact_sheet": contact,
            "hard_gates": {
                "exact_seven_pages": len(rows) == 7,
                "page_order_exact": tuple(row["page"] for row in rows) == PAGE_ORDER,
                "source_v10_hash_inventory_verified": True,
                "all_v10_v9_masks_and_owners_verified": all(row["hard_gates"]["v10_v9_mask_owner_audit_passed"] for row in rows),
                "all_occupied_rgb_byte_exact": all(row["hard_gates"]["occupied_rgb_byte_exact"] for row in rows),
                "all_occupied_exact_black_preserved": all(row["hard_gates"]["occupied_exact_black_preserved"] for row in rows),
                "all_unoccupied_texels_filled": all(row["hard_gates"]["all_unoccupied_texels_assigned_valid_donor"] for row in rows),
                "zero_exact_black_unoccupied_texels_all_pages": all(row["metrics"]["unoccupied_exact_black_texels_after"] == 0 for row in rows),
                "all_native_dimensions_preserved": all(row["hard_gates"]["native_dimensions_preserved"] for row in rows),
                "seven_previews_emitted": len([row["preview"] for row in rows]) == 7,
                "atomic_output_directory": True,
                "no_overwrite": True,
            },
        }
        require(all(report["hard_gates"].values()), "V11 full-bleed report hard gate failed")
        atomic_write_json(staging / "report.json", report)
        write_hash_inventory(staging)
        require(not output_root.exists(), f"Refusing to overwrite V11 output: {output_root}")
        os.rename(staging, output_root)
        verify_hash_inventory(output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> dict[str, Any]:
    image = np.zeros((13, 17, 3), dtype=np.uint8)
    occupied = np.zeros((13, 17), dtype=bool)
    occupied[3:10, 4:13] = True
    image[occupied] = (30, 80, 120)
    image[5:8, 7:10] = 0
    original = image.copy()
    metrics = _full_bleed_in_place(image, occupied)
    require(np.array_equal(image[occupied], original[occupied]), "Self-test occupied RGB changed")
    require(not bool(np.any(np.all(image[~occupied] == 0, axis=1))), "Self-test background black remains")
    require(metrics["occupied_exact_black_texels_before"] == 9, "Self-test occupied black count drift")
    return {
        "status": "passed",
        "occupied_rgb_byte_exact": True,
        "occupied_exact_black_preserved": metrics["occupied_exact_black_texels_after"],
        "unoccupied_exact_black_after": metrics["unoccupied_exact_black_texels_after"],
        "all_unoccupied_filled": metrics["unoccupied_texels_filled"],
        "network_or_blender_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print("V11_FULL_BLEED_SELF_TEST=" + json.dumps(self_test(), sort_keys=True), flush=True)
        return 0
    if args.dry_run:
        plan = dry_run(source_report_path=args.source_report, output_root=args.output_root)
        print("V11_FULL_BLEED_DRY_RUN=" + json.dumps(plan, sort_keys=True), flush=True)
        return 0
    report = build(source_report_path=args.source_report, output_root=args.output_root)
    print(
        "V11_FULL_BLEED="
        + json.dumps(
            {
                "status": report["status"],
                "output_root": str(args.output_root.expanduser().resolve()),
                "pages": [row["page"] for row in report["pages"]],
                "contact_sheet": report["preview_contact_sheet"]["path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
