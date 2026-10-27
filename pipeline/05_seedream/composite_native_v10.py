#!/usr/bin/env python3
"""Insert seven Topaz V10 donors into authoritative native UV support.

Generated RGB is accepted only under the fixed native material mask.  Cockpit
exclusions remain byte-exact from the current V9 final texture.  Newly
introduced exact black falls back to that baseline, then finite 16-pixel
single-donor gutters are rebuilt.  The output directory is additive and
published atomically; this script never launches Fal, Blender, or GitHub.
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
    FINAL_TEXTURE_SCHEMA,
    INPUT_ROOT,
    NATIVE_SIZES,
    PAGE_ORDER,
    RETRY_CANDIDATE_ORDER,
    SCRIPT_DIR,
    TOPAZ_SCHEMA,
    V8_ROOT,
    atomic_write_json,
    emit_png,
    grow_owned_gutters_in_place,
    inject_candidates,
    load_input_manifest,
    load_json,
    load_verified_png,
    require,
    resolve_record_path,
    self_test_common,
    sha256_array,
    sha256_file,
    verify_hash_inventory,
    write_hash_inventory,
)


DEFAULT_TOPAZ_REPORT = SCRIPT_DIR / "fal" / "topaz_v10" / "report.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "textures_final_v10"


def validate_topaz_report(
    report_path: Path,
    *,
    input_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report_path = report_path.expanduser().resolve()
    report = load_json(report_path)
    require(report.get("schema") == TOPAZ_SCHEMA, "Topaz V10 report schema drift")
    require(report.get("status") == "passed", "Topaz V10 report did not pass")
    require(report.get("input_manifest_sha256") == input_manifest_sha256, "Topaz V10 input-manifest binding drift")
    retry_sha = report.get("retry_report_sha256")
    require(isinstance(retry_sha, str) and len(retry_sha) == 64, "Topaz V10 report lacks the required retry-report binding")
    selection = report.get("seedream_selection")
    require(isinstance(selection, dict), "Topaz V10 report lacks Seedream selection provenance")
    require(tuple(selection.get("overridden_candidate_ids", ())) == RETRY_CANDIDATE_ORDER, "Topaz V10 retry override set drift")
    require(len(selection.get("original_candidate_ids", ())) == 6, "Topaz V10 must retain exactly six original candidates")
    source_map = selection.get("candidate_source")
    require(isinstance(source_map, dict), "Topaz V10 candidate source map missing")
    require(all(source_map.get(candidate_id) == "nomask_retry_override" for candidate_id in RETRY_CANDIDATE_ORDER), "Topaz V10 did not select all three retry outputs")
    gates = report.get("hard_gates")
    require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()), "Topaz V10 report hard gate failed")
    rows = report.get("pages")
    require(isinstance(rows, list) and tuple(row.get("page") for row in rows) == PAGE_ORDER, "Topaz V10 page order/set drift")
    by_page: dict[str, dict[str, Any]] = {}
    for row in rows:
        page = str(row["page"])
        size = NATIVE_SIZES[page]
        require(row.get("native_resolution") == [size, size], f"Topaz native resolution record drift: {page}")
        _, donor = load_verified_png(
            row["output"],
            root=report_path.parent,
            field=f"topaz.{page}.output",
            expected_shape=(size, size, 3),
            expected_dtype=np.uint8,
            flags=cv2.IMREAD_COLOR,
        )
        del donor
        by_page[page] = row
    require(len(by_page) == len(PAGE_ORDER), "Duplicate Topaz V10 page")
    verify_hash_inventory(report_path.parent)
    return report, by_page


def _native_artifact(
    record: Mapping[str, Any],
    *,
    field: str,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    flags: int = cv2.IMREAD_UNCHANGED,
) -> tuple[Path, np.ndarray]:
    return load_verified_png(
        record,
        root=V8_ROOT,
        field=field,
        expected_shape=shape,
        expected_dtype=dtype,
        flags=flags,
    )


def load_native_page(
    page: str,
    page_row: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    size = NATIVE_SIZES[page]
    source = page_row.get("source")
    require(isinstance(source, dict), f"Missing native source record: {page}")
    baseline_record = source.get("current_final_texture")
    require(isinstance(baseline_record, dict), f"Missing current-final texture record: {page}")
    baseline_path, baseline = _native_artifact(
        baseline_record,
        field=f"native.{page}.baseline",
        shape=(size, size, 3),
        dtype=np.uint8,
        flags=cv2.IMREAD_COLOR,
    )
    support = source.get("uv_support")
    require(isinstance(support, dict), f"Missing native UV support: {page}")
    artifacts = support.get("artifacts")
    require(isinstance(artifacts, dict), f"Missing native support artifacts: {page}")
    inventory_record = support.get("hash_inventory")
    require(isinstance(inventory_record, dict), f"Missing native support inventory record: {page}")
    inventory_path = resolve_record_path(
        inventory_record,
        root=V8_ROOT,
        field=f"native.{page}.support_hash_inventory",
    )

    if page != "tires":
        occupancy_path, occupancy = _native_artifact(
            artifacts["occupancy_mask"],
            field=f"native.{page}.occupancy",
            shape=(size, size),
            dtype=np.uint8,
        )
        object_path, object_owner = _native_artifact(
            artifacts["object_owner"],
            field=f"native.{page}.object_owner",
            shape=(size, size),
            dtype=np.uint16,
        )
        chart_path, chart_owner = _native_artifact(
            artifacts["chart_owner"],
            field=f"native.{page}.chart_owner",
            shape=(size, size),
            dtype=np.uint16,
        )
        support_inputs = {
            "occupancy_mask": {"path": str(occupancy_path), "sha256": sha256_file(occupancy_path)},
            "object_owner": {"path": str(object_path), "sha256": sha256_file(object_path)},
            "chart_owner": {"path": str(chart_path), "sha256": sha256_file(chart_path)},
            "kind": support.get("kind"),
            "hash_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        }
        report_record = support.get("report")
        require(isinstance(report_record, dict), f"Missing native reprojection report: {page}")
        report_path = resolve_record_path(
            report_record,
            root=V8_ROOT,
            field=f"native.{page}.reprojection_report",
        )
        support_inputs["report"] = {"path": str(report_path), "sha256": sha256_file(report_path)}
    else:
        occupancy_path, occupancy = _native_artifact(
            artifacts["occupied_mask"],
            field="native.tires.occupancy",
            shape=(size, size),
            dtype=np.uint8,
        )
        chart_path, chart_owner = _native_artifact(
            artifacts["chart_owner_id"],
            field="native.tires.chart_owner",
            shape=(size, size),
            dtype=np.uint16,
        )
        piece_path, piece_owner = _native_artifact(
            artifacts["piece_owner_id"],
            field="native.tires.piece_owner",
            shape=(size, size),
            dtype=np.uint16,
        )
        mapping_raw = support.get("piece_owner_normalization")
        require(isinstance(mapping_raw, dict) and mapping_raw, "Missing tire piece-owner normalization")
        mapping = {int(piece): int(owner) for piece, owner in mapping_raw.items()}
        observed = set(np.unique(piece_owner[piece_owner > 0]).tolist())
        require(observed == set(mapping), f"Tire piece-owner values drift: {sorted(observed)}")
        object_owner = np.zeros_like(piece_owner, dtype=np.uint16)
        for piece, owner in sorted(mapping.items()):
            object_owner[piece_owner == piece] = owner
        del piece_owner
        support_inputs = {
            "occupancy_mask": {"path": str(occupancy_path), "sha256": sha256_file(occupancy_path)},
            "piece_owner": {"path": str(piece_path), "sha256": sha256_file(piece_path)},
            "chart_owner": {"path": str(chart_path), "sha256": sha256_file(chart_path)},
            "piece_owner_normalization": {str(piece): owner for piece, owner in sorted(mapping.items())},
            "kind": support.get("kind"),
            "hash_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        }

    require(set(np.unique(occupancy).tolist()).issubset({0, 255}), f"Nonbinary native occupancy: {page}")
    occupied = occupancy > 0
    require(np.array_equal(occupied, object_owner > 0), f"Native object support drift: {page}")
    require(np.array_equal(occupied, chart_owner > 0), f"Native chart support drift: {page}")
    declared_size = page_row.get("native_resolution")
    require(declared_size == [size, size], f"Input manifest native resolution drift: {page}")
    return baseline, occupied, object_owner, chart_owner, {
        "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
        "support": support_inputs,
        "native_support_sha256": {
            "occupancy_decoded": sha256_array(occupancy),
            "object_owner_decoded": sha256_array(object_owner),
            "chart_owner_decoded": sha256_array(chart_owner),
        },
    }


def native_material_partition(
    page: str,
    page_row: Mapping[str, Any],
    occupied: np.ndarray,
    object_owner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if page != "cockpit":
        group = page_row["material_masks"]["full_occupied"]
        source_ids = tuple(int(value) for value in group["source_object_ids"])
        owner_values = np.asarray([value + 1 for value in source_ids], dtype=np.uint16)
        configured = np.isin(object_owner, owner_values)
        require(np.array_equal(configured, occupied), f"Native full-page object mask differs from occupancy: {page}")
        return occupied.copy(), np.zeros_like(occupied), {
            "generated_groups": ["full_occupied"],
            "excluded_groups": [],
            "generated_source_object_ids": list(source_ids),
            "excluded_source_object_ids": [],
            "generated_texels": int(occupied.sum()),
            "excluded_texels": 0,
            "partition_exact": True,
        }

    groups = page_row.get("material_masks")
    require(isinstance(groups, dict) and set(groups) == {"leather", "plastics", "carbon", "excluded"}, "Cockpit material groups drift")
    generated = np.zeros_like(occupied)
    excluded = np.zeros_like(occupied)
    group_metrics: dict[str, Any] = {}
    configured_source_ids: set[int] = set()
    for group_name in ("leather", "plastics", "carbon", "excluded"):
        group = groups[group_name]
        eligible = bool(group.get("eligible_for_generation"))
        require(eligible is (group_name != "excluded"), f"Cockpit generation eligibility drift: {group_name}")
        source_ids = tuple(int(value) for value in group["source_object_ids"])
        require(not (configured_source_ids & set(source_ids)), f"Cockpit source-object groups overlap: {group_name}")
        configured_source_ids.update(source_ids)
        owner_values = np.asarray([value + 1 for value in source_ids], dtype=np.uint16)
        mask = np.isin(object_owner, owner_values)
        require(bool(np.any(mask)), f"Empty native cockpit material group: {group_name}")
        require(not bool(np.any(mask & ~occupied)), f"Native cockpit group escapes occupancy: {group_name}")
        if group_name == "excluded":
            require(not bool(np.any(generated & mask)), "Native cockpit exclusion overlaps generated groups")
            excluded |= mask
        else:
            require(not bool(np.any((generated | excluded) & mask)), f"Native cockpit generated groups overlap: {group_name}")
            generated |= mask
        group_metrics[group_name] = {
            "source_object_ids": list(source_ids),
            "owner_values": [value + 1 for value in source_ids],
            "texels": int(mask.sum()),
            "eligible_for_generation": eligible,
        }
        del mask
    require(not bool(np.any(generated & excluded)), "Native cockpit generated/excluded masks overlap")
    require(np.array_equal(generated | excluded, occupied), "Native cockpit material partition does not exactly cover occupancy")
    observed_source_ids = {int(value) - 1 for value in np.unique(object_owner[occupied])}
    require(observed_source_ids == configured_source_ids, f"Native cockpit object set drift: {sorted(observed_source_ids)}")
    true_metals = np.isin(object_owner, np.asarray([24, 38], dtype=np.uint16))
    require(not bool(np.any(true_metals & generated)), "True cockpit metals entered generated mask")
    require(bool(np.all(true_metals <= excluded)), "True cockpit metals are not fully excluded")
    return generated, excluded, {
        "generated_groups": ["leather", "plastics", "carbon"],
        "excluded_groups": ["excluded"],
        "generated_source_object_ids": sorted(configured_source_ids - set(groups["excluded"]["source_object_ids"])),
        "excluded_source_object_ids": list(groups["excluded"]["source_object_ids"]),
        "true_metal_source_object_ids": [23, 37],
        "generated_texels": int(generated.sum()),
        "excluded_texels": int(excluded.sum()),
        "groups": group_metrics,
        "partition_exact": True,
        "true_metals_excluded": True,
    }


def process_page(
    page: str,
    *,
    input_page: Mapping[str, Any],
    topaz_row: Mapping[str, Any],
    topaz_report_path: Path,
    staging: Path,
) -> dict[str, Any]:
    size = NATIVE_SIZES[page]
    baseline, occupied, object_owner, chart_owner, native_provenance = load_native_page(page, input_page)
    donor_path, donor = load_verified_png(
        topaz_row["output"],
        root=topaz_report_path.parent,
        field=f"topaz.{page}.output",
        expected_shape=(size, size, 3),
        expected_dtype=np.uint8,
        flags=cv2.IMREAD_COLOR,
    )
    generated_mask, excluded_mask, partition = native_material_partition(
        page,
        input_page,
        occupied,
        object_owner,
    )
    surface, merge_metrics = inject_candidates(
        baseline,
        {page: donor},
        {page: generated_mask},
        occupied,
    )
    excluded_preserved = bool(np.array_equal(surface[excluded_mask], baseline[excluded_mask]))
    require(excluded_preserved, f"Native excluded material changed: {page}")
    unchanged_occupied = occupied & ~generated_mask
    require(np.array_equal(surface[unchanged_occupied], baseline[unchanged_occupied]), f"Generated donor escaped native mask: {page}")
    remaining_new_black = generated_mask & np.all(surface == 0, axis=2) & np.any(baseline != 0, axis=2)
    require(not bool(np.any(remaining_new_black)), f"Native new exact-black remains: {page}")

    object_for_gutter = object_owner.copy()
    chart_for_gutter = chart_owner.copy()
    padded_mask, gutter = grow_owned_gutters_in_place(
        surface,
        object_for_gutter,
        chart_for_gutter,
        padding=16,
    )
    require(gutter["cross_domain_averaging"] is False, f"Native cross-domain gutter averaging: {page}")
    require(gutter["rgb_object_chart_copied_from_same_single_donor"] is True, f"Native single-donor gutter proof failed: {page}")

    output_path = staging / f"{page}_seedream_v10_final.png"
    output = emit_png(
        output_path,
        surface,
        root=staging,
        semantic="Topaz_V10_inserted_only_under_authoritative_native_material_mask_with_new_black_fallback_and_16px_single_donor_gutters",
    )
    introduced = merge_metrics["candidates"][0]["introduced_exact_black_detected_and_replaced"]
    row = {
        "page": page,
        "native_resolution": [size, size],
        "inputs": {
            "current_final_baseline": native_provenance["baseline"],
            "native_uv_support": native_provenance["support"],
            "native_support_sha256": native_provenance["native_support_sha256"],
            "topaz_v10": {
                "path": str(donor_path),
                "sha256": sha256_file(donor_path),
                "decoded_pixel_sha256": sha256_array(donor),
            },
        },
        "native_material_partition": partition,
        "metrics": {
            "occupied_texels": int(occupied.sum()),
            "generated_mask_texels": int(generated_mask.sum()),
            "excluded_baseline_texels": int(excluded_mask.sum()),
            "changed_generated_texels": merge_metrics["candidates"][0]["changed_masked_texels"],
            "introduced_exact_black_detected_and_replaced": introduced,
            "remaining_new_exact_black_generated_texels": int(remaining_new_black.sum()),
            "changed_occupied_outside_generated_mask": 0,
        },
        "gutter": gutter,
        "output": output,
        "hard_gates": {
            "authoritative_native_support_verified": True,
            "generated_rgb_inserted_only_inside_native_material_mask": True,
            "cockpit_exclusion_preserved_from_baseline": excluded_preserved,
            "cockpit_true_metals_excluded": page != "cockpit" or partition["true_metals_excluded"],
            "zero_changed_occupied_outside_generated_mask": True,
            "zero_remaining_new_exact_black": int(remaining_new_black.sum()) == 0,
            "deterministic_16px_single_donor_gutters": True,
            "zero_cross_domain_averaging": gutter["cross_domain_averaging"] is False,
            "native_output_dimensions_exact": output["shape"] == [size, size, 3],
        },
    }
    del baseline, occupied, object_owner, chart_owner, donor
    del generated_mask, excluded_mask, surface, object_for_gutter, chart_for_gutter
    del padded_mask, remaining_new_black, unchanged_occupied
    gc.collect()
    return row


def build_composite(
    *,
    input_manifest_path: Path,
    topaz_report_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    input_manifest_path = input_manifest_path.expanduser().resolve()
    topaz_report_path = topaz_report_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    input_manifest, input_pages, _ = load_input_manifest(input_manifest_path)
    input_sha = sha256_file(input_manifest_path)
    topaz_report, topaz_pages = validate_topaz_report(
        topaz_report_path,
        input_manifest_sha256=input_sha,
    )
    require(not output_root.exists(), f"Refusing to overwrite final V10 textures: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=str(output_root.parent)))
    try:
        rows: list[dict[str, Any]] = []
        for page in PAGE_ORDER:
            print(f"V10_NATIVE_COMPOSITE_START {page}", flush=True)
            row = process_page(
                page,
                input_page=input_pages[page],
                topaz_row=topaz_pages[page],
                topaz_report_path=topaz_report_path,
                staging=staging,
            )
            rows.append(row)
            print(f"V10_NATIVE_COMPOSITE_DONE {page} {row['output']['file_sha256']}", flush=True)
        report = {
            "schema": FINAL_TEXTURE_SCHEMA,
            "status": "passed",
            "input_manifest": {
                "path": str(input_manifest_path),
                "sha256": input_sha,
                "schema": input_manifest["schema"],
            },
            "topaz_report": {
                "path": str(topaz_report_path),
                "sha256": sha256_file(topaz_report_path),
                "schema": topaz_report["schema"],
                "retry_report_sha256": topaz_report["retry_report_sha256"],
                "seedream_selection": topaz_report["seedream_selection"],
            },
            "policy": {
                "uv_layout_changed": False,
                "occupied_support_changed": False,
                "generated_rgb_scope": "authoritative_native_material_mask_only",
                "cockpit_exclusions": "current_final_baseline_byte_exact_on_all_excluded_occupied_texels_including_true_metals_23_37",
                "new_black_policy": "fallback_to_current_final_native_baseline",
                "gutter_policy": "16px_exact_single_donor_no_cross_domain_average",
                "output_publication": "fresh_staging_directory_then_atomic_rename_no_overwrite",
            },
            "pages": rows,
            "hard_gates": {
                "exact_seven_pages": len(rows) == 7,
                "page_order_exact": tuple(row["page"] for row in rows) == PAGE_ORDER,
                "all_page_gates_passed": all(all(row["hard_gates"].values()) for row in rows),
                "zero_remaining_new_exact_black": all(row["metrics"]["remaining_new_exact_black_generated_texels"] == 0 for row in rows),
                "zero_generated_changes_outside_native_masks": all(row["metrics"]["changed_occupied_outside_generated_mask"] == 0 for row in rows),
                "all_native_sizes_exact": all(row["output"]["shape"] == [NATIVE_SIZES[row["page"]], NATIVE_SIZES[row["page"]], 3] for row in rows),
                "cockpit_exclusions_preserved": next(row for row in rows if row["page"] == "cockpit")["hard_gates"]["cockpit_exclusion_preserved_from_baseline"],
                "cockpit_true_metals_excluded": next(row for row in rows if row["page"] == "cockpit")["hard_gates"]["cockpit_true_metals_excluded"],
                "strict_three_retry_overrides_bound": tuple(topaz_report["seedream_selection"]["overridden_candidate_ids"]) == RETRY_CANDIDATE_ORDER,
                "atomic_output_directory": True,
                "no_overwrite": True,
            },
        }
        require(all(report["hard_gates"].values()), "Final V10 texture report hard gate failed")
        atomic_write_json(staging / "report.json", report)
        write_hash_inventory(staging)
        require(not output_root.exists(), f"Refusing to overwrite final V10 textures: {output_root}")
        os.rename(staging, output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> dict[str, Any]:
    common = self_test_common()
    occupied = np.zeros((20, 20), dtype=bool)
    occupied[2:18, 2:18] = True
    owner = np.zeros((20, 20), dtype=np.uint16)
    owner[2:10, 2:18] = 20  # source object 19, leather
    owner[10:14, 2:18] = 18  # source object 17, plastic
    owner[14:16, 2:18] = 8  # source object 7, carbon
    owner[16:18, 2:18] = 24  # source object 23, excluded true metal
    fake_row = {
        "material_masks": {
            "leather": {"source_object_ids": [19], "eligible_for_generation": True},
            "plastics": {"source_object_ids": [17], "eligible_for_generation": True},
            "carbon": {"source_object_ids": [7], "eligible_for_generation": True},
            "excluded": {"source_object_ids": [23], "eligible_for_generation": False},
        }
    }
    generated, excluded, partition = native_material_partition("cockpit", fake_row, occupied, owner)
    require(np.array_equal(generated | excluded, occupied), "Synthetic native cockpit partition failed")
    require(not bool(np.any(generated & excluded)), "Synthetic native cockpit partition overlaps")
    require(bool(np.all(excluded[16:18, 2:18])), "Synthetic true-metal exclusion failed")
    return {
        "status": "passed",
        "common": common,
        "native_mask_partition": partition["partition_exact"],
        "true_metal_excluded": partition["true_metals_excluded"],
        "network_or_blender_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=INPUT_ROOT / "manifest.json")
    parser.add_argument("--topaz-report", type=Path, default=DEFAULT_TOPAZ_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print("V10_NATIVE_COMPOSITE_SELF_TEST=" + json.dumps(self_test(), sort_keys=True))
        return 0
    report = build_composite(
        input_manifest_path=args.input_manifest,
        topaz_report_path=args.topaz_report,
        output_root=args.output_root,
    )
    print(
        "V10_NATIVE_COMPOSITE="
        + json.dumps(
            {
                "status": report["status"],
                "output_root": str(args.output_root.expanduser().resolve()),
                "pages": [row["page"] for row in report["pages"]],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
