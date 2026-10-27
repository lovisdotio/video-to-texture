#!/usr/bin/env python3
"""Prepare seven deterministic V10 donors and optionally upscale them with Topaz.

The default mode is local preflight only.  Network execution additionally
requires both ``--execute`` and the exact ``TOPAZ_V10_NETWORK_GO`` token.
Interrupted network runs resume from a bound ``.incomplete`` directory; a
complete output directory is published by one atomic rename and is never
overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from v10_postprocess_common import (
    CANDIDATE_ORDER,
    COCKPIT_CANDIDATES,
    INPUT_ROOT,
    NATIVE_SIZES,
    PAGE_ORDER,
    PREPARED_TOPAZ_SCHEMA,
    RETRY_CANDIDATE_ORDER,
    SCRIPT_DIR,
    SEEDREAM_SCHEMA,
    SEEDREAM_RETRY_SCHEMA,
    TARGET_SIZE,
    TOPAZ_SCHEMA,
    artifact_record,
    atomic_write_json,
    emit_png,
    grow_owned_gutters_in_place,
    inject_candidates,
    load_input_manifest,
    load_json,
    load_verified_png,
    read_image,
    require,
    resolve_record_path,
    self_test_common,
    sha256_array,
    sha256_file,
    verify_hash_inventory,
    write_hash_inventory,
)


MODEL = "fal-ai/topaz/upscale/image"
MODEL_ARGUMENTS = {
    "model": "High Fidelity V2",
    "crop_to_fill": False,
    "output_format": "png",
    "subject_detection": "All",
    "face_enhancement": False,
    "sharpen": 0.1,
    "denoise": 0.0,
    "fix_compression": 0.05,
}
NETWORK_TOKEN = "TOPAZ_V10_NETWORK_GO"
DEFAULT_PREPARED_ROOT = SCRIPT_DIR / "topaz_inputs_2k"
DEFAULT_TOPAZ_ROOT = SCRIPT_DIR / "fal" / "topaz_v10"


def discover_seedream_report() -> Path:
    matches = sorted((SCRIPT_DIR / "runs").glob("*/seedream_v10_results_manifest.json"))
    passed: list[Path] = []
    for path in matches:
        try:
            report = load_json(path)
        except RuntimeError:
            continue
        if report.get("schema") == SEEDREAM_SCHEMA and report.get("status") == "passed":
            passed.append(path.resolve())
    require(len(passed) == 1, "Pass --seedream-report explicitly unless exactly one passed V10 Seedream report exists")
    return passed[0]


def validate_seedream_report(
    report_path: Path,
    *,
    input_manifest_sha256: str,
    input_manifest_root: Path,
    input_candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report_path = report_path.expanduser().resolve()
    report = load_json(report_path)
    require(report.get("schema") == SEEDREAM_SCHEMA, "Seedream V10 report schema drift")
    require(report.get("status") == "passed", "Seedream V10 report did not pass")
    require(tuple(report.get("candidate_order", ())) == CANDIDATE_ORDER, "Seedream V10 top-level candidate order drift")
    gates = report.get("gates")
    if gates is not None:
        require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()), "Seedream V10 report gate failed")
    rows = report.get("candidates")
    require(isinstance(rows, list), "Seedream V10 report has no candidates list")
    require(tuple(row.get("candidate_id") for row in rows if isinstance(row, dict)) == CANDIDATE_ORDER, "Seedream V10 result candidate order/set drift")
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        expected_input = input_candidates[candidate_id]
        require(row.get("page") == expected_input.get("page"), f"Seedream result page drift: {candidate_id}")
        require(row.get("material_group") == expected_input.get("material_group"), f"Seedream material group drift: {candidate_id}")
        require(row.get("source_object_ids") == expected_input.get("source_object_ids"), f"Seedream source object IDs drift: {candidate_id}")
        require(row.get("excluded_source_object_ids") == expected_input.get("excluded_source_object_ids"), f"Seedream excluded object IDs drift: {candidate_id}")
        require(row.get("input_manifest_sha256") == input_manifest_sha256, f"Seedream input-manifest binding drift: {candidate_id}")
        target_mask = row.get("target_material_mask")
        require(isinstance(target_mask, dict), f"Seedream target-mask record missing: {candidate_id}")
        require(target_mask.get("file_sha256") == expected_input["material_mask"]["file_sha256"], f"Seedream target-mask file hash drift: {candidate_id}")
        require(target_mask.get("decoded_pixel_sha256") == expected_input["material_mask"]["decoded_pixel_sha256"], f"Seedream target-mask decoded hash drift: {candidate_id}")
        mask_path, mask_image = load_verified_png(
            target_mask,
            root=report_path.parent,
            field=f"seedream.{candidate_id}.target_material_mask",
            expected_shape=(TARGET_SIZE, TARGET_SIZE),
            expected_dtype=np.uint8,
        )
        input_mask_path = resolve_record_path(
            expected_input["material_mask"],
            root=input_manifest_root,
            field=f"inputs.{candidate_id}.material_mask",
        )
        require(sha256_file(mask_path) == sha256_file(input_mask_path), f"Seedream target-mask artifact differs from V10 input: {candidate_id}")
        del mask_image
        result_record = row.get("result_record")
        require(isinstance(result_record, dict), f"Seedream raw-result record missing: {candidate_id}")
        resolve_record_path(result_record, root=report_path.parent, field=f"seedream.{candidate_id}.result_record")
        output = row.get("output")
        require(isinstance(output, dict), f"Seedream output record missing: {candidate_id}")
        _, image = load_verified_png(
            output,
            root=report_path.parent,
            field=f"seedream.{candidate_id}.output",
            expected_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            expected_dtype=np.uint8,
            flags=cv2.IMREAD_COLOR,
        )
        del image
        by_candidate[candidate_id] = row
    require(len(by_candidate) == len(CANDIDATE_ORDER), "Duplicate Seedream V10 candidate")
    return report, by_candidate


def validate_retry_report(
    report_path: Path,
    *,
    input_manifest_sha256: str,
    input_manifest_root: Path,
    input_candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the exact three no-mask retries and their downstream masks."""

    report_path = report_path.expanduser().resolve()
    report = load_json(report_path)
    require(report.get("schema") == SEEDREAM_RETRY_SCHEMA, "Seedream V10 retry report schema drift")
    require(report.get("status") == "passed", "Seedream V10 retry report did not pass")
    require(tuple(report.get("candidate_order", ())) == RETRY_CANDIDATE_ORDER, "Seedream V10 retry candidate order/set drift")
    gates = report.get("gates")
    require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()), "Seedream V10 retry report gate failed")
    rows = report.get("candidates")
    require(isinstance(rows, list), "Seedream V10 retry report has no candidates list")
    require(tuple(row.get("candidate_id") for row in rows if isinstance(row, dict)) == RETRY_CANDIDATE_ORDER, "Seedream V10 retry row order/set drift")
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        expected_input = input_candidates[candidate_id]
        require(row.get("page") == expected_input.get("page"), f"Retry page drift: {candidate_id}")
        require(row.get("material_group") == expected_input.get("material_group"), f"Retry material group drift: {candidate_id}")
        require(row.get("source_object_ids") == expected_input.get("source_object_ids"), f"Retry source object IDs drift: {candidate_id}")
        require(row.get("excluded_source_object_ids") == expected_input.get("excluded_source_object_ids"), f"Retry excluded object IDs drift: {candidate_id}")
        require(row.get("input_manifest_sha256") == input_manifest_sha256, f"Retry input-manifest binding drift: {candidate_id}")
        retry_policy = row.get("retry_policy")
        require(isinstance(retry_policy, dict), f"Retry policy missing: {candidate_id}")
        require(retry_policy.get("material_mask_uploaded") is False, f"Retry unexpectedly uploaded a material mask: {candidate_id}")
        require(retry_policy.get("references_start_at_figure") == 3, f"Retry reference numbering drift: {candidate_id}")
        require(retry_policy.get("output_required") == "full_color_rgb_albedo_never_white_mask_or_diagram", f"Retry output policy drift: {candidate_id}")

        target_mask = row.get("target_material_mask_downstream_only")
        require(isinstance(target_mask, dict), f"Retry downstream target-mask record missing: {candidate_id}")
        require(target_mask.get("file_sha256") == expected_input["material_mask"]["file_sha256"], f"Retry target-mask file hash drift: {candidate_id}")
        require(target_mask.get("decoded_pixel_sha256") == expected_input["material_mask"]["decoded_pixel_sha256"], f"Retry target-mask decoded hash drift: {candidate_id}")
        mask_path, mask_image = load_verified_png(
            target_mask,
            root=report_path.parent,
            field=f"retry.{candidate_id}.target_material_mask_downstream_only",
            expected_shape=(TARGET_SIZE, TARGET_SIZE),
            expected_dtype=np.uint8,
        )
        input_mask_path = resolve_record_path(
            expected_input["material_mask"],
            root=input_manifest_root,
            field=f"inputs.{candidate_id}.material_mask",
        )
        require(sha256_file(mask_path) == sha256_file(input_mask_path), f"Retry downstream target mask differs from V10 input: {candidate_id}")
        del mask_image

        result_record = row.get("result_record")
        require(isinstance(result_record, dict), f"Retry raw-result record missing: {candidate_id}")
        resolve_record_path(result_record, root=report_path.parent, field=f"retry.{candidate_id}.result_record")
        output = row.get("output")
        require(isinstance(output, dict), f"Retry output record missing: {candidate_id}")
        _, image = load_verified_png(
            output,
            root=report_path.parent,
            field=f"retry.{candidate_id}.output",
            expected_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            expected_dtype=np.uint8,
            flags=cv2.IMREAD_COLOR,
        )
        del image
        by_candidate[candidate_id] = row
    require(tuple(by_candidate) == RETRY_CANDIDATE_ORDER, "Duplicate or reordered Seedream V10 retries")
    return report, by_candidate


def _input_artifact(
    record: Mapping[str, Any],
    *,
    input_root: Path,
    field: str,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    flags: int = cv2.IMREAD_UNCHANGED,
) -> tuple[Path, np.ndarray]:
    return load_verified_png(
        record,
        root=input_root,
        field=field,
        expected_shape=shape,
        expected_dtype=dtype,
        flags=flags,
    )


def build_prepared_inputs(
    *,
    input_manifest_path: Path,
    seedream_report_path: Path,
    retry_report_path: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    input_manifest_path = input_manifest_path.expanduser().resolve()
    seedream_report_path = seedream_report_path.expanduser().resolve()
    retry_report_path = retry_report_path.expanduser().resolve() if retry_report_path is not None else None
    output_root = output_root.expanduser().resolve()
    input_manifest, input_pages, input_candidates = load_input_manifest(input_manifest_path)
    input_sha = sha256_file(input_manifest_path)
    seedream_report, seedream_candidates = validate_seedream_report(
        seedream_report_path,
        input_manifest_sha256=input_sha,
        input_manifest_root=input_manifest_path.parent,
        input_candidates=input_candidates,
    )
    seedream_sha = sha256_file(seedream_report_path)
    selected_candidates = dict(seedream_candidates)
    selection_source = {candidate_id: "original_nine_candidate_report" for candidate_id in CANDIDATE_ORDER}
    retry_report: dict[str, Any] | None = None
    retry_candidates: dict[str, dict[str, Any]] = {}
    retry_sha: str | None = None
    if retry_report_path is not None:
        retry_report, retry_candidates = validate_retry_report(
            retry_report_path,
            input_manifest_sha256=input_sha,
            input_manifest_root=input_manifest_path.parent,
            input_candidates=input_candidates,
        )
        require(tuple(retry_candidates) == RETRY_CANDIDATE_ORDER, "Retry override set is not exactly exterior/running/tires")
        retry_sha = sha256_file(retry_report_path)
        for candidate_id in RETRY_CANDIDATE_ORDER:
            selected_candidates[candidate_id] = retry_candidates[candidate_id]
            selection_source[candidate_id] = "nomask_retry_override"
    require(tuple(selected_candidates) == CANDIDATE_ORDER, "Selected Seedream candidate set/order drift")

    if output_root.exists():
        report = load_json(output_root / "manifest.json")
        require(report.get("schema") == PREPARED_TOPAZ_SCHEMA and report.get("status") == "passed", "Existing prepared Topaz inputs did not pass")
        require(report.get("input_manifest", {}).get("sha256") == input_sha, "Existing prepared inputs bind another input manifest")
        require(report.get("seedream_report", {}).get("sha256") == seedream_sha, "Existing prepared inputs bind another Seedream report")
        existing_retry = report.get("retry_report")
        if retry_sha is None:
            require(existing_retry is None, "Existing prepared inputs unexpectedly bind a retry report")
        else:
            require(isinstance(existing_retry, dict) and existing_retry.get("sha256") == retry_sha, "Existing prepared inputs bind another retry report")
        require(tuple(row.get("page") for row in report.get("pages", ())) == PAGE_ORDER, "Existing prepared page order drift")
        require(isinstance(report.get("hard_gates"), dict) and all(value is True for value in report["hard_gates"].values()), "Existing prepared-input hard gate failed")
        verify_hash_inventory(output_root)
        for row in report["pages"]:
            _, image = load_verified_png(
                row["output"],
                root=output_root,
                field=f"prepared.{row['page']}.output",
                expected_shape=(TARGET_SIZE, TARGET_SIZE, 3),
                expected_dtype=np.uint8,
                flags=cv2.IMREAD_COLOR,
            )
            del image
        return report

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=str(output_root.parent)))
    rows: list[dict[str, Any]] = []
    try:
        for page in PAGE_ORDER:
            page_row = input_pages[page]
            outputs = page_row["outputs"]
            _, baseline = _input_artifact(
                outputs["working_texture"],
                input_root=input_manifest_path.parent,
                field=f"inputs.{page}.working_texture",
                shape=(TARGET_SIZE, TARGET_SIZE, 3),
                dtype=np.uint8,
                flags=cv2.IMREAD_COLOR,
            )
            _, occupancy_raw = _input_artifact(
                outputs["occupancy_mask"],
                input_root=input_manifest_path.parent,
                field=f"inputs.{page}.occupancy",
                shape=(TARGET_SIZE, TARGET_SIZE),
                dtype=np.uint8,
            )
            _, object_owner = _input_artifact(
                outputs["object_owner"],
                input_root=input_manifest_path.parent,
                field=f"inputs.{page}.object_owner",
                shape=(TARGET_SIZE, TARGET_SIZE),
                dtype=np.uint16,
            )
            _, chart_owner = _input_artifact(
                outputs["chart_owner"],
                input_root=input_manifest_path.parent,
                field=f"inputs.{page}.chart_owner",
                shape=(TARGET_SIZE, TARGET_SIZE),
                dtype=np.uint16,
            )
            occupied = occupancy_raw > 0
            require(np.array_equal(occupied, object_owner > 0), f"2K object support drift: {page}")
            require(np.array_equal(occupied, chart_owner > 0), f"2K chart support drift: {page}")

            candidate_ids = COCKPIT_CANDIDATES if page == "cockpit" else (page,)
            donor_images: dict[str, np.ndarray] = {}
            masks: dict[str, np.ndarray] = {}
            candidate_provenance: list[dict[str, Any]] = []
            for candidate_id in candidate_ids:
                input_candidate = input_candidates[candidate_id]
                result_candidate = selected_candidates[candidate_id]
                selected_report_path = retry_report_path if selection_source[candidate_id] == "nomask_retry_override" else seedream_report_path
                require(selected_report_path is not None, f"Missing selected report path: {candidate_id}")
                donor_path, donor = load_verified_png(
                    result_candidate["output"],
                    root=selected_report_path.parent,
                    field=f"selected_seedream.{candidate_id}.output",
                    expected_shape=(TARGET_SIZE, TARGET_SIZE, 3),
                    expected_dtype=np.uint8,
                    flags=cv2.IMREAD_COLOR,
                )
                mask_path, mask_raw = _input_artifact(
                    input_candidate["material_mask"],
                    input_root=input_manifest_path.parent,
                    field=f"inputs.{candidate_id}.material_mask",
                    shape=(TARGET_SIZE, TARGET_SIZE),
                    dtype=np.uint8,
                )
                require(set(np.unique(mask_raw).tolist()).issubset({0, 255}), f"Nonbinary candidate mask: {candidate_id}")
                expected_mask = np.isin(
                    object_owner,
                    np.asarray([int(value) + 1 for value in input_candidate["source_object_ids"]], dtype=np.uint16),
                )
                require(np.array_equal(mask_raw > 0, expected_mask), f"Candidate mask/object-owner disagreement: {candidate_id}")
                donor_images[candidate_id] = donor
                masks[candidate_id] = mask_raw > 0
                candidate_provenance.append(
                    {
                        "candidate_id": candidate_id,
                        "page": page,
                        "material_group": input_candidate["material_group"],
                        "source_object_ids": input_candidate["source_object_ids"],
                        "selection_source": selection_source[candidate_id],
                        "selection_report": {
                            "path": str(selected_report_path),
                            "sha256": sha256_file(selected_report_path),
                        },
                        "seedream_output": {
                            "path": str(donor_path),
                            "file_sha256": sha256_file(donor_path),
                            "decoded_pixel_sha256": sha256_array(donor),
                        },
                        "material_mask": {
                            "path": str(mask_path),
                            "file_sha256": sha256_file(mask_path),
                            "decoded_pixel_sha256": sha256_array(mask_raw),
                        },
                    }
                )
                del mask_raw, expected_mask

            surface, merge_metrics = inject_candidates(
                baseline,
                donor_images,
                masks,
                occupied,
            )
            if page == "cockpit":
                excluded_record = page_row["material_masks"]["excluded"]["artifact"]
                excluded_path, excluded_raw = _input_artifact(
                    excluded_record,
                    input_root=input_manifest_path.parent,
                    field="inputs.cockpit.excluded_mask",
                    shape=(TARGET_SIZE, TARGET_SIZE),
                    dtype=np.uint8,
                )
                excluded = excluded_raw > 0
                expected_excluded = np.isin(
                    object_owner,
                    np.asarray(
                        [int(value) + 1 for value in page_row["material_masks"]["excluded"]["source_object_ids"]],
                        dtype=np.uint16,
                    ),
                )
                require(np.array_equal(excluded, expected_excluded), "Cockpit exclusion mask/object-owner disagreement")
                union = excluded.copy()
                for mask in masks.values():
                    require(not bool(np.any(union & mask)), "Cockpit candidate mask overlaps exclusion")
                    union |= mask
                require(np.array_equal(union, occupied), "Cockpit candidate plus exclusion masks do not exactly cover occupancy")
                require(np.array_equal(surface[excluded], baseline[excluded]), "Cockpit excluded RGB changed before Topaz")
                exclusion = {
                    "path": str(excluded_path),
                    "file_sha256": sha256_file(excluded_path),
                    "source_object_ids": page_row["material_masks"]["excluded"]["source_object_ids"],
                    "texels": int(excluded.sum()),
                    "baseline_preserved_byte_exact_before_topaz": True,
                }
                del excluded_raw, excluded, expected_excluded, union
            else:
                require(np.array_equal(next(iter(masks.values())), occupied), f"Full-page material mask differs from occupancy: {page}")
                exclusion = None

            object_for_gutter = object_owner.copy()
            chart_for_gutter = chart_owner.copy()
            padded_mask, gutter = grow_owned_gutters_in_place(
                surface,
                object_for_gutter,
                chart_for_gutter,
                padding=16,
            )
            require(gutter["cross_domain_averaging"] is False, f"2K cross-domain gutter averaging: {page}")
            require(gutter["rgb_object_chart_copied_from_same_single_donor"] is True, f"2K gutter donor proof failed: {page}")
            output_path = staging / page / f"{page}_seedream_v10_topaz_input_2k.png"
            output = emit_png(
                output_path,
                surface,
                root=staging,
                semantic="V10_Seedream_candidates_inserted_only_under_fixed_material_masks_plus_16px_single_donor_gutters",
            )
            rows.append(
                {
                    "page": page,
                    "native_resolution": [NATIVE_SIZES[page], NATIVE_SIZES[page]],
                    "topaz_upscale_factor": NATIVE_SIZES[page] // TARGET_SIZE,
                    "candidate_ids": list(candidate_ids),
                    "candidates": candidate_provenance,
                    "cockpit_exclusion": exclusion,
                    "merge_metrics": merge_metrics,
                    "gutter": gutter,
                    "output": output,
                    "hard_gates": {
                        "generated_rgb_inserted_only_inside_material_masks": True,
                        "new_exact_black_fell_back_to_current_final": merge_metrics["remaining_new_exact_black_texels"] == 0,
                        "cockpit_exclusions_preserved_before_topaz": page != "cockpit" or exclusion["baseline_preserved_byte_exact_before_topaz"],
                        "deterministic_16px_single_donor_gutters": True,
                        "zero_cross_domain_averaging": gutter["cross_domain_averaging"] is False,
                    },
                }
            )
            del baseline, occupancy_raw, object_owner, chart_owner, occupied
            del donor_images, masks, surface, object_for_gutter, chart_for_gutter, padded_mask
            gc.collect()

        manifest = {
            "schema": PREPARED_TOPAZ_SCHEMA,
            "status": "passed",
            "input_manifest": {
                "path": str(input_manifest_path),
                "sha256": input_sha,
                "schema": input_manifest["schema"],
            },
            "seedream_report": {
                "path": str(seedream_report_path),
                "sha256": seedream_sha,
                "schema": seedream_report["schema"],
            },
            "retry_report": (
                {
                    "path": str(retry_report_path),
                    "sha256": retry_sha,
                    "schema": retry_report["schema"],
                    "overridden_candidate_ids": list(RETRY_CANDIDATE_ORDER),
                }
                if retry_report is not None and retry_report_path is not None
                else None
            ),
            "selection": {
                "candidate_source": selection_source,
                "overridden_candidate_ids": list(RETRY_CANDIDATE_ORDER) if retry_report is not None else [],
                "original_candidate_ids": [
                    candidate_id
                    for candidate_id in CANDIDATE_ORDER
                    if selection_source[candidate_id] == "original_nine_candidate_report"
                ],
            },
            "policy": {
                "exact_nine_seedream_candidates": True,
                "exact_seven_topaz_inputs": True,
                "retry_override_policy": "if supplied, exactly exterior_opaque/running_metal/tires replace rejected originals; all other six remain original",
                "cockpit_merge": "leather_plastics_carbon_under_exact_disjoint_masks; excluded_occupancy_from_current_final_baseline",
                "generated_output_defines_uv_support": False,
                "new_black_policy": "fallback_to_current_final_baseline_inside_candidate_mask",
                "gutter_policy": "16px_single_exact_donor_no_cross_domain_average",
            },
            "pages": rows,
            "hard_gates": {
                "exact_nine_seedream_candidates": len(seedream_candidates) == 9,
                "retry_override_set_exact": retry_report is None or tuple(retry_candidates) == RETRY_CANDIDATE_ORDER,
                "selected_candidate_set_exact": tuple(selected_candidates) == CANDIDATE_ORDER,
                "exact_six_originals_retained_when_retry_used": retry_report is None or sum(value == "original_nine_candidate_report" for value in selection_source.values()) == 6,
                "retry_outputs_replace_original_three": retry_report is None or all(selection_source[candidate_id] == "nomask_retry_override" for candidate_id in RETRY_CANDIDATE_ORDER),
                "exact_seven_topaz_inputs": len(rows) == 7,
                "page_order_exact": tuple(row["page"] for row in rows) == PAGE_ORDER,
                "all_page_gates_passed": all(all(row["hard_gates"].values()) for row in rows),
                "atomic_output_directory": True,
                "no_overwrite": True,
            },
        }
        require(all(manifest["hard_gates"].values()), "Prepared Topaz manifest hard gate failed")
        atomic_write_json(staging / "manifest.json", manifest)
        write_hash_inventory(staging)
        require(not output_root.exists(), f"Refusing to overwrite prepared Topaz root: {output_root}")
        os.rename(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _download_resumable(url: str, partial: Path, *, attempts: int = 8) -> None:
    import requests

    partial.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(60, 180)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"TOPAZ_V10_DOWNLOAD_RETRY {attempt}/{attempts} bytes={partial.stat().st_size if partial.exists() else 0}", flush=True)
            time.sleep(min(2.0 * attempt, 10.0))
    raise RuntimeError(f"Topaz CDN download failed after {attempts} attempts: {last_error}")


def _result_output_url(result: Mapping[str, Any]) -> str:
    image = result.get("image")
    require(isinstance(image, dict) and isinstance(image.get("url"), str), "Topaz result lacks image URL")
    return str(image["url"])


def _run_topaz_page(
    row: Mapping[str, Any],
    *,
    prepared_root: Path,
    staging: Path,
) -> dict[str, Any]:
    import fal_client

    page = str(row["page"])
    native_size = NATIVE_SIZES[page]
    factor = native_size // TARGET_SIZE
    source = resolve_record_path(row["output"], root=prepared_root, field=f"prepared.{page}.output")
    source_sha = sha256_file(source)
    page_dir = staging / page
    page_dir.mkdir(parents=True, exist_ok=True)
    result_path = page_dir / "topaz_result.json"
    request_ids: list[str] = []
    if result_path.exists():
        saved = load_json(result_path)
        require(saved.get("model") == MODEL, f"Stale Topaz model metadata: {page}")
        require(saved.get("input_sha256") == source_sha, f"Stale Topaz input metadata: {page}")
        require(saved.get("model_arguments") == {**MODEL_ARGUMENTS, "upscale_factor": factor}, f"Stale Topaz arguments: {page}")
        result = saved.get("result")
        require(isinstance(result, dict), f"Missing saved Topaz result: {page}")
        request_id = saved.get("request_id")
        print(f"TOPAZ_V10_RESUME_RESULT {page} {request_id}", flush=True)
    else:
        print(f"TOPAZ_V10_UPLOAD {page}", flush=True)
        image_url = fal_client.upload_file(source)
        arguments = {"image_url": image_url, **MODEL_ARGUMENTS, "upscale_factor": factor}
        print(f"TOPAZ_V10_START {page} x{factor}", flush=True)
        result = fal_client.subscribe(
            MODEL,
            arguments=arguments,
            with_logs=False,
            on_enqueue=lambda request_id: (
                request_ids.append(request_id),
                print(f"TOPAZ_V10_REQUEST_ID {page} {request_id}", flush=True),
            ),
            start_timeout=120,
            client_timeout=1800,
        )
        request_id = request_ids[0] if request_ids else None
        atomic_write_json(
            result_path,
            {
                "schema": "car-city.ferrari-targeted-topaz-raw-result/v10",
                "model": MODEL,
                "page": page,
                "request_id": request_id,
                "input_sha256": source_sha,
                "model_arguments": {**MODEL_ARGUMENTS, "upscale_factor": factor},
                "result": result,
            },
        )

    output_url = _result_output_url(result)
    output_path = page_dir / f"{page}_seedream_v10_topaz_{native_size}.png"
    complete_path = page_dir / "complete.json"
    if complete_path.exists():
        complete = load_json(complete_path)
        require(complete.get("input_sha256") == source_sha, f"Stale completed Topaz input: {page}")
        require(not output_path.with_suffix(".partial.png").exists(), f"Stale Topaz partial beside completed page: {page}")
        _, image = load_verified_png(
            complete["output"],
            root=staging,
            field=f"topaz.{page}.output",
            expected_shape=(native_size, native_size, 3),
            expected_dtype=np.uint8,
            flags=cv2.IMREAD_COLOR,
        )
        del image
        return complete["page_row"]

    if not output_path.exists():
        partial = output_path.with_suffix(".partial.png")
        image = cv2.imread(str(partial), cv2.IMREAD_COLOR) if partial.exists() else None
        if image is None or image.shape != (native_size, native_size, 3) or image.dtype != np.uint8:
            _download_resumable(output_url, partial)
            image = read_image(partial, flags=cv2.IMREAD_COLOR)
        require(image.shape == (native_size, native_size, 3) and image.dtype == np.uint8, f"Bad Topaz output: {page}: {image.shape}/{image.dtype}")
        try:
            os.link(partial, output_path)
        except FileExistsError as exc:
            raise RuntimeError(f"Refusing to overwrite Topaz output: {output_path}") from exc
        partial.unlink()
    require(not output_path.with_suffix(".partial.png").exists(), f"Stale Topaz partial beside complete output: {page}")
    image = read_image(output_path, flags=cv2.IMREAD_COLOR)
    require(image.shape == (native_size, native_size, 3) and image.dtype == np.uint8, f"Bad Topaz output after reopen: {page}")
    output = artifact_record(
        output_path,
        image,
        root=staging,
        semantic="Topaz_High_Fidelity_V2_upscale_of_mask_fused_Seedream_V10_2K_donor",
    )
    page_row = {
        "page": page,
        "native_resolution": [native_size, native_size],
        "upscale_factor": factor,
        "request_id": request_id,
        "input": {
            "path": str(source),
            "file_sha256": source_sha,
            "prepared_record": row["output"],
        },
        "raw_result": {
            "path": result_path.relative_to(staging).as_posix(),
            "file_sha256": sha256_file(result_path),
        },
        "output": output,
        "hard_gates": {
            "native_dimensions_exact": output["shape"] == [native_size, native_size, 3],
            "lossless_png_reopened": True,
            "result_saved_before_download": True,
            "input_hash_bound": True,
        },
    }
    complete = {
        "schema": "car-city.ferrari-targeted-topaz-page-completion/v10",
        "status": "passed",
        "page": page,
        "input_sha256": source_sha,
        "page_row": page_row,
        "output": output,
    }
    atomic_write_json(complete_path, complete)
    print(f"TOPAZ_V10_DONE {page} {output['file_sha256']}", flush=True)
    del image
    return page_row


def _verify_complete_topaz(
    output_root: Path,
    *,
    prepared_sha: str,
    input_sha: str,
    seedream_sha: str,
    retry_sha: str | None,
    seedream_selection: Mapping[str, Any],
) -> dict[str, Any]:
    report = load_json(output_root / "report.json")
    require(report.get("schema") == TOPAZ_SCHEMA and report.get("status") == "passed", "Existing Topaz V10 report did not pass")
    require(report.get("prepared_inputs_manifest", {}).get("sha256") == prepared_sha, "Existing Topaz V10 prepared-input binding drift")
    require(report.get("input_manifest_sha256") == input_sha, "Existing Topaz V10 input binding drift")
    require(report.get("seedream_report_sha256") == seedream_sha, "Existing Topaz V10 Seedream binding drift")
    require(report.get("retry_report_sha256") == retry_sha, "Existing Topaz V10 retry-report binding drift")
    require(report.get("seedream_selection") == seedream_selection, "Existing Topaz V10 Seedream selection drift")
    require(report.get("model") == MODEL, "Existing Topaz V10 model drift")
    require(report.get("model_arguments_without_factor") == MODEL_ARGUMENTS, "Existing Topaz V10 model arguments drift")
    require(tuple(row.get("page") for row in report.get("pages", ())) == PAGE_ORDER, "Existing Topaz V10 page order drift")
    require(isinstance(report.get("hard_gates"), dict) and all(value is True for value in report["hard_gates"].values()), "Existing Topaz V10 hard gate failed")
    verify_hash_inventory(output_root)
    for row in report["pages"]:
        size = NATIVE_SIZES[row["page"]]
        _, image = load_verified_png(
            row["output"],
            root=output_root,
            field=f"topaz.{row['page']}.output",
            expected_shape=(size, size, 3),
            expected_dtype=np.uint8,
            flags=cv2.IMREAD_COLOR,
        )
        del image
    return report


def run_topaz(
    *,
    prepared_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    prepared_root = prepared_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    prepared_manifest_path = prepared_root / "manifest.json"
    prepared = load_json(prepared_manifest_path)
    require(prepared.get("schema") == PREPARED_TOPAZ_SCHEMA and prepared.get("status") == "passed", "Prepared Topaz V10 inputs did not pass")
    require(tuple(row.get("page") for row in prepared.get("pages", ())) == PAGE_ORDER, "Prepared Topaz page order drift")
    verify_hash_inventory(prepared_root)
    prepared_sha = sha256_file(prepared_manifest_path)
    input_sha = str(prepared["input_manifest"]["sha256"])
    seedream_sha = str(prepared["seedream_report"]["sha256"])
    retry_sha = prepared.get("retry_report", {}).get("sha256") if isinstance(prepared.get("retry_report"), dict) else None
    seedream_selection = prepared.get("selection")
    require(isinstance(seedream_selection, dict), "Prepared Topaz inputs lack Seedream selection provenance")
    if output_root.exists():
        return _verify_complete_topaz(
            output_root,
            prepared_sha=prepared_sha,
            input_sha=input_sha,
            seedream_sha=seedream_sha,
            retry_sha=retry_sha,
            seedream_selection=seedream_selection,
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.incomplete")
    state_payload = {
        "schema": "car-city.ferrari-targeted-topaz-run-state/v10",
        "prepared_inputs_manifest_sha256": prepared_sha,
        "input_manifest_sha256": input_sha,
        "seedream_report_sha256": seedream_sha,
        "retry_report_sha256": retry_sha,
        "seedream_selection": seedream_selection,
        "page_order": list(PAGE_ORDER),
        "model": MODEL,
        "model_arguments_without_factor": MODEL_ARGUMENTS,
    }
    state_path = staging / "run_state.json"
    if staging.exists():
        require(load_json(state_path) == state_payload, f"Stale/incompatible Topaz staging directory: {staging}")
    else:
        staging.mkdir()
        atomic_write_json(state_path, state_payload)

    rows_by_page: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_run_topaz_page, row, prepared_root=prepared_root, staging=staging): row["page"]
            for row in prepared["pages"]
        }
        for future in concurrent.futures.as_completed(future_map):
            page = future_map[future]
            rows_by_page[page] = future.result()
    rows = [rows_by_page[page] for page in PAGE_ORDER]
    report = {
        "schema": TOPAZ_SCHEMA,
        "status": "passed",
        "model": MODEL,
        "model_arguments_without_factor": MODEL_ARGUMENTS,
        "prepared_inputs_manifest": {
            "path": str(prepared_manifest_path),
            "sha256": prepared_sha,
            "schema": PREPARED_TOPAZ_SCHEMA,
        },
        "input_manifest_sha256": input_sha,
        "seedream_report_sha256": seedream_sha,
        "retry_report_sha256": retry_sha,
        "seedream_selection": seedream_selection,
        "pages": rows,
        "hard_gates": {
            "exact_seven_pages": len(rows) == 7,
            "page_order_exact": tuple(row["page"] for row in rows) == PAGE_ORDER,
            "all_page_gates_passed": all(all(row["hard_gates"].values()) for row in rows),
            "all_native_dimensions_exact": all(row["output"]["shape"] == [NATIVE_SIZES[row["page"]], NATIVE_SIZES[row["page"]], 3] for row in rows),
            "result_metadata_saved_before_download": True,
            "resumable_downloads": True,
            "atomic_output_directory": True,
            "no_overwrite": True,
        },
    }
    require(all(report["hard_gates"].values()), "Topaz V10 report hard gate failed")
    report_path = staging / "report.json"
    if report_path.exists():
        require(load_json(report_path) == report, "Existing staged Topaz report differs from deterministic result")
    else:
        atomic_write_json(report_path, report)
    if (staging / "hashes.sha256").exists():
        verify_hash_inventory(staging)
    else:
        write_hash_inventory(staging)
    require(not output_root.exists(), f"Refusing to overwrite Topaz V10 root: {output_root}")
    os.rename(staging, output_root)
    return report


def self_test() -> dict[str, Any]:
    common = self_test_common()
    require(tuple(NATIVE_SIZES[page] // TARGET_SIZE for page in PAGE_ORDER) == (4, 2, 2, 2, 4, 2, 4), "Topaz factor contract drift")
    require(len(CANDIDATE_ORDER) == 9 and len(PAGE_ORDER) == 7, "V10 cardinality contract drift")
    require(set(COCKPIT_CANDIDATES) == {"cockpit_leather", "cockpit_plastics", "cockpit_carbon"}, "Cockpit candidate contract drift")
    require(RETRY_CANDIDATE_ORDER == ("exterior_opaque", "running_metal", "tires"), "Retry override contract drift")
    with tempfile.TemporaryDirectory(prefix="topaz-v10-self-test-") as temporary:
        proof = Path(temporary) / "no_overwrite.json"
        atomic_write_json(proof, {"value": 1})
        overwrite_refused = False
        try:
            atomic_write_json(proof, {"value": 2})
        except RuntimeError:
            overwrite_refused = True
        require(overwrite_refused and load_json(proof) == {"value": 1}, "Atomic no-overwrite contract failed")
    return {
        "status": "passed",
        "common": common,
        "nine_candidates": True,
        "seven_pages": True,
        "strict_three_retry_overrides": list(RETRY_CANDIDATE_ORDER),
        "atomic_no_overwrite": True,
        "native_factors": {page: NATIVE_SIZES[page] // TARGET_SIZE for page in PAGE_ORDER},
        "network_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=INPUT_ROOT / "manifest.json")
    parser.add_argument("--seedream-report", type=Path)
    parser.add_argument(
        "--retry-report",
        type=Path,
        help="Optional strict no-mask override report; must contain exactly exterior_opaque, running_metal and tires",
    )
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TOPAZ_ROOT)
    parser.add_argument("--execute", action="store_true", help="Allow the seven Fal/Topaz calls after local preflight")
    parser.add_argument("--go-token", default="", help=f"Must equal {NETWORK_TOKEN} with --execute")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print("TOPAZ_V10_SELF_TEST=" + json.dumps(self_test(), sort_keys=True))
        return 0
    require(1 <= args.workers <= 3, "--workers must be between 1 and 3")
    seedream_report = args.seedream_report or discover_seedream_report()
    prepared = build_prepared_inputs(
        input_manifest_path=args.input_manifest,
        seedream_report_path=seedream_report,
        retry_report_path=args.retry_report,
        output_root=args.prepared_root,
    )
    print(
        "TOPAZ_V10_PREFLIGHT="
        + json.dumps(
            {
                "status": prepared["status"],
                "prepared_root": str(args.prepared_root.expanduser().resolve()),
                "pages": [row["page"] for row in prepared["pages"]],
                "retry_overrides": prepared["selection"]["overridden_candidate_ids"],
                "network_called": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.execute:
        return 0
    require(args.go_token == NETWORK_TOKEN, f"Network execution requires --go-token {NETWORK_TOKEN}")
    require(bool(os.environ.get("FAL_KEY")), "FAL_KEY is unavailable")
    report = run_topaz(
        prepared_root=args.prepared_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    print(
        "TOPAZ_V10_COMPLETE="
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
