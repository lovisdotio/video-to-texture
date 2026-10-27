#!/usr/bin/env python3
"""Plan or execute the nine strictly masked Seedream V10 candidate edits.

The default mode is an offline preflight.  Network access is impossible unless
both ``--execute`` and the exact ``--go-token`` are supplied.  Every remote
request ID is journalled atomically and can be resumed without resubmission.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from targeted_seedream_v10_common import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_RUNS_ROOT,
    EXPECTED_CANDIDATE_ORDER,
    MODEL,
    NETWORK_GO_TOKEN,
    REFERENCE_POLICY,
    RESULT_MANIFEST_SCHEMA,
    RUN_CONTRACT_SCHEMA,
    RUN_STATE_SCHEMA,
    TARGET_SIZE,
    ValidatedInputs,
    atomic_write_json,
    build_prompt,
    candidate_local_inputs,
    canonical_json_bytes,
    load_json,
    read_image,
    require,
    select_candidates,
    sha256_array,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_inputs,
    validate_references,
)


SCRIPT_PATH = Path(__file__).resolve()
PREFLIGHT_SCHEMA = "car-city.ferrari-targeted-seedream-preflight/v10"
REQUEST_PLAN_SCHEMA = "car-city.ferrari-targeted-seedream-request-plan/v10"
REQUEST_SUBMISSION_SCHEMA = "car-city.ferrari-targeted-seedream-request-submission/v10"
REQUEST_RESULT_SCHEMA = "car-city.ferrari-targeted-seedream-candidate-result/v10"
UPLOAD_CACHE_SCHEMA = "car-city.ferrari-targeted-seedream-upload-cache/v10"
ERROR_SCHEMA = "car-city.ferrari-targeted-seedream-error/v10"
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def _relative_to(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    require(resolved.is_relative_to(resolved_root), f"Path escapes run root: {resolved}")
    return resolved.relative_to(resolved_root).as_posix()


def _artifact_from_local_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": record["path"],
        "file_sha256": record["file_sha256"],
        "decoded_pixel_sha256": record["decoded_pixel_sha256"],
        "shape": list(record["shape"]),
        "dtype": record["dtype"],
        "bytes": record["bytes"],
    }


def build_preflight(
    validated: ValidatedInputs,
    candidate_ids: Sequence[str],
    reference_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    unique_files: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        local_inputs = candidate_local_inputs(validated, candidate_id)
        reference_keys = REFERENCE_POLICY[candidate_id]
        prompt = build_prompt(candidate_id, reference_keys)
        figures: list[dict[str, Any]] = []
        for record in local_inputs:
            row = dict(record)
            figures.append(row)
            unique_files.setdefault(
                row["file_sha256"],
                {
                    "path": row["path"],
                    "file_sha256": row["file_sha256"],
                    "bytes": row["bytes"],
                },
            )
        for offset, key in enumerate(reference_keys, start=4):
            ref = reference_records[key]
            row = {
                "figure": offset,
                "kind": "reference",
                "reference_key": key,
                **dict(ref),
            }
            figures.append(row)
            unique_files.setdefault(
                row["file_sha256"],
                {
                    "path": row["path"],
                    "file_sha256": row["file_sha256"],
                    "bytes": row["bytes"],
                },
            )
        require(len(figures) <= 10, f"Candidate {candidate_id}: Seedream input count exceeds 10")
        require([row["figure"] for row in figures] == list(range(1, len(figures) + 1)), f"Candidate {candidate_id}: figure numbering drift")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "page": validated.candidates[candidate_id]["page"],
                "material_group": validated.candidates[candidate_id]["material_group"],
                "source_object_ids": validated.candidates[candidate_id]["source_object_ids"],
                "excluded_source_object_ids": validated.candidates[candidate_id]["excluded_source_object_ids"],
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "figures": figures,
                "figure_count": len(figures),
                "model_arguments_without_urls": {
                    "image_size": {"width": TARGET_SIZE, "height": TARGET_SIZE},
                    "num_images": 1,
                    "output_format": "png",
                    "enable_safety_checker": True,
                },
                "expected_output": {
                    "shape": [TARGET_SIZE, TARGET_SIZE, 3],
                    "dtype": "uint8",
                    "semantic": "lighting-neutral masked albedo donor; not an authoritative UV support",
                },
            }
        )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "network_performed": False,
        "model": MODEL,
        "input_manifest": {
            "path": str(validated.manifest_path),
            "file_sha256": validated.manifest_sha256,
        },
        "input_audit": validated.audit,
        "candidate_order": list(candidate_ids),
        "candidates": candidates,
        "reference_records": {key: dict(reference_records[key]) for key in reference_records},
        "unique_upload_file_count": len(unique_files),
        "unique_upload_files": list(unique_files.values()),
        "network_guard": {
            "requires_execute_flag": True,
            "requires_exact_go_token": NETWORK_GO_TOKEN,
            "requires_fal_key_environment": True,
        },
    }


def _contract_core(run_id: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": RUN_CONTRACT_SCHEMA,
        "status": "frozen",
        "run_id": run_id,
        "model": MODEL,
        "runner": {
            "path": str(SCRIPT_PATH),
            "file_sha256": sha256_file(SCRIPT_PATH),
        },
        "input_manifest": dict(preflight["input_manifest"]),
        "input_audit": dict(preflight["input_audit"]),
        "candidate_order": list(preflight["candidate_order"]),
        "candidates": list(preflight["candidates"]),
        "reference_records": dict(preflight["reference_records"]),
        "fal_arguments_contract": {
            "image_size": {"width": TARGET_SIZE, "height": TARGET_SIZE},
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": True,
        },
        "resume_policy": {
            "request_id_journalled_immediately_after_submit": True,
            "submitted_request_resumed_by_model_and_request_id": True,
            "completed_outputs_verified_before_skip": True,
            "automatic_resubmission_of_recorded_request": False,
            "exactly_once_limitation": "provider exposes no idempotency key; a process death between remote submit and local submitted.json can leave one orphan request",
        },
    }


def _open_run(run_id: str, runs_root: Path, preflight: Mapping[str, Any], resume: bool) -> tuple[Path, dict[str, Any], str]:
    require(bool(RUN_ID_RE.fullmatch(run_id)), f"Unsafe run ID: {run_id!r}")
    root = runs_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_root = (root / run_id).resolve()
    require(run_root.parent == root, "Run path escaped runs root")
    contract_path = run_root / "run_contract.json"
    expected_core = _contract_core(run_id, preflight)
    if resume:
        require(run_root.is_dir(), f"Resume run does not exist: {run_root}")
        existing = load_json(contract_path)
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        require(
            canonical_json_bytes(comparable) == canonical_json_bytes(expected_core),
            "Resume contract differs from current code, inputs, prompts, references or selection",
        )
        contract = existing
    else:
        require(not run_root.exists(), f"Run already exists; use --resume or a new --run-id: {run_root}")
        os.mkdir(run_root)
        contract = {**expected_core, "created_utc": utc_now()}
        atomic_write_json(contract_path, contract)
    contract_sha256 = sha256_file(contract_path)
    return run_root, contract, contract_sha256


def _state_path(run_root: Path) -> Path:
    return run_root / "state.json"


def _write_state(
    run_root: Path,
    run_id: str,
    contract_sha256: str,
    statuses: Mapping[str, Mapping[str, Any]],
    *,
    status: str,
) -> None:
    atomic_write_json(
        _state_path(run_root),
        {
            "schema": RUN_STATE_SCHEMA,
            "status": status,
            "run_id": run_id,
            "run_contract_sha256": contract_sha256,
            "updated_utc": utc_now(),
            "candidates": dict(statuses),
        },
    )


def _upload_specs(preflight: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for candidate in preflight["candidates"]:
        for figure in candidate["figures"]:
            file_hash = figure["file_sha256"]
            path = Path(figure["path"]).resolve()
            existing = specs.get(file_hash)
            if existing is not None:
                require(existing["path"] == str(path), f"Same hash unexpectedly maps to multiple paths: {file_hash}")
                continue
            specs[file_hash] = {
                "path": str(path),
                "file_sha256": file_hash,
                "bytes": path.stat().st_size,
            }
    return specs


def _load_upload_cache(run_root: Path, contract_sha256: str) -> dict[str, Any]:
    path = run_root / "uploaded_urls.json"
    if not path.is_file():
        return {
            "schema": UPLOAD_CACHE_SCHEMA,
            "run_contract_sha256": contract_sha256,
            "updated_utc": utc_now(),
            "artifacts": {},
        }
    cache = load_json(path)
    require(cache.get("schema") == UPLOAD_CACHE_SCHEMA, "Upload cache schema drift")
    require(cache.get("run_contract_sha256") == contract_sha256, "Upload cache contract hash drift")
    require(isinstance(cache.get("artifacts"), dict), "Upload cache artifacts must be an object")
    return cache


def _ensure_uploads(
    fal_client_module: Any,
    run_root: Path,
    contract_sha256: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    cache_path = run_root / "uploaded_urls.json"
    cache = _load_upload_cache(run_root, contract_sha256)
    specs = _upload_specs(preflight)
    cached_keys = set(cache["artifacts"])
    require(cached_keys.issubset(specs), f"Upload cache has files outside current contract: {sorted(cached_keys - set(specs))}")
    for file_hash, spec in specs.items():
        path = Path(spec["path"])
        require(path.is_file(), f"Upload source disappeared: {path}")
        require(path.stat().st_size == spec["bytes"], f"Upload source size drift: {path}")
        require(sha256_file(path) == file_hash, f"Upload source hash drift: {path}")
        existing = cache["artifacts"].get(file_hash)
        if existing is not None:
            require(existing.get("path") == str(path), f"Cached upload path drift: {path}")
            require(existing.get("file_sha256") == file_hash, f"Cached upload hash drift: {path}")
            require(isinstance(existing.get("url"), str) and existing["url"], f"Cached upload URL missing: {path}")
            continue
        print(f"UPLOAD {path.name} {file_hash}", flush=True)
        url = fal_client_module.upload_file(path)
        require(isinstance(url, str) and url, f"Fal upload returned no URL: {path}")
        cache["artifacts"][file_hash] = {
            **spec,
            "url": url,
            "uploaded_utc": utc_now(),
        }
        cache["updated_utc"] = utc_now()
        atomic_write_json(cache_path, cache)
    require(set(cache["artifacts"]) == set(specs), "Upload cache is incomplete after upload phase")
    return cache


def _candidate_preflight(preflight: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = [row for row in preflight["candidates"] if row["candidate_id"] == candidate_id]
    require(len(rows) == 1, f"Missing or duplicate candidate preflight: {candidate_id}")
    return rows[0]


def _request_paths(run_root: Path, candidate_id: str) -> tuple[Path, Path]:
    request_root = run_root / "requests" / candidate_id
    return request_root / "planned.json", request_root / "submitted.json"


def _result_paths(run_root: Path, candidate_id: str) -> tuple[Path, Path]:
    result_root = run_root / "results" / candidate_id
    return result_root / f"{candidate_id}_seedream_v10_2k.png", result_root / "result.json"


def _request_plan(
    candidate: Mapping[str, Any],
    upload_cache: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    image_urls = [upload_cache["artifacts"][figure["file_sha256"]]["url"] for figure in candidate["figures"]]
    require(len(image_urls) == candidate["figure_count"] <= 10, f"Bad image URL count: {candidate['candidate_id']}")
    arguments = {
        "prompt": candidate["prompt"],
        "image_urls": image_urls,
        **candidate["model_arguments_without_urls"],
    }
    return {
        "schema": REQUEST_PLAN_SCHEMA,
        "status": "planned",
        "run_contract_sha256": contract_sha256,
        "candidate_id": candidate["candidate_id"],
        "page": candidate["page"],
        "model": MODEL,
        "prompt_sha256": candidate["prompt_sha256"],
        "input_file_sha256": [figure["file_sha256"] for figure in candidate["figures"]],
        "arguments": arguments,
    }


def _ensure_request_plan(run_root: Path, expected: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    planned_path, _ = _request_paths(run_root, expected["candidate_id"])
    if planned_path.is_file():
        existing = load_json(planned_path)
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        require(canonical_json_bytes(comparable) == canonical_json_bytes(expected), f"Request plan drift: {expected['candidate_id']}")
        return planned_path, existing
    value = {**dict(expected), "created_utc": utc_now()}
    atomic_write_json(planned_path, value)
    return planned_path, value


def _validate_submitted(path: Path, candidate_id: str, planned_sha256: str, contract_sha256: str) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") == REQUEST_SUBMISSION_SCHEMA, f"Submission schema drift: {candidate_id}")
    require(value.get("status") == "submitted", f"Submission status drift: {candidate_id}")
    require(value.get("candidate_id") == candidate_id, f"Submission candidate drift: {candidate_id}")
    require(value.get("model") == MODEL, f"Submission model drift: {candidate_id}")
    require(value.get("run_contract_sha256") == contract_sha256, f"Submission contract drift: {candidate_id}")
    require(value.get("request_plan_sha256") == planned_sha256, f"Submission plan drift: {candidate_id}")
    for key in ("request_id", "response_url", "status_url", "cancel_url"):
        require(isinstance(value.get(key), str) and value[key], f"Submission lacks {key}: {candidate_id}")
    return value


def _submit_or_resume(
    fal_client_module: Any,
    run_root: Path,
    candidate: Mapping[str, Any],
    upload_cache: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    expected_plan = _request_plan(candidate, upload_cache, contract_sha256)
    planned_path, _ = _ensure_request_plan(run_root, expected_plan)
    planned_sha256 = sha256_file(planned_path)
    _, submitted_path = _request_paths(run_root, candidate["candidate_id"])
    if submitted_path.is_file():
        return _validate_submitted(submitted_path, candidate["candidate_id"], planned_sha256, contract_sha256)
    print(f"SUBMIT {candidate['candidate_id']}", flush=True)
    handle = fal_client_module.submit(MODEL, expected_plan["arguments"], start_timeout=120)
    require(isinstance(handle.request_id, str) and handle.request_id, f"Fal returned no request ID: {candidate['candidate_id']}")
    submitted = {
        "schema": REQUEST_SUBMISSION_SCHEMA,
        "status": "submitted",
        "submitted_utc": utc_now(),
        "candidate_id": candidate["candidate_id"],
        "page": candidate["page"],
        "model": MODEL,
        "run_contract_sha256": contract_sha256,
        "request_plan_sha256": planned_sha256,
        "request_id": handle.request_id,
        "response_url": handle.response_url,
        "status_url": handle.status_url,
        "cancel_url": handle.cancel_url,
    }
    atomic_write_json(submitted_path, submitted)
    print(f"SUBMITTED {candidate['candidate_id']} {handle.request_id}", flush=True)
    return submitted


def _validate_result(run_root: Path, candidate_id: str, contract_sha256: str) -> dict[str, Any] | None:
    output_path, result_path = _result_paths(run_root, candidate_id)
    if not result_path.is_file():
        return None
    result = load_json(result_path)
    require(result.get("schema") == REQUEST_RESULT_SCHEMA, f"Result schema drift: {candidate_id}")
    require(result.get("status") == "passed", f"Result status drift: {candidate_id}")
    require(result.get("candidate_id") == candidate_id, f"Result candidate drift: {candidate_id}")
    require(result.get("run_contract_sha256") == contract_sha256, f"Result contract drift: {candidate_id}")
    output = result.get("output")
    require(isinstance(output, dict), f"Result output missing: {candidate_id}")
    resolved = (run_root / output.get("path", "")).resolve()
    require(resolved == output_path.resolve() and resolved.is_relative_to(run_root.resolve()), f"Result output path drift: {candidate_id}")
    require(output_path.is_file(), f"Result raster missing: {candidate_id}")
    require(output_path.stat().st_size == output.get("bytes"), f"Result byte-size drift: {candidate_id}")
    require(sha256_file(output_path) == output.get("file_sha256"), f"Result file hash drift: {candidate_id}")
    image = read_image(output_path)
    require(image.shape == (TARGET_SIZE, TARGET_SIZE, 3), f"Result shape drift: {candidate_id}")
    require(str(image.dtype) == "uint8", f"Result dtype drift: {candidate_id}")
    require(list(image.shape) == output.get("shape"), f"Result recorded shape drift: {candidate_id}")
    require(str(image.dtype) == output.get("dtype"), f"Result recorded dtype drift: {candidate_id}")
    require(sha256_array(image) == output.get("decoded_pixel_sha256"), f"Result pixel hash drift: {candidate_id}")
    return result


def _download_output(requests_module: Any, url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial-{uuid.uuid4().hex}")
    try:
        with requests_module.get(url, timeout=(30, 300), stream=True) as response:
            response.raise_for_status()
            with temporary.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        image = cv2.imread(str(temporary), cv2.IMREAD_UNCHANGED)
        require(image is not None, f"Seedream output is not a decodable PNG: {url}")
        require(image.shape == (TARGET_SIZE, TARGET_SIZE, 3), f"Seedream output must be RGB 2048x2048, got {image.shape}")
        require(str(image.dtype) == "uint8", f"Seedream output must be uint8, got {image.dtype}")
        os.replace(temporary, output_path)
        try:
            descriptor = os.open(output_path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _retrieve_one(
    fal_client_module: Any,
    requests_module: Any,
    run_root: Path,
    candidate: Mapping[str, Any],
    submitted: Mapping[str, Any],
    contract_sha256: str,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    existing = _validate_result(run_root, candidate_id, contract_sha256)
    if existing is not None:
        return existing
    print(f"WAIT {candidate_id} {submitted['request_id']}", flush=True)
    provider_result = fal_client_module.result(MODEL, submitted["request_id"])
    require(isinstance(provider_result, dict), f"Fal result must be an object: {candidate_id}")
    images = provider_result.get("images")
    require(isinstance(images, list) and len(images) == 1, f"Fal must return exactly one image: {candidate_id}")
    provider_image = images[0]
    require(isinstance(provider_image, dict) and isinstance(provider_image.get("url"), str) and provider_image["url"], f"Fal image URL missing: {candidate_id}")
    output_path, result_path = _result_paths(run_root, candidate_id)
    _download_output(requests_module, provider_image["url"], output_path)
    image = read_image(output_path)
    submitted_path = _request_paths(run_root, candidate_id)[1]
    result = {
        "schema": REQUEST_RESULT_SCHEMA,
        "status": "passed",
        "completed_utc": utc_now(),
        "candidate_id": candidate_id,
        "page": candidate["page"],
        "material_group": candidate["material_group"],
        "model": MODEL,
        "request_id": submitted["request_id"],
        "run_contract_sha256": contract_sha256,
        "request_submission_sha256": sha256_file(submitted_path),
        "input_manifest_sha256": input_manifest_sha256,
        "prompt_sha256": candidate["prompt_sha256"],
        "input_file_sha256": [figure["file_sha256"] for figure in candidate["figures"]],
        "provider_result": provider_result,
        "output": {
            "path": _relative_to(output_path, run_root),
            "semantic": "seedream_v10_candidate_masked_albedo_donor_2k_rgb",
            "file_sha256": sha256_file(output_path),
            "decoded_pixel_sha256": sha256_array(image),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "bytes": output_path.stat().st_size,
        },
        "gates": {
            "one_provider_image": True,
            "decoded_rgb_2048_square": True,
            "generated_output_never_defines_uv_support": True,
            "candidate_mask_required_for_downstream_composite": True,
        },
    }
    atomic_write_json(result_path, result)
    print(f"DONE {candidate_id} {result['output']['file_sha256']}", flush=True)
    return result


def _write_error(run_root: Path, candidate_id: str, phase: str, exc: BaseException) -> Path:
    error_root = run_root / "errors"
    error_root.mkdir(parents=True, exist_ok=True)
    suffix = utc_now().replace(":", "").replace("-", "").replace(".", "")
    path = error_root / f"{candidate_id}_{phase}_{suffix}.json"
    atomic_write_json(
        path,
        {
            "schema": ERROR_SCHEMA,
            "status": "failed",
            "failed_utc": utc_now(),
            "candidate_id": candidate_id,
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    return path


def _final_manifest(
    run_root: Path,
    run_id: str,
    contract_sha256: str,
    preflight: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = tuple(preflight["candidate_order"])
    rows: list[dict[str, Any]] = []
    for candidate_id in selected:
        result = results[candidate_id]
        candidate = _candidate_preflight(preflight, candidate_id)
        result_path = _result_paths(run_root, candidate_id)[1]
        rows.append(
            {
                "candidate_id": candidate_id,
                "page": candidate["page"],
                "material_group": candidate["material_group"],
                "source_object_ids": candidate["source_object_ids"],
                "excluded_source_object_ids": candidate["excluded_source_object_ids"],
                "request_id": result["request_id"],
                "model": MODEL,
                "prompt_sha256": result["prompt_sha256"],
                "input_manifest_sha256": result["input_manifest_sha256"],
                "result_record": {
                    "path": _relative_to(result_path, run_root),
                    "file_sha256": sha256_file(result_path),
                },
                "output": dict(result["output"]),
                "target_material_mask": _artifact_from_local_record(candidate["figures"][2]),
            }
        )
    all_nine = selected == EXPECTED_CANDIDATE_ORDER
    return {
        "schema": RESULT_MANIFEST_SCHEMA,
        "status": "passed" if all_nine else "partial_passed",
        "completed_utc": utc_now(),
        "run_id": run_id,
        "model": MODEL,
        "run_root": str(run_root),
        "run_contract": {
            "path": "run_contract.json",
            "file_sha256": contract_sha256,
        },
        "input_manifest": dict(preflight["input_manifest"]),
        "candidate_order": list(selected),
        "candidates": rows,
        "gates": {
            "all_selected_candidates_complete": True,
            "all_nine_candidates_complete": all_nine,
            "all_outputs_rgb_2048_uint8": True,
            "all_outputs_have_file_and_decoded_pixel_hashes": True,
            "uv_support_must_come_from_pinned_input_manifest": True,
            "downstream_must_apply_target_material_masks": True,
        },
    }


def execute(
    args: argparse.Namespace,
    validated: ValidatedInputs,
    preflight: Mapping[str, Any],
) -> Path:
    require(args.execute, "Internal error: execute() called without --execute")
    require(args.go_token == NETWORK_GO_TOKEN, "Network execution refused: exact --go-token is absent")
    require(bool(os.environ.get("FAL_KEY")), "Network execution refused: FAL_KEY is unavailable")
    require(args.run_id, "--run-id is required with --execute")

    # Lazy imports are deliberate: offline preflight never imports a network client.
    import fal_client
    import requests

    run_root, _contract, contract_sha256 = _open_run(args.run_id, args.runs_root, preflight, args.resume)
    candidate_ids = tuple(preflight["candidate_order"])
    statuses: dict[str, dict[str, Any]] = {candidate_id: {"status": "pending"} for candidate_id in candidate_ids}
    results: dict[str, dict[str, Any]] = {}
    submissions: dict[str, dict[str, Any]] = {}

    for candidate_id in candidate_ids:
        existing = _validate_result(run_root, candidate_id, contract_sha256)
        if existing is not None:
            results[candidate_id] = existing
            statuses[candidate_id] = {
                "status": "passed",
                "request_id": existing["request_id"],
                "output_file_sha256": existing["output"]["file_sha256"],
            }
    _write_state(run_root, args.run_id, contract_sha256, statuses, status="executing")

    upload_cache = _ensure_uploads(fal_client, run_root, contract_sha256, preflight)
    submission_failure: tuple[str, BaseException] | None = None
    for candidate_id in candidate_ids:
        if candidate_id in results:
            continue
        candidate = _candidate_preflight(preflight, candidate_id)
        try:
            submitted = _submit_or_resume(fal_client, run_root, candidate, upload_cache, contract_sha256)
            submissions[candidate_id] = submitted
            statuses[candidate_id] = {"status": "submitted", "request_id": submitted["request_id"]}
            _write_state(run_root, args.run_id, contract_sha256, statuses, status="executing")
        except BaseException as exc:
            error_path = _write_error(run_root, candidate_id, "submit", exc)
            statuses[candidate_id] = {"status": "failed", "error_record": _relative_to(error_path, run_root)}
            submission_failure = (candidate_id, exc)
            _write_state(run_root, args.run_id, contract_sha256, statuses, status="incomplete")
            break

    retrieval_failures: list[tuple[str, BaseException]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(
                _retrieve_one,
                fal_client,
                requests,
                run_root,
                _candidate_preflight(preflight, candidate_id),
                submitted,
                contract_sha256,
                validated.manifest_sha256,
            ): candidate_id
            for candidate_id, submitted in submissions.items()
        }
        for future in concurrent.futures.as_completed(future_map):
            candidate_id = future_map[future]
            try:
                result = future.result()
                results[candidate_id] = result
                statuses[candidate_id] = {
                    "status": "passed",
                    "request_id": result["request_id"],
                    "output_file_sha256": result["output"]["file_sha256"],
                }
            except BaseException as exc:
                error_path = _write_error(run_root, candidate_id, "retrieve", exc)
                statuses[candidate_id] = {"status": "failed", "error_record": _relative_to(error_path, run_root)}
                retrieval_failures.append((candidate_id, exc))
            _write_state(run_root, args.run_id, contract_sha256, statuses, status="executing")

    if submission_failure is not None or retrieval_failures or set(results) != set(candidate_ids):
        _write_state(run_root, args.run_id, contract_sha256, statuses, status="incomplete")
        details: list[str] = []
        if submission_failure is not None:
            details.append(f"submit {submission_failure[0]}: {submission_failure[1]}")
        details.extend(f"retrieve {candidate_id}: {exc}" for candidate_id, exc in retrieval_failures)
        missing = sorted(set(candidate_ids) - set(results))
        details.append(f"missing results: {missing}")
        raise RuntimeError("Seedream V10 run is incomplete; resume with the same command and --resume. " + " | ".join(details))

    manifest = _final_manifest(run_root, args.run_id, contract_sha256, preflight, results)
    manifest_path = run_root / "seedream_v10_results_manifest.json"
    atomic_write_json(manifest_path, manifest)
    _write_state(run_root, args.run_id, contract_sha256, statuses, status="passed")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--candidate", action="append", choices=EXPECTED_CANDIDATE_ORDER)
    parser.add_argument("--emit-preflight", type=Path, help="Write the verified offline preflight report atomically")
    parser.add_argument("--execute", action="store_true", help="Allow the remote execution path (also requires exact --go-token)")
    parser.add_argument("--go-token", default="", help="Second, explicit network execution lock")
    parser.add_argument("--run-id", help="New immutable run directory name; required for --execute")
    parser.add_argument("--resume", action="store_true", help="Resume the exact existing run contract by recorded request IDs")
    parser.add_argument("--max-workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    require(not args.resume or args.execute, "--resume only applies to --execute")
    require(not args.execute or args.go_token == NETWORK_GO_TOKEN, "Network execution refused: pass the exact approved --go-token")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_ids = select_candidates(args.candidate)
    validated = validate_inputs(args.input_root)
    references = validate_references(candidate_ids)
    preflight = build_preflight(validated, candidate_ids, references)
    if args.emit_preflight is not None:
        atomic_write_json(args.emit_preflight.expanduser().resolve(), {**preflight, "created_utc": utc_now()})
        print(args.emit_preflight.expanduser().resolve(), flush=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "network_performed": False,
                "input_manifest_sha256": validated.manifest_sha256,
                "candidate_order": list(candidate_ids),
                "unique_upload_file_count": preflight["unique_upload_file_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.execute:
        return 0
    manifest_path = execute(args, validated, preflight)
    print(manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

