#!/usr/bin/env python3
"""Fail-closed pure contract for texture-only V11 full-bleed integration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
V10_CODE_ROOT = V8_ROOT / "targeted_final_v10"
if str(V10_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(V10_CODE_ROOT))

from targeted_final_v10_common import (  # noqa: E402
    BUILD_SCHEMA as SOURCE_BUILD_SCHEMA,
    canonical_json_sha256,
    load_json,
    read_png_header,
    require,
    sha256,
)


SOURCE_ROOT = V10_CODE_ROOT / "outputs"
SOURCE_BLEND = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v10_final.blend"
SOURCE_ASSEMBLY_REPORT = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v10_final.assembly_report.json"
SOURCE_REOPEN_AUDIT = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v10_final.reopen_audit.json"
SOURCE_BUILD_REPORT = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v10_final.build_report.json"
SOURCE_PINS = {
    SOURCE_BLEND: "eb7808bdd0ff32b5d6994fd62f763f2ca2d0ef943d6eba7ea000f153537ad30f",
    SOURCE_ASSEMBLY_REPORT: "fe4dc43972872809fd6c8b8c3eebc3b898bc5caf36b72b95f95d856913bd6c56",
    SOURCE_REOPEN_AUDIT: "d01a6d4f767889272e4cd8c70be7c2644a91619dc79ea4fb3f234b18fdbaed7e",
    SOURCE_BUILD_REPORT: "ae312a1361e12760e65b4a59694d5bfefaa430f491e0a1ee6b681d7042082bc8",
}

GENERATOR_ROOT = V8_ROOT / "targeted_seedream_v10/textures_final_v11_full_bleed"
GENERATOR_REPORT = GENERATOR_ROOT / "report.json"
GENERATOR_REPORT_SHA256 = "e138adb8f023ff449a799332497dabaaa9f73e0cec8aac59dc7eaf4b0d1cd518"
GENERATOR_SCHEMA = "car-city.ferrari-full-bleed-textures/v11"
INDEPENDENT_AUDIT = V8_ROOT / "targeted_seedream_v11/audits/full_bleed_output_v1/full_bleed_output_independent_audit.json"
INDEPENDENT_AUDIT_SHA256 = "516c5468c104c4c0d1acbbc69ced6d4a51ffb957c7df1657fd9a853b7336b246"
INDEPENDENT_AUDIT_SCHEMA = "car-city.ferrari-v11-full-bleed-independent-output-audit/v1"

PAGE_IDS = (
    "body_hero_8k",
    "exterior_opaque",
    "glass",
    "lights_translucent",
    "running_metal",
    "tires",
    "cockpit",
)
PAGE_SIZES = {
    "body_hero_8k": (8192, 8192),
    "exterior_opaque": (4096, 4096),
    "glass": (4096, 4096),
    "lights_translucent": (4096, 4096),
    "running_metal": (8192, 8192),
    "tires": (4096, 4096),
    "cockpit": (8192, 8192),
}
EXPECTED_FILENAMES = {page: f"{page}_seedream_v11_full_bleed.png" for page in PAGE_IDS}
EXPECTED_MESHES = 66
EXPECTED_TRIANGLES = 179_488
EXPECTED_IMAGE_BINDINGS = 32
EXPECTED_PAGE_MATERIAL_COUNTS = {
    "body_hero_8k": 1,
    "exterior_opaque": 5,
    "glass": 1,
    "lights_translucent": 5,
    "running_metal": 5,
    "tires": 1,
    "cockpit": 14,
}
ASSET_ID = "ferrari_targeted_reunwrap_seedream_v11_full_bleed_final"
IMAGE_SCHEMA = "car-city.ferrari-targeted-final-image/v11-full-bleed"
ASSEMBLY_SCHEMA = "car-city.ferrari-texture-only-final-assembly/v11-full-bleed"
REOPEN_SCHEMA = "car-city.ferrari-texture-only-final-reopen-audit/v11-full-bleed"
BUILD_SCHEMA = "car-city.ferrari-texture-only-final-build/v11-full-bleed"
RENDER_SCHEMA = "car-city.ferrari-texture-only-v11-full-bleed-around-preview/v1"
SHEET_SCHEMA = "car-city.ferrari-texture-only-v11-full-bleed-around5-sheet/v1"
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
WORK_ROOT = SCRIPT_DIR / "work"
RENDER_ROOT = OUTPUT_ROOT / "renders"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextureRecord:
    page_id: str
    path: Path
    sha256: str
    size: tuple[int, int]
    bytes: int
    decoded_pixel_sha256: str
    semantic: str


@dataclass(frozen=True)
class TextureSet:
    generator_report_path: Path
    generator_report_sha256: str
    generator_report_canonical_sha256: str
    independent_audit_path: Path
    independent_audit_sha256: str
    independent_audit_canonical_sha256: str
    asset_id: str
    textures: dict[str, TextureRecord]
    generator_raw: dict[str, Any]
    audit_raw: dict[str, Any]


def validate_source_v10() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in SOURCE_PINS.items():
        require(path.is_file(), f"Pinned V10 source artifact missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"Pinned V10 source drift: {path} ({actual} != {expected})")
        observed[str(path)] = actual
    build = load_json(SOURCE_BUILD_REPORT)
    assembly = load_json(SOURCE_ASSEMBLY_REPORT)
    reopen = load_json(SOURCE_REOPEN_AUDIT)
    require(build.get("schema") == SOURCE_BUILD_SCHEMA, "Pinned V10 build schema drift")
    for label, report in (("build", build), ("assembly", assembly), ("reopen", reopen)):
        require(report.get("status") == "passed", f"Pinned V10 {label} report did not pass")
        gates = report.get("hard_gates")
        require(isinstance(gates, dict) and gates and all(gates.values()), f"Pinned V10 {label} report has a failed gate")
    require(Path(build.get("final_blend", "")).resolve() == SOURCE_BLEND, "Pinned V10 build points to another blend")
    require(build.get("final_blend_sha256") == SOURCE_PINS[SOURCE_BLEND], "Pinned V10 build blend SHA drift")
    require(assembly.get("staged_blend_sha256") == SOURCE_PINS[SOURCE_BLEND], "Pinned V10 assembly blend SHA drift")
    require(reopen.get("opened_blend_sha256") == SOURCE_PINS[SOURCE_BLEND], "Pinned V10 reopen blend SHA drift")
    state = reopen.get("state", {})
    require(state.get("meshes") == EXPECTED_MESHES and state.get("triangles") == EXPECTED_TRIANGLES, "Pinned V10 geometry inventory drift")
    require(len(assembly.get("production_images", {})) == 7, "Pinned V10 production image inventory drift")
    bindings = assembly.get("v10_bindings", {})
    require(len(bindings.get("nodes", [])) == EXPECTED_IMAGE_BINDINGS, "Pinned V10 image binding inventory drift")
    require(bindings.get("page_material_counts") == EXPECTED_PAGE_MATERIAL_COUNTS, "Pinned V10 material/page partition drift")
    return observed


def load_texture_set(*, validate_source: bool = True) -> TextureSet:
    require(GENERATOR_REPORT.is_file(), f"V11 generator report missing: {GENERATOR_REPORT}")
    require(INDEPENDENT_AUDIT.is_file(), f"V11 independent audit missing: {INDEPENDENT_AUDIT}")
    generator_sha = sha256(GENERATOR_REPORT)
    audit_sha = sha256(INDEPENDENT_AUDIT)
    require(generator_sha == GENERATOR_REPORT_SHA256, f"V11 generator report SHA drift: {generator_sha}")
    require(audit_sha == INDEPENDENT_AUDIT_SHA256, f"V11 independent audit SHA drift: {audit_sha}")
    generator = load_json(GENERATOR_REPORT)
    audit = load_json(INDEPENDENT_AUDIT)
    require(generator.get("schema") == GENERATOR_SCHEMA and generator.get("status") == "passed", "V11 generator schema/status failed")
    require(audit.get("schema") == INDEPENDENT_AUDIT_SCHEMA and audit.get("status") == "passed", "V11 independent audit schema/status failed")
    for label, report in (("generator", generator), ("independent audit", audit)):
        gates = report.get("hard_gates")
        require(isinstance(gates, dict) and gates and all(gates.values()), f"V11 {label} contains a failed gate")
    require(generator.get("page_order") == list(PAGE_IDS), "V11 generator page order drift")
    source = generator.get("source_final_v10", {})
    require(source.get("file_sha256") == "2e67f2a945d6b4d5a7fbf4f4693958453159a068fd83077796556cdf73c742a6", "V11 generator source V10 report drift")
    audit_inputs = audit.get("inputs", {})
    require(Path(audit_inputs.get("v11_report", {}).get("path", "")).resolve() == GENERATOR_REPORT, "Independent audit points to another V11 report")
    require(audit_inputs.get("v11_report", {}).get("file_sha256") == generator_sha, "Independent audit V11 report SHA drift")
    require(audit.get("summary", {}).get("pages") == 7, "Independent audit page count drift")
    require(audit.get("summary", {}).get("total_occupied_rgb_mismatch_texels") == 0, "Independent audit found occupied RGB drift")
    require(audit.get("summary", {}).get("total_background_nearest_donor_mismatch_texels") == 0, "Independent audit found donor mismatch")
    require(audit.get("summary", {}).get("total_background_exact_black_after") == 0, "Independent audit found unfilled background")

    rows = generator.get("pages")
    audit_rows = audit.get("pages")
    require(isinstance(rows, list) and [row.get("page") for row in rows] == list(PAGE_IDS), "V11 generator page coverage drift")
    require(isinstance(audit_rows, list) and [row.get("page") for row in audit_rows] == list(PAGE_IDS), "V11 independent audit page coverage drift")
    audit_by_page = {row["page"]: row for row in audit_rows}
    textures: dict[str, TextureRecord] = {}
    for row in rows:
        page = row["page"]
        gates = row.get("hard_gates")
        require(isinstance(gates, dict) and gates and all(gates.values()), f"V11 generator page gate failed: {page}")
        output = row.get("output", {})
        filename = output.get("path")
        require(filename == EXPECTED_FILENAMES[page] and Path(filename).name == filename, f"V11 output filename drift: {page}")
        path = (GENERATOR_ROOT / filename).resolve()
        require(path.parent == GENERATOR_ROOT.resolve() and path.is_file(), f"V11 texture missing/escaping root: {page}")
        expected_sha = output.get("file_sha256")
        require(isinstance(expected_sha, str) and SHA256_RE.fullmatch(expected_sha), f"Invalid V11 texture SHA: {page}")
        require(sha256(path) == expected_sha and path.stat().st_size == output.get("bytes"), f"V11 texture hash/bytes drift: {page}")
        size = PAGE_SIZES[page]
        require(output.get("shape") == [size[1], size[0], 3] and output.get("dtype") == "uint8", f"V11 texture shape/dtype drift: {page}")
        header = read_png_header(path)
        require((header["width"], header["height"], header["bit_depth"], header["color_type"]) == (size[0], size[1], 8, 2), f"V11 PNG header drift: {page}")
        independent = audit_by_page[page]
        independent_gates = independent.get("gates")
        require(independent.get("pass") is True and isinstance(independent_gates, dict) and all(independent_gates.values()), f"Independent page audit failed: {page}")
        measured = independent.get("measured", {})
        require(measured.get("occupied_rgb_mismatch_texels") == 0, f"Occupied RGB changed: {page}")
        require(measured.get("background_nearest_donor_mismatch_texels") == 0, f"Background donor mismatch: {page}")
        require(measured.get("background_exact_black_after") == 0, f"Background not full bleed: {page}")
        audit_output = independent.get("output_v11", {})
        require(Path(audit_output.get("path", "")).resolve() == path, f"Independent audit output path drift: {page}")
        require(audit_output.get("file_sha256") == expected_sha, f"Independent audit output SHA drift: {page}")
        decoded_sha = output.get("decoded_pixel_sha256")
        require(audit_output.get("decoded_pixel_sha256") == decoded_sha, f"Independent decoded hash drift: {page}")
        textures[page] = TextureRecord(page, path, expected_sha, size, path.stat().st_size, decoded_sha, str(output.get("semantic", "")))
    require(tuple(textures) == PAGE_IDS and len({r.sha256 for r in textures.values()}) == 7, "V11 texture inventory uniqueness/order drift")
    if validate_source:
        validate_source_v10()
    return TextureSet(
        GENERATOR_REPORT.resolve(), generator_sha, canonical_json_sha256(generator),
        INDEPENDENT_AUDIT.resolve(), audit_sha, canonical_json_sha256(audit),
        ASSET_ID, textures, generator, audit,
    )


def safe_output_path(name: str) -> Path:
    require(isinstance(name, str) and Path(name).name == name, "Output name must be a basename")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.blend", name) is not None, "Unsafe output blend name")
    output = (OUTPUT_ROOT / name).resolve()
    require(output.parent == OUTPUT_ROOT.resolve() and output != SOURCE_BLEND, "Output escapes V11 root or aliases V10 source")
    return output


def preflight_summary(texture_set: TextureSet) -> dict[str, Any]:
    return {
        "asset_id": texture_set.asset_id,
        "mode": "texture_only_full_bleed_from_pinned_v10",
        "source_blend": str(SOURCE_BLEND),
        "source_blend_sha256": SOURCE_PINS[SOURCE_BLEND],
        "generator_report": str(texture_set.generator_report_path),
        "generator_report_sha256": texture_set.generator_report_sha256,
        "independent_audit": str(texture_set.independent_audit_path),
        "independent_audit_sha256": texture_set.independent_audit_sha256,
        "expected_meshes": EXPECTED_MESHES,
        "expected_triangles": EXPECTED_TRIANGLES,
        "expected_image_bindings": EXPECTED_IMAGE_BINDINGS,
        "textures": {page: {"path": str(r.path), "sha256": r.sha256, "size": list(r.size), "bytes": r.bytes} for page, r in texture_set.textures.items()},
        "occupied_rgb_mismatch_texels": texture_set.audit_raw["summary"]["total_occupied_rgb_mismatch_texels"],
        "background_nearest_donor_mismatch_texels": texture_set.audit_raw["summary"]["total_background_nearest_donor_mismatch_texels"],
    }


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-source-pins", action="store_true", help="Tests only")
    args = parser.parse_args()
    try:
        textures = load_texture_set(validate_source=not args.skip_source_pins)
    except Exception as exc:
        print(f"TARGETED_FINAL_V11_PREFLIGHT_REFUSED: {exc}", file=sys.stderr)
        return 2
    print("TARGETED_FINAL_V11_PREFLIGHT_PASSED=" + json.dumps(preflight_summary(textures), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
