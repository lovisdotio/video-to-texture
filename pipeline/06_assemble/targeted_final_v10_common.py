#!/usr/bin/env python3
"""Fail-closed, pure-Python contract for the texture-only V10 blend build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
SOURCE_ROOT = V8_ROOT / "targeted_final_v9/outputs"
SOURCE_BLEND = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v9_final.blend"
SOURCE_ASSEMBLY_REPORT = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v9_final.assembly_report.json"
SOURCE_REOPEN_AUDIT = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v9_final.reopen_audit.json"
SOURCE_BUILD_REPORT = SOURCE_ROOT / "ferrari_targeted_reunwrap_seedream_v9_final.build_report.json"
SOURCE_PINS = {
    SOURCE_BLEND: "d1a1cc62cf4a890342cd8a9215054036a9a9c53de0ab3e9984cc7898b9ba5b2c",
    SOURCE_ASSEMBLY_REPORT: "a9b504fb824c9f525e0fa7252858c2bc58e2c6ffdf36421f41fec779155c6168",
    SOURCE_REOPEN_AUDIT: "9f73124f667b92d35f84cfc9d1842cb697720eeb5079dd185b58ad4929dc550a",
    SOURCE_BUILD_REPORT: "48c76eb1bf746cd223aa650f37cc855513dc2ca474dcff3f7cd607e55a885d94",
}

TEXTURE_ROOT = V8_ROOT / "targeted_seedream_v10/textures_final_v10"
TEXTURE_REPORT = TEXTURE_ROOT / "report.json"
TEXTURE_REPORT_SHA256 = "2e67f2a945d6b4d5a7fbf4f4693958453159a068fd83077796556cdf73c742a6"
TEXTURE_REPORT_SCHEMA = "car-city.ferrari-targeted-final-textures/v10"

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
EXPECTED_FILENAMES = {
    page_id: f"{page_id}_seedream_v10_final.png" for page_id in PAGE_IDS
}
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
ASSET_ID = "ferrari_targeted_reunwrap_seedream_v10_final"

ASSEMBLY_SCHEMA = "car-city.ferrari-texture-only-final-assembly/v10"
REOPEN_SCHEMA = "car-city.ferrari-texture-only-final-reopen-audit/v10"
BUILD_SCHEMA = "car-city.ferrari-texture-only-final-build/v10"
RENDER_SCHEMA = "car-city.ferrari-texture-only-v10-around-preview/v1"
IMAGE_SCHEMA = "car-city.ferrari-targeted-final-image/v10"

OUTPUT_ROOT = SCRIPT_DIR / "outputs"
WORK_ROOT = SCRIPT_DIR / "work"
RENDER_ROOT = OUTPUT_ROOT / "renders"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """Raised for any missing, stale, ambiguous, or unsafe build prerequisite."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required JSON is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_png_header(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        header = stream.read(33)
    require(len(header) >= 33 and header[:8] == b"\x89PNG\r\n\x1a\n", f"Not a PNG: {path}")
    require(header[12:16] == b"IHDR", f"PNG lacks leading IHDR: {path}")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    return {"width": width, "height": height, "bit_depth": bit_depth, "color_type": color_type}


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
    report_path: Path
    report_sha256: str
    report_canonical_sha256: str
    asset_id: str
    textures: dict[str, TextureRecord]
    raw: dict[str, Any]


def validate_source_v9() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in SOURCE_PINS.items():
        require(path.is_file(), f"Pinned V9 source artifact is missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"Pinned V9 source drift: {path} ({actual} != {expected})")
        observed[str(path)] = actual

    build = load_json(SOURCE_BUILD_REPORT)
    assembly = load_json(SOURCE_ASSEMBLY_REPORT)
    reopen = load_json(SOURCE_REOPEN_AUDIT)
    for label, report in (("build", build), ("assembly", assembly), ("reopen", reopen)):
        require(report.get("status") == "passed", f"Pinned V9 {label} report did not pass")
        gates = report.get("hard_gates")
        require(isinstance(gates, dict) and gates and all(gates.values()), f"Pinned V9 {label} report has a failed gate")
    require(Path(build.get("final_blend", "")).resolve() == SOURCE_BLEND, "Pinned V9 build points to another blend")
    require(build.get("final_blend_sha256") == SOURCE_PINS[SOURCE_BLEND], "Pinned V9 build blend SHA drift")
    require(assembly.get("staged_blend_sha256") == SOURCE_PINS[SOURCE_BLEND], "Pinned V9 assembly blend SHA drift")
    require(reopen.get("opened_blend_sha256") == SOURCE_PINS[SOURCE_BLEND], "Pinned V9 reopen blend SHA drift")
    source_state = reopen.get("state", {})
    require(source_state.get("meshes") == EXPECTED_MESHES, "Pinned V9 mesh inventory drift")
    require(source_state.get("triangles") == EXPECTED_TRIANGLES, "Pinned V9 triangle inventory drift")
    require(len(assembly.get("production_images", {})) == len(PAGE_IDS), "Pinned V9 production-image inventory drift")
    bindings = assembly.get("image_bindings", {})
    require(len(bindings.get("nodes", [])) == EXPECTED_IMAGE_BINDINGS, "Pinned V9 image-binding inventory drift")
    require(bindings.get("page_material_counts") == EXPECTED_PAGE_MATERIAL_COUNTS, "Pinned V9 material/page partition drift")
    return observed


def load_texture_set(*, validate_source: bool = True) -> TextureSet:
    report_path = TEXTURE_REPORT.resolve()
    require(report_path.is_file(), f"V10 texture report is missing: {report_path}")
    report_sha = sha256(report_path)
    require(report_sha == TEXTURE_REPORT_SHA256, f"V10 texture report SHA drift: {report_sha}")
    raw = load_json(report_path)
    require(raw.get("schema") == TEXTURE_REPORT_SCHEMA, "Wrong V10 texture report schema")
    require(raw.get("status") == "passed", "V10 texture report did not pass")
    gates = raw.get("hard_gates")
    require(isinstance(gates, dict) and gates and all(gates.values()), "V10 texture report contains a failed gate")
    policy = raw.get("policy")
    require(isinstance(policy, dict), "V10 texture policy is missing")
    require(policy.get("uv_layout_changed") is False, "V10 texture report declares a UV-layout change")
    require(policy.get("occupied_support_changed") is False, "V10 texture report declares occupied-support drift")

    rows = raw.get("pages")
    require(isinstance(rows, list), "V10 texture page list is missing")
    require([row.get("page") for row in rows if isinstance(row, dict)] == list(PAGE_IDS), "V10 texture page order/coverage drift")
    textures: dict[str, TextureRecord] = {}
    for row in rows:
        require(isinstance(row, dict), "V10 texture page row is invalid")
        page_id = str(row["page"])
        page_gates = row.get("hard_gates")
        require(isinstance(page_gates, dict) and page_gates and all(page_gates.values()), f"V10 texture page gate failed: {page_id}")
        output = row.get("output")
        require(isinstance(output, dict), f"V10 texture output is missing: {page_id}")
        filename = output.get("path")
        require(filename == EXPECTED_FILENAMES[page_id] and Path(filename).name == filename, f"V10 texture filename drift: {page_id}")
        path = (report_path.parent / filename).resolve()
        require(path.parent == report_path.parent, f"V10 texture escapes texture directory: {page_id}")
        require(path.is_file(), f"V10 texture is missing: {page_id}: {path}")
        expected_sha = output.get("file_sha256")
        require(isinstance(expected_sha, str) and SHA256_RE.fullmatch(expected_sha), f"Invalid V10 texture SHA: {page_id}")
        require(sha256(path) == expected_sha, f"V10 texture file SHA drift: {page_id}")
        require(path.stat().st_size == int(output.get("bytes", -1)), f"V10 texture byte count drift: {page_id}")
        expected_size = PAGE_SIZES[page_id]
        require(output.get("shape") == [expected_size[1], expected_size[0], 3], f"V10 texture shape drift: {page_id}")
        require(output.get("dtype") == "uint8", f"V10 texture dtype drift: {page_id}")
        decoded_sha = output.get("decoded_pixel_sha256")
        require(isinstance(decoded_sha, str) and SHA256_RE.fullmatch(decoded_sha), f"Invalid decoded-pixel SHA: {page_id}")
        header = read_png_header(path)
        require((header["width"], header["height"]) == expected_size, f"V10 PNG dimensions drift: {page_id}")
        require(header["bit_depth"] == 8 and header["color_type"] == 2, f"V10 PNG must be 8-bit RGB: {page_id}")
        semantic = output.get("semantic")
        require(isinstance(semantic, str) and semantic, f"V10 texture semantic is missing: {page_id}")
        textures[page_id] = TextureRecord(
            page_id=page_id,
            path=path,
            sha256=expected_sha,
            size=expected_size,
            bytes=path.stat().st_size,
            decoded_pixel_sha256=decoded_sha,
            semantic=semantic,
        )
    require(tuple(textures) == PAGE_IDS, "V10 texture record order drift")
    require(len({record.sha256 for record in textures.values()}) == len(PAGE_IDS), "V10 texture file SHAs are not unique")

    if validate_source:
        validate_source_v9()
    return TextureSet(
        report_path=report_path,
        report_sha256=report_sha,
        report_canonical_sha256=canonical_json_sha256(raw),
        asset_id=ASSET_ID,
        textures=textures,
        raw=raw,
    )


def safe_output_path(name: str) -> Path:
    require(isinstance(name, str) and Path(name).name == name, "Output name must be a basename")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.blend", name) is not None, "Unsafe output blend name")
    output = (OUTPUT_ROOT / name).resolve()
    require(output.parent == OUTPUT_ROOT.resolve(), "Output escapes targeted_final_v10/outputs")
    require(output != SOURCE_BLEND.resolve(), "Output aliases pinned V9 source")
    return output


def preflight_summary(texture_set: TextureSet) -> dict[str, Any]:
    return {
        "asset_id": texture_set.asset_id,
        "mode": "texture_only_from_pinned_v9",
        "source_blend": str(SOURCE_BLEND),
        "source_blend_sha256": SOURCE_PINS[SOURCE_BLEND],
        "expected_meshes": EXPECTED_MESHES,
        "expected_triangles": EXPECTED_TRIANGLES,
        "texture_report": str(texture_set.report_path),
        "texture_report_sha256": texture_set.report_sha256,
        "textures": {
            page_id: {
                "path": str(record.path),
                "sha256": record.sha256,
                "size": list(record.size),
                "bytes": record.bytes,
            }
            for page_id, record in texture_set.textures.items()
        },
        "expected_image_bindings": EXPECTED_IMAGE_BINDINGS,
        "preservation": [
            "geometry",
            "topology",
            "uv",
            "normals",
            "transforms",
            "material_slots",
            "shader_structure_and_sampling",
            "camera",
            "world",
        ],
    }


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-source-pins", action="store_true", help="Tests only")
    args = parser.parse_args()
    try:
        texture_set = load_texture_set(validate_source=not args.skip_source_pins)
    except (ContractError, OSError) as exc:
        print(f"TARGETED_FINAL_V10_PREFLIGHT_REFUSED: {exc}", file=sys.stderr)
        return 2
    print("TARGETED_FINAL_V10_PREFLIGHT_PASSED=" + json.dumps(preflight_summary(texture_set), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
