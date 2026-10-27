#!/usr/bin/env python3
"""Immutable contract for additive, targeted Ferrari UV repair v9.

V9 is deliberately an overlay pipeline.  It reads the passed v8 hero
sidecars (and the immutable BODY render proxy), writes only below this
directory, and never rewrites a v8/v6/canonical asset.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V8_ROOT = ROOT.parent
OBJECT_V2 = V8_ROOT.parent

V8_HERO_ROOT = V8_ROOT / "artifacts/unwrap_pack_hero_8k"
V8_HERO_REPORT = V8_HERO_ROOT / "unwrap_pack_report.json"
V8_PROXY_MANIFEST = V8_ROOT / "artifacts/proxy_manifest.json"
V8_INVENTORY = V8_ROOT / "inventory/nonbody_inventory_v8.json"
BODY_PROXY = OBJECT_V2 / "smart_uv_v5_geometry/artifacts/body_render_proxy_v5.npz"

V8_HERO_REPORT_SHA256 = "74099dd40f0ed3f2f3be0d3c6cc341a21ef84afadb54213fe0da96974e2c2a7c"
V8_PROXY_MANIFEST_SHA256 = "5a8f26955fd189bd5b84011eeda81a20401c452d8f335b5b9de3fadcdc6c8c1a"
V8_INVENTORY_SHA256 = "1f2a50476d640cf0e7a468088b5adbe7a97f6235330c31eb92b7171077deddda"
BODY_PROXY_SHA256 = "ac70a58c8cfdf6964f6b8a3ffd876ef61494fe7894b314ca25010f8f83da15a9"

SCHEMA = "car-city.ferrari-targeted-reunwrap/v9"
PATCH_SCHEMA = "car-city.ferrari-targeted-uv-overlay/v9"
PACK_MARGIN_PIXELS = 16
MICROCHART_MAX_FACES = 4
UV_EPSILON = 2.0e-7


@dataclass(frozen=True)
class QualityGate:
    charts_max: int
    avoidable_microcharts_max: int
    seam_edge_percent_max: float
    condition_p95_max: float
    density_p95_p05_max: float
    charts_min: int = 1


@dataclass(frozen=True)
class TargetSpec:
    selector: str
    slug: str
    source_kind: str
    object_id: int
    component_id: int
    page_id: str
    resolution: int
    gate: QualityGate
    peer_selector: str | None = None
    peer_density_tolerance_fraction: float | None = None
    peer_chart_delta_max: int | None = None
    family_object_ids: tuple[int, ...] = ()


TARGETS = (
    TargetSpec("1:17", "o01_c17_left_mirror_shell", "body_proxy", 1, 17, "body_hero_8k", 8192,
               QualityGate(10, 4, 3.0, 1.60, 1.25), "1:18", 0.02),
    TargetSpec("6:-1", "o06_all_brakes", "v8_sidecar", 6, -1, "lights_translucent", 4096,
               QualityGate(192, 64, 8.0, 1.35, 1.25)),
    TargetSpec("7:-1", "o07_all_carbon_fibre", "v8_sidecar", 7, -1, "cockpit", 8192,
               QualityGate(8, 0, 10.0, 1.50, 1.35)),
    TargetSpec("15:3", "o15_c03_front_optical_cover_left", "v8_sidecar", 15, 3, "glass", 4096,
               QualityGate(3, 0, 1.0, 1.20, 1.25), "15:4", None, 1),
    TargetSpec("17:-1", "o17_all_interior_dark", "v8_sidecar", 17, -1, "cockpit", 8192,
               QualityGate(200, 50, 6.0, 1.30, 1.25)),
    TargetSpec("18:-1", "o18_all_interior_light", "v8_sidecar", 18, -1, "cockpit", 8192,
               QualityGate(500, 100, 5.0, 1.50, 1.30)),
    TargetSpec("19:-1", "o19_all_leather", "v8_sidecar", 19, -1, "cockpit", 8192,
               QualityGate(64, 4, 4.0, 1.50, 1.35)),
    TargetSpec("23:-1", "o23_all_metal", "v8_sidecar", 23, -1, "cockpit", 8192,
               QualityGate(70, 5, 5.0, 1.50, 1.35)),
    TargetSpec("26:-1", "o26_all_nuts_002", "v8_sidecar", 26, -1, "running_metal", 8192,
               QualityGate(12, 0, 5.0, 1.10, 1.05, charts_min=9),
               family_object_ids=(24, 25, 26, 27), peer_density_tolerance_fraction=0.02),
    TargetSpec("27:-1", "o27_all_nuts_003", "v8_sidecar", 27, -1, "running_metal", 8192,
               QualityGate(12, 0, 5.0, 1.10, 1.05, charts_min=9),
               family_object_ids=(24, 25, 26, 27), peer_density_tolerance_fraction=0.02),
    TargetSpec("28:-1", "o28_all_plastic_gray", "v8_sidecar", 28, -1, "cockpit", 8192,
               QualityGate(110, 45, 8.0, 1.35, 1.20)),
    TargetSpec("36:-1", "o36_all_steering_leather", "v8_sidecar", 36, -1, "cockpit", 8192,
               QualityGate(8, 0, 3.0, 1.35, 1.30)),
    TargetSpec("49:-1", "o49_all_wipers", "v8_sidecar", 49, -1, "exterior_opaque", 4096,
               QualityGate(50, 20, 15.0, 1.30, 1.25)),
)

TARGET_BY_SELECTOR = {target.selector: target for target in TARGETS}
TARGET_BY_SLUG = {target.slug: target for target in TARGETS}
if len(TARGET_BY_SELECTOR) != len(TARGETS) or len(TARGET_BY_SLUG) != len(TARGETS):
    raise RuntimeError("Duplicate targeted-reunwrap selector/slug")

EXPECTED_SELECTORS = (
    "1:17", "6:-1", "7:-1", "15:3", "17:-1", "18:-1", "19:-1",
    "23:-1", "26:-1", "27:-1", "28:-1", "36:-1", "49:-1",
)
if tuple(target.selector for target in TARGETS) != EXPECTED_SELECTORS:
    raise RuntimeError("V9 shortlist drift")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_authorities() -> None:
    expected = (
        (V8_HERO_REPORT, V8_HERO_REPORT_SHA256),
        (V8_PROXY_MANIFEST, V8_PROXY_MANIFEST_SHA256),
        (V8_INVENTORY, V8_INVENTORY_SHA256),
        (BODY_PROXY, BODY_PROXY_SHA256),
    )
    for path, digest in expected:
        require(path.is_file(), f"Missing immutable authority: {path}")
        require(sha256_file(path) == digest, f"Immutable authority hash drift: {path}")


def safe_output(path: Path) -> Path:
    value = path.expanduser().resolve()
    require(value == ROOT or ROOT in value.parents, f"Output escaped targeted_reunwrap_v9: {value}")
    return value


def parse_selectors(value: str | None) -> tuple[TargetSpec, ...]:
    if value is None or not value.strip():
        return TARGETS
    selectors = tuple(part.strip() for part in value.split(",") if part.strip())
    require(selectors, "No target selector supplied")
    unknown = sorted(set(selectors) - set(TARGET_BY_SELECTOR))
    require(not unknown, f"Unknown or non-shortlisted target selectors: {unknown}")
    require(len(set(selectors)) == len(selectors), "Duplicate target selector")
    return tuple(TARGET_BY_SELECTOR[selector] for selector in selectors)
