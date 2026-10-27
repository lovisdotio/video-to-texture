#!/usr/bin/env python3
"""Pinned, dependency-free contract for the isolated non-BODY UV v8 pass."""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OBJECT_V2 = ROOT.parent
DATASET = OBJECT_V2 / "dataset_object_v3"
GEOMETRY = DATASET / "geometry/ferrari.npz"
PRODUCTION_REPORT = (
    OBJECT_V2
    / "body_bake_v3/diagnostics/production_v3_final/extracted/multi_atlas_bake_report.json"
)

GEOMETRY_SHA256 = "ba944d9c58e0fe4e7ac4c02da59ea356cbc70909001c244c91b6414cf3a974f4"
PRODUCTION_REPORT_SHA256 = "bea587f517ad65a22078aed054e19cac7a61cd9bb049b4278954a1051fe80e1d"

BODY_OBJECT_ID = 1
EXPECTED_OBJECTS = 51
EXPECTED_SOURCE_TRIANGLES = 358_788
EXPECTED_NONBODY_SOURCE_TRIANGLES = 278_222
EXPECTED_NONBODY_GEOMETRY_DEGENERATES = 84
# Exact weld/dedup over every source face, including collapsed/near-zero
# geometry.  Only the renderable subset enters the Blender solver below.
EXPECTED_NONBODY_GEOMETRIC_UNIQUE_TRIANGLES = 139_253
# Solver/renderable representatives after the 84 source-declared degenerates
# are quarantined before exact deduplication.
EXPECTED_NONBODY_PROXY_TRIANGLES = 139_211

# The preferred deliverable is a new v8 geometry set made from the unique
# renderable proxies, one object per source object.  Keeping the doubled source
# triangles as the displayed mesh is supported only as a diagnostic transfer
# target: it would retain coincident surfaces and therefore retain z-fighting.
GEOMETRY_DELIVERY_MODE = "unique_proxy_semantic_subpieces"
GEOMETRY_DELIVERY_RECOMMENDED = True
EXPECTED_V8_SOURCE_PARENT_OBJECTS = 50
EXPECTED_V8_NONBODY_RENDER_OBJECTS = 63
EXPECTED_FINAL_SCENE_OBJECTS = 66
EXPECTED_V8_GEOMETRY_TRIANGLES = EXPECTED_NONBODY_PROXY_TRIANGLES
EXPECTED_FINAL_SCENE_TRIANGLES = 179_488
PRESERVE_SOURCE_OBJECT_BOUNDARIES = True
PRESERVE_SOURCE_MATERIAL_ASSIGNMENT = True
V8_MATERIAL_BACKFACE_CULLING = False
WRITE_SOURCE_TRANSFER_SIDECAR = True
PIECE_ID_FIELD = "piece_id"
PIECE_ID_KIND = "source_object_id,component_id"
PIECE_ID_EQUALS_OBJECT_ID = False
UNSPLIT_COMPONENT_ID = -1
ALLOW_INTER_OBJECT_UV_STACKING = False
ALLOW_MIRRORED_INTER_OBJECT_UV_REUSE = False
SOURCE_PBR_REUSE_ALLOWED = False
SOURCE_MATERIAL_TABLE_SHA256 = "5f55238f237e09407e2acda4f269cb2f79f62b7d6e99ee3a4b18c0c54a123d64"

# Evidence from the already validated v3 transparency pass.  These five IDs
# participate in the transparent-occluder treatment.  LEDs (20) retain an
# emissive shader but are not added to this exact z-buffer exclusion set.
TRANSPARENT_ZBUFFER_OBJECT_IDS = (6, 15, 21, 22, 38)

BASE_PAGE_SIZE = 4096
HERO_PAGE_SIZE = 8192
HERO_8K_PAGE_IDS = ("running_metal", "cockpit")
PACK_MARGIN_PIXELS = 16
DEGENERATE_RESERVED_STRIP_PIXELS = 8
UNWRAP_METHOD = "ANGLE_BASED"
UVGAMI_SEAM_ANGLE_DEGREES = 66.0
UVGAMI_MIN_PROXY_FACES = 64

UVGAMI_REVISION = "8774be0"
UVGAMI_COMMIT = "8774be08537ab5e05e46381f2a1e3a3acc5c1324"
UVGAMI_REPOSITORY = "https://github.com/danielboxer/UVgami"
UVGAMI_LICENSE = "GPL-3.0-or-later"
DEFAULT_UVGAMI_ROOT = Path(os.environ.get("UVGAMI_ROOT", "UVgami"))


@dataclass(frozen=True)
class SourceAtlasSpec:
    slug: str
    object_ids: tuple[int, ...]
    atlas_npz_sha256: str

    @property
    def atlas_npz(self) -> Path:
        return DATASET / "geometry/atlases" / f"{self.slug}.npz"


SOURCE_ATLAS_SPECS = {
    "exterior_details": SourceAtlasSpec(
        "exterior_details",
        (0, 6, 14, 15, 16, 20, 21, 22, 44, 49, 50),
        "024aa9506f85ab235a4c4f3350135e96a8484be13d14211661657e789884ed8d",
    ),
    "running_gear": SourceAtlasSpec(
        "running_gear",
        (2, 3, 4, 5, 10, 11, 12, 13, 24, 25, 26, 27, 29, 30, 31, 32, 40, 41, 42, 43, 45, 46, 47, 48),
        "dcbe69798e554941cdf736bb6a016df0619307b72bafcd65d454d429ee2aa946",
    ),
    "interior": SourceAtlasSpec(
        "interior",
        (7, 8, 9, 17, 18, 19, 23, 28, 33, 34, 35, 36, 37, 38, 39),
        "453edd0015dad2b54232466f517611fbce2b2cabd643e56ebcbc95ba25ecb64a",
    ),
}
SOURCE_ATLAS_IDS = tuple(SOURCE_ATLAS_SPECS)


@dataclass(frozen=True)
class TargetPageSpec:
    slug: str
    object_ids: tuple[int, ...]
    base_size: int = BASE_PAGE_SIZE
    allowed_sizes: tuple[int, ...] = (BASE_PAGE_SIZE,)
    shader_class: str = "opaque_unlit_capture"


# V8 deliberately does not inherit the three legacy bake pages.  Glass,
# translucent lights and tyres require different future shader treatment.
# The dense running-gear and cockpit pages are fixed at 8K for the hero-quality
# delivery; this is a quality policy, not a conditional response to a failed
# semantic floor.  The other four pages remain fixed at 4K.
TARGET_PAGE_SPECS = {
    "exterior_opaque": TargetPageSpec(
        "exterior_opaque",
        (0, 14, 16, 44, 49, 50),
        shader_class="opaque_exterior_capture",
    ),
    "glass": TargetPageSpec(
        "glass",
        (15,),
        shader_class="glass_capture_proxy",
    ),
    "lights_translucent": TargetPageSpec(
        "lights_translucent",
        (6, 20, 21, 22, 38),
        shader_class="translucent_emissive_capture",
    ),
    "running_metal": TargetPageSpec(
        "running_metal",
        (2, 3, 4, 5, 10, 11, 12, 13, 24, 25, 26, 27, 29, 30, 31, 32, 45, 46, 47, 48),
        base_size=HERO_PAGE_SIZE,
        allowed_sizes=(HERO_PAGE_SIZE,),
        shader_class="opaque_metal_capture",
    ),
    "tires": TargetPageSpec(
        "tires",
        (40, 41, 42, 43),
        shader_class="opaque_rubber_capture",
    ),
    "cockpit": TargetPageSpec(
        "cockpit",
        (7, 8, 9, 17, 18, 19, 23, 28, 33, 34, 35, 36, 37, 39),
        base_size=HERO_PAGE_SIZE,
        allowed_sizes=(HERO_PAGE_SIZE,),
        shader_class="opaque_cockpit_capture",
    ),
}
TARGET_PAGE_IDS = tuple(TARGET_PAGE_SPECS)
NONBODY_OBJECT_IDS = tuple(
    object_id for spec in TARGET_PAGE_SPECS.values() for object_id in spec.object_ids
)


@dataclass(frozen=True)
class ObjectSemanticSpec:
    """Material/shader metadata for one separately addressable v8 piece.

    ``source_material`` is provenance only.  ``target_pbr`` names a new v8
    material template; the production PBR datablock must never be copied.
    Symmetric instances share priors and density, but never atlas coordinates.
    """

    source_material: str
    target_pbr: str
    appearance: str
    transparency: str = "opaque"
    symmetry_group: str | None = None
    symmetry_instance: str | None = None
    allow_uv_stacking: bool = False


# Explicit object/material contract.  The source material names intentionally
# remain visible here because several names do not match their object names
# (for example interior_dark/interior_light); they are evidence, not reusable
# shader definitions.
OBJECT_SEMANTIC_SPECS = {
    0: ObjectSemanticSpec("_0098_DodgerBlue", "v8_paint_detail", "paint_detail"),
    14: ObjectSemanticSpec("metal_chrome", "v8_mirror_chrome", "mirror_chrome_opaque"),
    16: ObjectSemanticSpec("Tires", "v8_exterior_opaque", "grille_opaque"),
    44: ObjectSemanticSpec("Leather_red", "v8_exterior_opaque", "trim_opaque"),
    49: ObjectSemanticSpec("Tires", "v8_exterior_opaque", "wiper_opaque"),
    50: ObjectSemanticSpec("Ferrari_Yellow", "v8_paint_detail", "paint_detail"),
    15: ObjectSemanticSpec(
        "Glass_Gray", "v8_glass_transmission", "window_glass", "transmission_alpha"
    ),
    6: ObjectSemanticSpec(
        "Taillight_Glass", "v8_lamp_lens", "lamp_lens", "translucent"
    ),
    20: ObjectSemanticSpec(
        "Turn_Signal_LED", "v8_lamp_emissive", "lamp_emitter", "translucent_emissive"
    ),
    21: ObjectSemanticSpec(
        "Projector_Glass", "v8_lamp_lens", "projector_lens", "translucent"
    ),
    22: ObjectSemanticSpec(
        "Taillight_Glass", "v8_lamp_lens", "lamp_lens_red", "translucent"
    ),
    38: ObjectSemanticSpec(
        "Taillight_Glass", "v8_lamp_emissive", "cockpit_lamp", "translucent_emissive"
    ),
    2: ObjectSemanticSpec("metal_gray", "v8_metal", "brake_metal", symmetry_group="brake", symmetry_instance="base"),
    3: ObjectSemanticSpec("metal_gray", "v8_metal", "brake_metal", symmetry_group="brake", symmetry_instance="001"),
    4: ObjectSemanticSpec("metal_gray", "v8_metal", "brake_metal", symmetry_group="brake", symmetry_instance="002"),
    5: ObjectSemanticSpec("metal_gray", "v8_metal", "brake_metal", symmetry_group="brake", symmetry_instance="003"),
    10: ObjectSemanticSpec("Ferrari_Yellow", "v8_paint_detail", "wheel_centre", symmetry_group="centre", symmetry_instance="base"),
    11: ObjectSemanticSpec("Ferrari_Yellow", "v8_paint_detail", "wheel_centre", symmetry_group="centre", symmetry_instance="001"),
    12: ObjectSemanticSpec("Ferrari_Yellow", "v8_paint_detail", "wheel_centre", symmetry_group="centre", symmetry_instance="002"),
    13: ObjectSemanticSpec("Ferrari_Yellow", "v8_paint_detail", "wheel_centre", symmetry_group="centre", symmetry_instance="003"),
    24: ObjectSemanticSpec("Interior_dark", "v8_metal", "wheel_nuts", symmetry_group="nuts", symmetry_instance="base"),
    25: ObjectSemanticSpec("Interior_dark", "v8_metal", "wheel_nuts", symmetry_group="nuts", symmetry_instance="001"),
    26: ObjectSemanticSpec("Interior_dark", "v8_metal", "wheel_nuts", symmetry_group="nuts", symmetry_instance="002"),
    27: ObjectSemanticSpec("Interior_dark", "v8_metal", "wheel_nuts", symmetry_group="nuts", symmetry_instance="003"),
    29: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_rim", symmetry_group="rim", symmetry_instance="fl"),
    30: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_rim", symmetry_group="rim", symmetry_instance="fr"),
    31: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_rim", symmetry_group="rim", symmetry_instance="rl"),
    32: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_rim", symmetry_group="rim", symmetry_instance="rr"),
    45: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_metal", symmetry_group="wheel", symmetry_instance="base"),
    46: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_metal", symmetry_group="wheel", symmetry_instance="001"),
    47: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_metal", symmetry_group="wheel", symmetry_instance="002"),
    48: ObjectSemanticSpec("metal_gray", "v8_metal", "wheel_metal", symmetry_group="wheel", symmetry_instance="003"),
    40: ObjectSemanticSpec("Tires", "v8_rubber", "tire_rubber", symmetry_group="tire", symmetry_instance="base"),
    41: ObjectSemanticSpec("Tires", "v8_rubber", "tire_rubber", symmetry_group="tire", symmetry_instance="001"),
    42: ObjectSemanticSpec("Tires", "v8_rubber", "tire_rubber", symmetry_group="tire", symmetry_instance="002"),
    43: ObjectSemanticSpec("Tires", "v8_rubber", "tire_rubber", symmetry_group="tire", symmetry_instance="003"),
    7: ObjectSemanticSpec("Carbon_Fiber", "v8_carbon", "cockpit_carbon"),
    8: ObjectSemanticSpec("Carbon_Fiber", "v8_carbon", "cockpit_carbon_trim"),
    9: ObjectSemanticSpec("Carpet", "v8_fabric", "cockpit_carpet"),
    17: ObjectSemanticSpec("Interior_light", "v8_plastic", "cockpit_light"),
    18: ObjectSemanticSpec("Interior_dark", "v8_plastic", "cockpit_dark"),
    19: ObjectSemanticSpec("Leather", "v8_leather", "cockpit_leather"),
    23: ObjectSemanticSpec("metal_gray", "v8_metal", "cockpit_metal"),
    28: ObjectSemanticSpec("plastic_gray", "v8_plastic", "cockpit_plastic"),
    33: ObjectSemanticSpec("Carbon_Fiber", "v8_carbon", "steering_carbon"),
    34: ObjectSemanticSpec("Ferrari_Yellow", "v8_paint_detail", "steering_badge"),
    35: ObjectSemanticSpec("Interior_dark", "v8_plastic", "steering_column"),
    36: ObjectSemanticSpec("Leather", "v8_leather", "steering_leather"),
    37: ObjectSemanticSpec("metal_gray", "v8_metal", "steering_metal"),
    39: ObjectSemanticSpec("Leather_red", "v8_leather", "steering_trim"),
}

if set(OBJECT_SEMANTIC_SPECS) != set(NONBODY_OBJECT_IDS):
    missing = sorted(set(NONBODY_OBJECT_IDS) - set(OBJECT_SEMANTIC_SPECS))
    extra = sorted(set(OBJECT_SEMANTIC_SPECS) - set(NONBODY_OBJECT_IDS))
    raise RuntimeError(f"V8 semantic metadata partition drift: missing={missing}, extra={extra}")
if any(spec.allow_uv_stacking for spec in OBJECT_SEMANTIC_SPECS.values()):
    raise RuntimeError("V8 forbids inter-object UV stacking")


def source_material_table_sha256() -> str:
    payload = "".join(
        f"{object_id}\0{OBJECT_SEMANTIC_SPECS[object_id].source_material}\n"
        for object_id in sorted(OBJECT_SEMANTIC_SPECS)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if source_material_table_sha256() != SOURCE_MATERIAL_TABLE_SHA256:
    raise RuntimeError("V8 authoritative object-to-source-material table drift")


# Importance is indexed only by authoritative object ID.  In particular IDs
# 17/18 have labels opposite to their source material names, so neither shader
# class nor weight may be inferred from the object label.
SEMANTIC_DENSITY_BOOST_BY_OBJECT = {
    0: 2.4, 14: 1.25, 16: 1.0, 44: 1.45, 49: 1.0, 50: 2.4,
    15: 1.35,
    6: 1.0, 20: 2.4, 21: 2.4, 22: 2.1, 38: 2.8,
    2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
    10: 2.2, 11: 2.2, 12: 2.2, 13: 2.2,
    24: 2.0, 25: 2.0, 26: 2.0, 27: 2.0,
    29: 1.45, 30: 1.45, 31: 1.45, 32: 1.45,
    45: 1.45, 46: 1.45, 47: 1.45, 48: 1.45,
    40: 1.35, 41: 1.35, 42: 1.35, 43: 1.35,
    7: 1.4, 8: 1.4, 9: 0.75, 17: 1.25, 18: 1.15, 19: 1.2,
    23: 1.2, 28: 1.0, 33: 1.7, 34: 2.8, 35: 1.2, 36: 1.55,
    37: 2.8, 39: 2.6,
}
if set(SEMANTIC_DENSITY_BOOST_BY_OBJECT) != set(NONBODY_OBJECT_IDS):
    raise RuntimeError("V8 object-indexed semantic density table drift")


# Exact connected-component evidence for the two shader-sensitive objects.
# Component IDs are deterministic labels emitted by proxy_core_v8.
SPECIAL_COMPONENT_GATES = {
    14: {
        "expected_components": 66,
        "required_proxy_component_faces": {26: 46, 27: 46},
        "classification": {
            26: "right_rearview_mirror_surface",
            27: "left_rearview_mirror_surface",
        },
        "target_pbr": "v8_mirror_chrome",
    },
    15: {
        "expected_components": 12,
        "required_proxy_component_faces": {2: 266, 11: 124},
        "classification": {
            2: "windshield",
            11: "rear_window",
        },
        "remaining_components": "lamp_glass_and_minor_glass",
        "target_pbr": "v8_glass_transmission",
    },
}

# The two BODY mirror shells (2 x 3,525 proxy faces in the BODY pipeline) are
# deliberately outside non-BODY v8 and stay owned by object 1.
BODY_MIRROR_SHELLS_OUTSIDE_V8 = True

# Final scene separability.  V8 parameterizes 50 non-BODY source parents, but
# the integration materializes 63 non-BODY render objects: chrome remainder +
# two mirror surfaces, all 12 glass components, and 48 unchanged parents.  BODY
# adds remainder + two mirror shells without changing its v6 UV/image.
FINAL_COMPONENT_SPLITS = {
    1: {
        "component_ids": (-1, 17, 18),
        "roles": {
            -1: "body_remainder",
            17: "left_mirror_shell",
            18: "right_mirror_shell",
        },
        "uv_texture_policy": "preserve_BODY_v6_exact",
    },
    14: {
        "component_ids": (-1, 26, 27),
        "roles": {
            -1: "chrome_remainder",
            26: "right_rearview_mirror_surface",
            27: "left_rearview_mirror_surface",
        },
        "uv_texture_policy": "v8_exterior_opaque",
    },
    15: {
        "component_ids": tuple(range(12)),
        "roles": {
            0: "minor_glass_00",
            1: "minor_glass_01",
            2: "windshield",
            3: "front_optical_cover_left",
            4: "front_optical_cover_right",
            5: "minor_glass_05",
            6: "lamp_glass_06",
            7: "minor_glass_07",
            8: "minor_glass_08",
            9: "lamp_glass_09",
            10: "minor_glass_10",
            11: "rear_window",
        },
        "uv_texture_policy": "v8_glass",
    },
}


def piece_component_id(source_object_id: int, topology_component_id: int) -> int:
    """Return the stable final subpiece component, or -1 for an unsplit parent."""

    if source_object_id == 15:
        return topology_component_id
    if source_object_id == 14 and topology_component_id in (26, 27):
        return topology_component_id
    return UNSPLIT_COMPONENT_ID


def piece_component_role(source_object_id: int, topology_component_id: int) -> str:
    component_id = piece_component_id(source_object_id, topology_component_id)
    split = FINAL_COMPONENT_SPLITS.get(source_object_id)
    if split is not None and component_id in split["roles"]:
        return split["roles"][component_id]
    return "source_object_remainder"


def render_piece_ids_for_source_object(source_object_id: int) -> tuple[tuple[int, int], ...]:
    split = FINAL_COMPONENT_SPLITS.get(source_object_id)
    component_ids = split["component_ids"] if split is not None else (UNSPLIT_COMPONENT_ID,)
    return tuple((source_object_id, int(component_id)) for component_id in component_ids)


# Post-split semantic floors.  Object-level floors still apply to ordinary
# pieces; these signature glass/mirror subpieces must survive page scaling on
# their own rather than borrowing texels from a large source parent.
PIECE_MIN_EFFECTIVE_TEXELS_OVERRIDES = {
    (14, 26): 32_768,
    (14, 27): 32_768,
    (15, 2): 131_072,
    (15, 11): 65_536,
    (15, 3): 32_768,
    (15, 4): 32_768,
    (15, 6): 32_768,
    (15, 9): 32_768,
}


# These are new semantic materials to be constructed by the future Blender
# integrator.  Values are intent/validation classes rather than a copy of any
# source PBR.  Texture colour may be projected; source shader parameters may
# not be imported.
PBR_TEMPLATE_SPECS = {
    "v8_glass_transmission": {
        "surface": "principled_transmission",
        "transparency": "transmission_alpha",
        "metallic_class": "dielectric",
    },
    "v8_mirror_chrome": {
        "surface": "principled_metal",
        "transparency": "opaque",
        "metallic_class": "mirror_chrome",
    },
    "v8_lamp_lens": {
        "surface": "principled_transmission_lens",
        "transparency": "translucent",
        "metallic_class": "dielectric",
    },
    "v8_lamp_emissive": {
        "surface": "principled_emission_lens",
        "transparency": "translucent_emissive",
        "metallic_class": "dielectric",
    },
    "v8_metal": {"surface": "principled_metal", "transparency": "opaque", "metallic_class": "metal"},
    "v8_rubber": {"surface": "principled_rubber", "transparency": "opaque", "metallic_class": "dielectric"},
    "v8_carbon": {"surface": "principled_carbon", "transparency": "opaque", "metallic_class": "dielectric"},
    "v8_fabric": {"surface": "principled_fabric", "transparency": "opaque", "metallic_class": "dielectric"},
    "v8_leather": {"surface": "principled_leather", "transparency": "opaque", "metallic_class": "dielectric"},
    "v8_plastic": {"surface": "principled_plastic", "transparency": "opaque", "metallic_class": "dielectric"},
    "v8_paint_detail": {"surface": "principled_paint", "transparency": "opaque", "metallic_class": "dielectric"},
    "v8_exterior_opaque": {"surface": "principled_opaque", "transparency": "opaque", "metallic_class": "dielectric"},
}

if {spec.target_pbr for spec in OBJECT_SEMANTIC_SPECS.values()} - set(PBR_TEMPLATE_SPECS):
    raise RuntimeError("V8 object semantics reference an unknown target PBR template")


def source_atlas_for_object(object_id: int) -> str:
    for slug, spec in SOURCE_ATLAS_SPECS.items():
        if object_id in spec.object_ids:
            return slug
    raise ValueError(f"Object {object_id} is not in a source non-BODY atlas")


def target_page_for_object(object_id: int) -> str:
    for slug, spec in TARGET_PAGE_SPECS.items():
        if object_id in spec.object_ids:
            return slug
    raise ValueError(f"Object {object_id} is not in a v8 target page")


def safe_object_stem(object_id: int, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "object"
    return f"{object_id:02d}_{safe}"
