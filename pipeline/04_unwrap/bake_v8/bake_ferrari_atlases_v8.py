#!/usr/bin/env python3
"""Six-page, piece-owned Ferrari inverse-projection baker for a fal H100.

Importing this module defines two fal entrypoints but never submits work.  BODY
v6 is immutable and is not an output texture.  The complete 179,488-triangle
representative car is used for full/opaque visibility while the six canonical
non-BODY pages are baked sequentially at 4K/8K (or 512/1024 smoke size).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import fal
from fal.toolkit import File


SCRIPT_DIR = Path(__file__).resolve().parent
OBJECT_V2 = SCRIPT_DIR.parent.parent
if str(OBJECT_V2) not in sys.path:
    sys.path.insert(0, str(OBJECT_V2))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bake_object_textures as legacy
import contract as bake_contract
import runtime_contract as runtime_contract_module
from body_bake_v3 import contract as v3_contract_module
from body_bake_v3.contract import (
    EXPECTED_ELIGIBLE_ALIGNMENT_FRAMES,
    EXPECTED_OBJECTS,
    deterministic_train_holdout,
    load_alignment_contract,
)
from runtime_contract import PageRuntimeContract, RuntimeV8Contract, load_runtime_contract


IMAGE = legacy.IMAGE
RESULT_SCHEMA = "car-city.object-video-fit-six-page-bake/v8"
EXPECTED_TEXTURES = tuple(spec.page_id for spec in bake_contract.PAGE_SPECS)
SMOKE_ALLOWED_ATLAS_SIZES = (512, 1024)
SMOKE_PADDING = 4
SMOKE_RENDER_SIZE = (1280, 720)
MIN_REQUIRED_TRAIN_ALIGNMENT_FRAMES = 3
MIN_REQUIRED_TRAIN_ALIGNMENT_PIXELS = 16
MIN_TRAIN_ALIGNMENT_CONFIDENCE = 0.5
MIN_SMOKE_DIRECT_TEXELS_PER_RECOVERABLE_PIECE = 4
MIN_PRODUCTION_DIRECT_TEXELS_PER_RECOVERABLE_PIECE = 32
MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE = 4
MIN_PRODUCTION_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE = 16
MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE = 4
MIN_DIRECT_COVERAGE_PER_RECOVERABLE_PIECE = 0.001
MIN_OBSERVED_CHART_DOMAIN_RATIO = 0.01
MAX_PAGE_HOLDOUT_MEAN_L1 = 0.35
EXPECTED_NO_ALIGNMENT_OBJECT_IDS = (21,)
EXPECTED_INSUFFICIENT_MULTIVIEW_OBJECT_IDS = (0, 21, 37)
EXPECTED_INSUFFICIENT_MULTIVIEW_PIECE_IDS = ("0:-1", "21:-1", "37:-1")
EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS = ("15:6", "15:7", "15:9", "15:10")
COUNTERFACTUAL_SELF_DEPTH_PIECE_INDICES = frozenset(
    index
    for index, piece in enumerate(bake_contract.EXPECTED_PIECE_IDS)
    if bake_contract.piece_key(piece) in EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS
)
COUNTERFACTUAL_SELF_DEPTH_TOLERANCE_METERS = 0.0001
COUNTERFACTUAL_SELF_DEPTH_CROP_SIZE = 128
COUNTERFACTUAL_SELF_DEPTH_CROP_MARGIN_PIXELS = 8
COUNTERFACTUAL_SELF_DEPTH_MIN_VALID_WEIGHT = 0.75
COUNTERFACTUAL_SELF_DEPTH_MIN_NORMAL_COSINE = 0.5
DENSE_OBJECT_SURROGATE_POLICY = {
    # ``steering_trim`` is one to two screen pixels in the canonical render.
    # The dense semantic raster consistently quantises those exact owner pixels
    # to adjacent ``steering_leather``. Their declared source materials differ
    # (36=Leather, 39=Leather_red), but object36 local flow is not accepted on
    # these views, so label36 contributes only the common global warp. Geometry,
    # piece ownership and depth remain exact 39; this is a coarse-label
    # surrogate, never same-source-material or exact-label evidence.
    "39:-1": (36,),
}
DENSE_OBJECT_SURROGATE_MIN_CONFIDENCE = 0.5
DENSE_OBJECT_SURROGATE_PIECE_INDEX = {
    bake_contract.piece_key(piece): index
    for index, piece in enumerate(bake_contract.EXPECTED_PIECE_IDS)
}
SPECIAL_EVIDENCE_POLICY = {
    0: "neutral_allowed_only_when_canonical_multiview_remains_below_3_frames_or_16_pixels",
    21: "neutral_allowed_only_when_canonical_alignment_has_zero_object_pixels",
    37: "direct_video_preferred; neutral_allowed_only_if_canonical_multiview_remains_below_16_pixels",
}
RUNNER_EXECUTION_POLICY = {
    "fal_remote_allowed": True,
    "smoke_entrypoint_requires_explicit_invocation": True,
    "production_requires_matching_hashed_smoke_proof": True,
    "module_import_submits_job": False,
    "source_pbr_used": False,
    "geometry_camera_depth_owner_alignment_authority": "canonical_dataset_v3_720p",
    "radiometry_override": "pinned_existing_seedvr_rgb_only",
    "radiometry_proof_must_match_active_override": True,
}


def _default_value(environment_name: str, filenames: tuple[str, ...]) -> str:
    value = os.environ.get(environment_name, "")
    if value:
        return value
    for filename in filenames:
        for directory in (SCRIPT_DIR, OBJECT_V2):
            try:
                candidate = (directory / filename).read_text().strip()
            except OSError:
                continue
            if candidate:
                return candidate
    return ""


DEFAULT_DATASET_URL = _default_value(
    "OBJECT_DATASET_V3_URL", ("dataset_url_object_v3.txt",)
)
DEFAULT_ALIGNMENT_URL = _default_value(
    "OBJECT_ALIGNMENT_URL", ("alignment_url_object_v2.txt", "alignment_url.txt")
)
DEFAULT_UNWRAP_URL = _default_value(
    "OBJECT_UNWRAP_V8_URL", ("unwrap_url_v8.txt", "unwrap_hero_8k_url.txt")
)
DEFAULT_SMOKE_PROOF_URL = _default_value(
    "OBJECT_BAKE_V8_SMOKE_PROOF_URL", ("smoke_proof_url_v8.txt",)
)
DEFAULT_RADIOMETRY_VIDEO_URL = _default_value(
    "OBJECT_BAKE_V8_RADIOMETRY_URL", ("radiometry_url_v8.txt",)
) or bake_contract.SEEDVR_RGB_VIDEO_URL


def _log(message: str) -> None:
    print(f"[six-page-bake-v8] {message}", flush=True)


def _compute_implementation_identity() -> dict:
    component_paths = {
        "runner": Path(__file__).resolve(),
        "runtime_contract": Path(runtime_contract_module.__file__).resolve(),
        "bake_contract": Path(bake_contract.__file__).resolve(),
        "dataset_alignment_contract_v3": Path(v3_contract_module.__file__).resolve(),
        "legacy_projection_and_image": Path(legacy.__file__).resolve(),
    }
    components = {
        name: legacy._sha256(path) for name, path in sorted(component_paths.items())
    }
    canonical = json.dumps(
        components, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "components": components,
    }


_IMPLEMENTATION_IDENTITY = _compute_implementation_identity()


def _implementation_identity() -> dict:
    return {
        "sha256": _IMPLEMENTATION_IDENTITY["sha256"],
        "components": dict(_IMPLEMENTATION_IDENTITY["components"]),
    }


def _native_to_rgb_pixel_center(value, native_extent: int, rgb_extent: int):
    """Map native pixel-center coordinates without changing geometry space."""

    return (value + 0.5) * (float(rgb_extent) / float(native_extent)) - 0.5


def _validate_seedvr_probe_payload(probe: dict, selected_frame_ids) -> None:
    """Fail closed on the one existing SeedVR RGB carrier and its safe subset."""

    expected = {
        "video_streams": 1,
        "audio_streams": 0,
        "width": bake_contract.SEEDVR_RGB_WIDTH,
        "height": bake_contract.SEEDVR_RGB_HEIGHT,
        "r_frame_rate": f"{bake_contract.SEEDVR_RGB_FPS}/1",
        "avg_frame_rate": f"{bake_contract.SEEDVR_RGB_FPS}/1",
        "nb_frames": bake_contract.SEEDVR_RGB_FRAME_COUNT,
        "nb_read_frames": bake_contract.SEEDVR_RGB_FRAME_COUNT,
    }
    for field, value in expected.items():
        if probe.get(field) != value:
            raise ValueError(
                f"SeedVR RGB-only probe {field} mismatch: "
                f"{probe.get(field)!r} != {value!r}"
            )
    duration = float(probe.get("duration", -1.0))
    expected_duration = (
        bake_contract.SEEDVR_RGB_FRAME_COUNT / bake_contract.SEEDVR_RGB_FPS
    )
    if abs(duration - expected_duration) > 1e-5:
        raise ValueError("SeedVR RGB-only duration is not exactly 358/24 seconds")
    frame_ids = [int(value) for value in selected_frame_ids]
    if (
        len(frame_ids) != EXPECTED_ELIGIBLE_ALIGNMENT_FRAMES
        or frame_ids != sorted(set(frame_ids))
        or not frame_ids
        or frame_ids[0] != 0
        or frame_ids[-1] != bake_contract.SEEDVR_RGB_MAX_CANONICAL_V8_FRAME_ID
        or frame_ids[-1] > bake_contract.SEEDVR_RGB_LAST_FRAME_ID
        or bake_contract.SEEDVR_RGB_LAST_FRAME_ID in frame_ids
    ):
        raise ValueError(
            "SeedVR RGB-only selection must be the exact 104 canonical IDs 0..330"
        )


def _probe_seedvr_rgb_video(path: Path) -> dict:
    import subprocess

    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,width,height,r_frame_rate,"
            "avg_frame_rate,nb_frames,nb_read_frames,duration,pix_fmt,"
            "color_range,color_space,color_transfer,color_primaries"
        ),
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
    )
    streams = payload.get("streams", [])
    video = [row for row in streams if row.get("codec_type") == "video"]
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    if len(video) != 1:
        raise ValueError("SeedVR RGB-only carrier must contain exactly one video stream")
    row = video[0]
    ffmpeg_version = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return {
        "video_streams": len(video),
        "audio_streams": len(audio),
        "codec_name": row.get("codec_name"),
        "pix_fmt": row.get("pix_fmt"),
        "width": int(row.get("width", -1)),
        "height": int(row.get("height", -1)),
        "r_frame_rate": row.get("r_frame_rate"),
        "avg_frame_rate": row.get("avg_frame_rate"),
        "duration": row.get("duration"),
        "nb_frames": int(row.get("nb_frames", -1)),
        "nb_read_frames": int(row.get("nb_read_frames", -1)),
        "color_range": row.get("color_range"),
        "color_space": row.get("color_space"),
        "color_transfer": row.get("color_transfer"),
        "color_primaries": row.get("color_primaries"),
        "ffmpeg_version": ffmpeg_version,
    }


def _prepare_seedvr_radiometry(
    radiometry_video_url: str,
    workspace: Path,
    selected_frame_ids,
    native_source,
    PIL_Image,
) -> tuple[dict[int, Path], dict]:
    if not radiometry_video_url:
        raise ValueError(
            "This v8 freeze requires the pinned existing SeedVR RGB-only override"
        )
    if (
        int(native_source.width) != 1280
        or int(native_source.height) != 720
        or native_source.source_video_sha256
        != bake_contract.DATASET_V3_VIDEO_SHA256
    ):
        raise ValueError("SeedVR RGB override cannot replace native 720p authority")
    rgb_video = workspace / "radiometry_seedvr_2160p.mp4"
    legacy._download(radiometry_video_url, rgb_video)
    rgb_sha256 = legacy._sha256(rgb_video)
    if rgb_sha256 != bake_contract.SEEDVR_RGB_VIDEO_SHA256:
        raise ValueError("SeedVR RGB-only video hash is not the pinned existing asset")
    probe = _probe_seedvr_rgb_video(rgb_video)
    _validate_seedvr_probe_payload(probe, selected_frame_ids)
    targets = legacy._extract_video_frames(
        rgb_video, list(selected_frame_ids), workspace / "radiometry_rgb_frames"
    )
    decoded_frame_png_sha256 = {}
    for frame_id in selected_frame_ids:
        path = targets[int(frame_id)]
        with PIL_Image.open(path) as image:
            if image.mode != "RGB" or image.size != (
                bake_contract.SEEDVR_RGB_WIDTH,
                bake_contract.SEEDVR_RGB_HEIGHT,
            ):
                raise ValueError(
                    f"Decoded SeedVR frame {frame_id} is not native 3840x2160 RGB"
                )
        decoded_frame_png_sha256[str(int(frame_id))] = legacy._sha256(path)
    identity = {
        "schema": "car-city.ferrari-radiometry-video/v1",
        "mode": "seedvr_existing_rgb_only",
        "sha256": rgb_sha256,
        "probe": probe,
        "frame_mapping": bake_contract.SEEDVR_RGB_FRAME_MAPPING,
        "selected_frame_ids": [int(value) for value in selected_frame_ids],
        "selected_max_frame_id": int(max(selected_frame_ids)),
        "last_available_frame_id": bake_contract.SEEDVR_RGB_LAST_FRAME_ID,
        "frame_357_selected": False,
        "native_geometry_size": [
            int(native_source.width),
            int(native_source.height),
        ],
        "rgb_size": [
            bake_contract.SEEDVR_RGB_WIDTH,
            bake_contract.SEEDVR_RGB_HEIGHT,
        ],
        "native_to_rgb_pixel_center": [
            list(row)
            for row in bake_contract.SEEDVR_RGB_NATIVE_TO_RGB_PIXEL_CENTER
        ],
        "sampling": "grid_sample_align_corners_false",
        "color_policy": bake_contract.SEEDVR_RGB_COLOR_POLICY,
        "decoded_frame_png_sha256": decoded_frame_png_sha256,
        "geometry_camera_depth_owner_alignment_authority": {
            "dataset_manifest_sha256": native_source.manifest_sha256,
            "geometry_sha256": native_source.geometry_sha256,
            "native_video_sha256": native_source.source_video_sha256,
            "resolution": [int(native_source.width), int(native_source.height)],
        },
        "seedvr_forbidden_authority": [
            "intrinsics",
            "camera_pose",
            "depth",
            "piece_owner",
            "object_owner",
            "dense_alignment",
        ],
    }
    return targets, identity


def _alignment_supported_objects(alignment) -> set[int]:
    import numpy as np

    supported: set[int] = set()
    for payload in alignment.frames.values():
        valid = payload["dense_valid"] > 0
        for value in np.unique(payload["dense_object_id"][valid]):
            object_id = int(value)
            if 0 <= object_id < EXPECTED_OBJECTS:
                supported.add(object_id)
    return supported


def _training_alignment_evidence(alignment, frame_ids) -> dict[int, dict[str, int]]:
    import numpy as np

    evidence = {
        object_id: {"pixels": 0, "frames": 0}
        for object_id in range(EXPECTED_OBJECTS)
    }
    for frame_id in map(int, frame_ids):
        payload = alignment.frames[frame_id]
        confidence = np.asarray(payload["dense_confidence"], dtype=np.float32)
        valid = (
            (payload["dense_valid"] > 0)
            & np.isfinite(confidence)
            & (confidence >= MIN_TRAIN_ALIGNMENT_CONFIDENCE)
        )
        object_ids, counts = np.unique(
            payload["dense_object_id"][valid], return_counts=True
        )
        for value, count in zip(object_ids.tolist(), counts.tolist()):
            object_id = int(value)
            if 0 <= object_id < EXPECTED_OBJECTS:
                evidence[object_id]["pixels"] += int(count)
                evidence[object_id]["frames"] += 1
    return evidence


def _split_smoke_views(selected_ids: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Use the complete coverage-aware canonical v8 83/21 split.

    Eight time-stratified views exercised the renderer but could not exercise
    all 63 piece owners: small left/right parts were absent from either the six
    training or the two holdout views.  A smoke proof that authorizes the full
    production bake must therefore reduce spatial resolution only, never view
    coverage.  The pinned one-for-one swap relative to the legacy positional
    split places exact-label evidence for minor glass piece 15:0 in holdout
    without changing cardinality or adding a semantic-label surrogate.
    """

    ordered = tuple(map(int, selected_ids))
    if (
        len(ordered) != EXPECTED_ELIGIBLE_ALIGNMENT_FRAMES
        or len(set(ordered)) != EXPECTED_ELIGIBLE_ALIGNMENT_FRAMES
        or tuple(sorted(ordered)) != ordered
    ):
        raise ValueError("V8 smoke requires all 104 canonical aligned views")
    base_train, base_holdout = deterministic_train_holdout(ordered)
    train_to_holdout = int(
        bake_contract.V8_COVERAGE_AWARE_TRAIN_TO_HOLDOUT_FRAME_ID
    )
    holdout_to_train = int(
        bake_contract.V8_COVERAGE_AWARE_HOLDOUT_TO_TRAIN_FRAME_ID
    )
    if train_to_holdout not in base_train or holdout_to_train not in base_holdout:
        raise ValueError(
            "Canonical v8 coverage-aware split calibration frames are absent "
            "from their required legacy partitions"
        )
    train = tuple(
        sorted((set(base_train) - {train_to_holdout}) | {holdout_to_train})
    )
    holdout = tuple(
        sorted((set(base_holdout) - {holdout_to_train}) | {train_to_holdout})
    )
    if (
        len(train) != 83
        or len(holdout) != 21
        or set(train) & set(holdout)
        or set(train) | set(holdout) != set(ordered)
    ):
        raise ValueError("Canonical v8 coverage-aware 83/21 split invariant failed")
    return train, holdout


def _explicit_low_evidence_exception(row: dict) -> bool:
    piece_id = str(row.get("piece_id", ""))
    object_id = int(row.get("object_id", -1))
    return bool(
        piece_id in EXPECTED_INSUFFICIENT_MULTIVIEW_PIECE_IDS
        and object_id in EXPECTED_INSUFFICIENT_MULTIVIEW_OBJECT_IDS
        and row.get("training_alignment_sufficient_for_required_texture") is False
        and row.get("special_evidence_policy")
        == SPECIAL_EVIDENCE_POLICY[object_id]
    )


def _smoke_geometry_not_visible_exception(row: dict, split: str) -> bool:
    """Return the exact-piece N/A policy for a geometrically absent split."""

    return bool(
        _explicit_low_evidence_exception(row)
        and int(row.get(f"selected_{split}_screen_owner_pixels", -1)) == 0
        and int(
            row.get(f"selected_{split}_visibility_opportunity_texels", -1)
        )
        == 0
    )


def _mapped_object_surrogates_for_piece(piece_id: str) -> tuple[int, ...]:
    return tuple(DENSE_OBJECT_SURROGATE_POLICY.get(str(piece_id), ()))


def _counterfactual_self_depth_policy_for_piece(piece_id) -> dict | None:
    piece_key = (
        bake_contract.piece_key(piece_id)
        if isinstance(piece_id, tuple)
        else str(piece_id)
    )
    if piece_key not in EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS:
        return None
    return {
        "target_piece_only_in_transmissive_layer": True,
        "sibling_transmissive_pieces_excluded": True,
        "opaque_occluders_retained": True,
        "exact_target_piece_owner_depth_and_interior_required": True,
        "self_depth_tolerance_meters": COUNTERFACTUAL_SELF_DEPTH_TOLERANCE_METERS,
        "target_only_crop_size": COUNTERFACTUAL_SELF_DEPTH_CROP_SIZE,
        "target_only_crop_margin_pixels": COUNTERFACTUAL_SELF_DEPTH_CROP_MARGIN_PIXELS,
        "minimum_sampled_valid_weight": COUNTERFACTUAL_SELF_DEPTH_MIN_VALID_WEIGHT,
        "minimum_front_face_normal_cosine": COUNTERFACTUAL_SELF_DEPTH_MIN_NORMAL_COSINE,
        "depth_evaluation": "sampled_front_triangle_plane_at_exact_target_camera_ray",
        "raster_depth_sampling": "premultiplied_depth_divided_by_valid_coverage",
        "opaque_occlusion_tolerance": "max(0.003m,abs(camera_z)*0.0015)",
        "source_video_radiometry_remains_frontmost_composite": True,
        "transmitted_composite_sample": True,
        "implementation": "target_piece_crop128_front_face_plane_exact_ray",
    }


def _alignment_surrogate_label_has_no_local_flow(np, payload: dict, object_id: int) -> bool:
    """Fail closed unless the pinned alignment says the surrogate used global flow."""

    if "object_ids" not in payload or "object_grid_accepted" not in payload:
        return False
    object_ids = np.asarray(payload["object_ids"], dtype=np.int32).reshape(-1)
    accepted = np.asarray(payload["object_grid_accepted"], dtype=np.uint8).reshape(-1)
    if len(object_ids) != len(accepted):
        return False
    matches = np.nonzero(object_ids == int(object_id))[0]
    return bool(len(matches) == 1 and int(accepted[int(matches[0])]) == 0)


def _smoke_geometry_exercised(row: dict, split: str) -> bool:
    return bool(
        int(row.get(f"selected_{split}_screen_owner_pixels", -1)) > 0
        and int(
            row.get(f"selected_{split}_visibility_opportunity_texels", -1)
        )
        >= MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE
    )


def _uv_surface_map_v8(torch, dr, ctx, page: PageRuntimeContract, atlas_size: int, device):
    uv = torch.from_numpy(page.uv).to(device=device, dtype=torch.float32).contiguous()
    uv_faces = torch.from_numpy(page.uv_faces).to(device=device, dtype=torch.int32).contiguous()
    world_vertex = torch.from_numpy(page.world_by_uv_vertex).to(
        device=device, dtype=torch.float32
    ).contiguous()
    local_object = torch.from_numpy(page.face_object).to(device=device, dtype=torch.int32)
    local_piece = torch.from_numpy(page.face_piece).to(device=device, dtype=torch.int32)
    local_chart = torch.from_numpy(page.face_chart).to(device=device, dtype=torch.int32)
    local_visibility_face = torch.from_numpy(page.visibility_face_indices).to(
        device=device, dtype=torch.int32
    )
    clip = torch.cat(
        [
            uv * 2.0 - 1.0,
            torch.zeros_like(uv[:, :1]),
            torch.ones_like(uv[:, :1]),
        ],
        dim=-1,
    )
    rast, _ = dr.rasterize(
        ctx, clip.unsqueeze(0).contiguous(), uv_faces, [atlas_size, atlas_size]
    )
    face = rast[..., 3].long() - 1
    occupied = face >= 0
    safe = face.clamp(min=0)
    object_map = local_object.index_select(0, safe.reshape(-1)).reshape_as(face)
    piece_map = local_piece.index_select(0, safe.reshape(-1)).reshape_as(face)
    chart_map = local_chart.index_select(0, safe.reshape(-1)).reshape_as(face)
    visibility_face_map = local_visibility_face.index_select(
        0, safe.reshape(-1)
    ).reshape_as(face)
    object_map = torch.where(occupied, object_map, torch.full_like(object_map, -1))
    piece_map = torch.where(occupied, piece_map, torch.full_like(piece_map, -1))
    chart_map = torch.where(occupied, chart_map, torch.full_like(chart_map, -1))
    visibility_face_map = torch.where(
        occupied,
        visibility_face_map,
        torch.full_like(visibility_face_map, -1),
    )
    world, _ = dr.interpolate(world_vertex.unsqueeze(0), rast, uv_faces)
    result = {
        "occupied": occupied.squeeze(0),
        "face": face.squeeze(0).to(torch.int32),
        "object": object_map.squeeze(0).to(torch.int16),
        "piece": piece_map.squeeze(0).to(torch.int16),
        "chart": chart_map.squeeze(0).to(torch.int32),
        "visibility_face": visibility_face_map.squeeze(0).to(torch.int32),
        "world": world.squeeze(0).contiguous(),
    }
    del (
        uv,
        uv_faces,
        world_vertex,
        local_object,
        local_piece,
        local_chart,
        local_visibility_face,
        clip,
        rast,
        face,
        safe,
        object_map,
        piece_map,
        chart_map,
        visibility_face_map,
        world,
    )
    return result


def _raster_visibility_v8(
    np,
    torch,
    F,
    dr,
    ctx,
    vertices,
    faces,
    face_object,
    face_piece,
    corrected_w2c,
    scaled_k,
    width,
    height,
    transparent_object_ids,
    visibility_piece_count,
    self_depth_piece_ids,
):
    projection = legacy._opencv_projection(torch, scaled_k, width, height)
    mvp = projection @ corrected_w2c
    homo_vertices = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    camera_vertices = homo_vertices @ corrected_w2c.transpose(0, 1)
    clip = homo_vertices @ mvp.transpose(0, 1)

    def raster_one(
        local_faces,
        source_face_indices=None,
        *,
        local_clip=None,
        raster_width=None,
        raster_height=None,
        sampling_center=None,
        sampling_half_extent=None,
    ):
        local_clip = clip if local_clip is None else local_clip
        raster_width = width if raster_width is None else int(raster_width)
        raster_height = height if raster_height is None else int(raster_height)
        rast, _ = dr.rasterize(
            ctx,
            local_clip.unsqueeze(0).contiguous(),
            local_faces,
            [raster_height, raster_width],
        )
        local_triangle = rast[..., 3:].long() - 1
        valid = local_triangle >= 0
        safe_local = local_triangle.clamp(min=0)
        if source_face_indices is None:
            source_triangle = safe_local
        else:
            source_triangle = source_face_indices.index_select(
                0, safe_local.reshape(-1)
            ).reshape_as(safe_local)
        object_map = face_object.index_select(
            0, source_triangle.reshape(-1)
        ).reshape_as(local_triangle)
        piece_map = face_piece.index_select(
            0, source_triangle.reshape(-1)
        ).reshape_as(local_triangle)
        object_map = torch.where(valid, object_map, torch.full_like(object_map, -1))
        piece_map = torch.where(valid, piece_map, torch.full_like(piece_map, -1))
        face_map = torch.where(
            valid,
            source_triangle,
            torch.full_like(source_triangle, -1),
        )
        depth, _ = dr.interpolate(
            camera_vertices[:, 2:3].unsqueeze(0).contiguous(), rast, local_faces
        )
        interior, radius = legacy._adaptive_object_interior(
            torch,
            F,
            piece_map,
            valid,
            object_count=visibility_piece_count,
        )
        return {
            "rast": rast,
            "valid": valid.float(),
            "object": object_map,
            "piece": piece_map,
            "face": face_map.to(torch.int32),
            "depth": depth,
            "interior": interior.float(),
            "radius": radius,
            "sampling_center": sampling_center,
            "sampling_half_extent": sampling_half_extent,
        }

    body_face = face_object == bake_contract.BODY_OBJECT_ID
    if int(body_face.sum().item()) != bake_contract.BODY_V6_PROXY_TRIANGLES:
        raise ValueError("Full-z visibility does not contain the pinned BODY raster faces")
    if not bool(
        (face_piece[body_face] == bake_contract.EXPECTED_PIECES).all()
    ):
        raise ValueError("BODY raster faces do not carry the dedicated BODY piece owner")
    full = raster_one(faces)
    transparent_face = torch.isin(face_object, transparent_object_ids)
    opaque_source_faces = torch.nonzero(~transparent_face, as_tuple=False)[:, 0]
    if not len(opaque_source_faces):
        raise ValueError("Opaque visibility mesh is empty")
    opaque_faces = faces.index_select(0, opaque_source_faces)
    opaque_body_count = int(
        (face_object.index_select(0, opaque_source_faces) == bake_contract.BODY_OBJECT_ID)
        .sum()
        .item()
    )
    if opaque_body_count != bake_contract.BODY_V6_PROXY_TRIANGLES:
        raise ValueError("Opaque-z visibility dropped BODY raster geometry")
    opaque = raster_one(opaque_faces, opaque_source_faces)
    piece_self = {}
    for piece_index in map(int, self_depth_piece_ids):
        piece_source_faces = torch.nonzero(
            face_piece == piece_index, as_tuple=False
        )[:, 0]
        if not len(piece_source_faces):
            raise ValueError(
                f"Counterfactual self-depth piece {piece_index} has no raster faces"
            )
        piece_faces = faces.index_select(0, piece_source_faces)
        piece_vertex_indices = torch.unique(piece_faces.reshape(-1).long())
        piece_clip = clip.index_select(0, piece_vertex_indices)
        positive_w = piece_clip[:, 3] > 1e-6
        if not bool(positive_w.any()):
            # The main in-bounds test will reject a target behind the camera;
            # keep a finite identity crop so the proof path stays deterministic.
            crop_center = torch.zeros(2, dtype=clip.dtype, device=clip.device)
            crop_half = torch.ones(2, dtype=clip.dtype, device=clip.device)
        else:
            piece_ndc = piece_clip[positive_w, :2] / piece_clip[positive_w, 3:4]
            crop_min = piece_ndc.amin(dim=0)
            crop_max = piece_ndc.amax(dim=0)
            crop_center = (crop_min + crop_max) * 0.5
            raw_half = (crop_max - crop_min) * 0.5
            margin_scale = float(COUNTERFACTUAL_SELF_DEPTH_CROP_SIZE) / float(
                COUNTERFACTUAL_SELF_DEPTH_CROP_SIZE
                - 2 * COUNTERFACTUAL_SELF_DEPTH_CROP_MARGIN_PIXELS
            )
            minimum_half = torch.tensor(
                [1.0 / float(width), 1.0 / float(height)],
                dtype=clip.dtype,
                device=clip.device,
            )
            crop_half = torch.maximum(raw_half * margin_scale, minimum_half)
        cropped_clip = clip.clone()
        cropped_clip[:, 0] = (
            clip[:, 0] - crop_center[0] * clip[:, 3]
        ) / crop_half[0]
        cropped_clip[:, 1] = (
            clip[:, 1] - crop_center[1] * clip[:, 3]
        ) / crop_half[1]
        piece_self[piece_index] = raster_one(
            piece_faces,
            piece_source_faces,
            local_clip=cropped_clip,
            raster_width=COUNTERFACTUAL_SELF_DEPTH_CROP_SIZE,
            raster_height=COUNTERFACTUAL_SELF_DEPTH_CROP_SIZE,
            sampling_center=crop_center,
            sampling_half_extent=crop_half,
        )
    return {
        "full": full,
        "opaque": opaque,
        "piece_self": piece_self,
        "camera_vertices": camera_vertices,
        "clip": clip,
        "homo_vertices": homo_vertices,
        "opaque_source_faces": opaque_source_faces,
        "opaque_faces": opaque_faces,
    }


def _release_visibility(screen) -> None:
    screen.clear()


def _front_face_plane_depth_at_target(
    torch,
    faces,
    camera_vertices,
    sampled_visibility_face,
    target_visibility_face,
    target_camera,
):
    """Evaluate the sampled front triangle at the exact target camera ray.

    Raster depth is defined at pixel centres.  Comparing it directly with a
    sub-pixel atlas surface point introduces a geometric error much larger than
    the 0.1 mm layer-separation tolerance.  The cropped raster is used only to
    select the front triangle; its camera-space plane is then intersected with
    the exact target ray.  A same-facing normal guard prevents a rear shell
    triangle from being accepted through a coincident/near-coincident plane.
    """

    sampled_valid = sampled_visibility_face >= 0
    sampled_safe = sampled_visibility_face.clamp(min=0).long()
    target_safe = target_visibility_face.clamp(min=0).long()
    sampled_triangles = faces.index_select(0, sampled_safe)
    target_triangles = faces.index_select(0, target_safe)
    sampled_vertices = camera_vertices[:, :3].index_select(
        0, sampled_triangles.reshape(-1).long()
    ).reshape(-1, 3, 3)
    target_vertices = camera_vertices[:, :3].index_select(
        0, target_triangles.reshape(-1).long()
    ).reshape(-1, 3, 3)
    sampled_normal = torch.cross(
        sampled_vertices[:, 1] - sampled_vertices[:, 0],
        sampled_vertices[:, 2] - sampled_vertices[:, 0],
        dim=1,
    )
    target_normal = torch.cross(
        target_vertices[:, 1] - target_vertices[:, 0],
        target_vertices[:, 2] - target_vertices[:, 0],
        dim=1,
    )
    sampled_length = torch.linalg.vector_norm(sampled_normal, dim=1)
    target_length = torch.linalg.vector_norm(target_normal, dim=1)
    normal_denominator = sampled_length * target_length
    normal_cosine = (
        (sampled_normal * target_normal).sum(dim=1)
        / normal_denominator.clamp_min(1e-12)
    )
    target_camera_xyz = target_camera[:, :3]
    target_z = target_camera_xyz[:, 2]
    ray = target_camera_xyz / target_z[:, None].clamp_min(1e-8)
    plane_denominator = (sampled_normal * ray).sum(dim=1)
    plane_constant = (sampled_normal * sampled_vertices[:, 0]).sum(dim=1)
    plane_depth = plane_constant / torch.where(
        plane_denominator.abs() > 1e-12,
        plane_denominator,
        torch.ones_like(plane_denominator),
    )
    plane_valid = (
        sampled_valid
        & (target_visibility_face >= 0)
        & (sampled_length > 1e-12)
        & (target_length > 1e-12)
        & (plane_denominator.abs() > 1e-12)
        & torch.isfinite(plane_depth)
        & (normal_cosine >= COUNTERFACTUAL_SELF_DEPTH_MIN_NORMAL_COSINE)
    )
    return plane_depth, normal_cosine, plane_valid


PROJECTION_GATE_AUDIT_KEYS = (
    "geometric_visible_samples",
    "dense_weight_valid_samples",
    "mapped_object_exact_samples",
    "mapped_object_global_warp_surrogate_samples",
    "mapped_object_negative_samples",
    "mapped_object_other_samples",
    "alignment_valid_samples",
    "target_bounds_valid_samples",
    "finite_color_samples",
    "positive_score_samples",
    "accepted_samples",
    "accepted_exact_mapped_object_samples",
    "accepted_global_warp_surrogate_samples",
)


def _new_projection_gate_audit(torch, piece_count: int, device):
    return {
        "counts": {
            key: torch.zeros((piece_count,), device=device, dtype=torch.int64)
            for key in PROJECTION_GATE_AUDIT_KEYS
        },
        "mapped_object_label_counts": torch.zeros(
            (piece_count, EXPECTED_OBJECTS + 1),
            device=device,
            dtype=torch.int64,
        ),
    }


def _accumulate_projection_gate_audit(
    torch, audit, piece_id, mapped_object, conditions
) -> None:
    if audit is None:
        return
    piece_count = int(audit["mapped_object_label_counts"].shape[0])
    for key, mask in conditions.items():
        if bool(mask.any()):
            audit["counts"][key] += torch.bincount(
                piece_id[mask].long(), minlength=piece_count
            )[:piece_count]
    label_mask = conditions["dense_weight_valid_samples"] & (
        (mapped_object >= -1) & (mapped_object < EXPECTED_OBJECTS)
    )
    if bool(label_mask.any()):
        slots = EXPECTED_OBJECTS + 1
        encoded = (
            piece_id[label_mask].long() * slots
            + mapped_object[label_mask].long()
            + 1
        )
        audit["mapped_object_label_counts"] += torch.bincount(
            encoded, minlength=piece_count * slots
        )[: piece_count * slots].reshape(piece_count, slots)


def _projection_gate_audit_row(audit, piece_index: int) -> dict:
    counts = {
        key: int(audit["counts"][key][piece_index].detach().cpu().item())
        for key in PROJECTION_GATE_AUDIT_KEYS
    }
    histogram_values = (
        audit["mapped_object_label_counts"][piece_index].detach().cpu().tolist()
    )
    return {
        **counts,
        "mapped_object_label_histogram_after_dense_weight": {
            str(label): int(histogram_values[label + 1])
            for label in range(-1, EXPECTED_OBJECTS)
            if int(histogram_values[label + 1]) > 0
        },
    }


def _global_warp_surrogate_policy_for_piece(piece_id: str) -> dict | None:
    surrogates = _mapped_object_surrogates_for_piece(piece_id)
    if not surrogates:
        return None
    target_object_id, _component_id = bake_contract.parse_piece_key(piece_id)
    target_material = bake_contract.material_template_for_object(target_object_id)
    surrogate_future_materials = {
        str(object_id): bake_contract.material_template_for_object(object_id)
        for object_id in surrogates
    }
    if set(surrogate_future_materials.values()) != {target_material}:
        raise ValueError(
            f"Dense object surrogate crosses future material cohort for {piece_id}"
        )
    return {
        "policy": "future_material_cohort_global_warp_coarse_label_surrogate",
        "target_piece_id": piece_id,
        "target_object_id": target_object_id,
        "allowed_mapped_object_ids": list(surrogates),
        "minimum_dense_confidence": DENSE_OBJECT_SURROGATE_MIN_CONFIDENCE,
        "boundary_only_piece_erosion_radius_lt": 2,
        "exact_geometry_piece_owner_and_depth_required": True,
        "target_material_template": target_material,
        "surrogate_future_material_templates": surrogate_future_materials,
        "declared_source_materials": {"39": "Leather_red", "36": "Leather"},
        "source_materials_match": False,
        "surrogate_label_selects_source_geometry_or_rgb": False,
        "local_flow_from_surrogate_object_allowed": False,
    }


def _projection_gate_audit_is_consistent(row: dict, split: str) -> bool:
    audit = row.get("projection_gate_audit", {}).get(split, {})
    try:
        counts = {key: int(audit[key]) for key in PROJECTION_GATE_AUDIT_KEYS}
        histogram = {
            int(label): int(value)
            for label, value in audit[
                "mapped_object_label_histogram_after_dense_weight"
            ].items()
        }
    except (KeyError, TypeError, ValueError):
        return False
    if (
        any(value < 0 for value in counts.values())
        or any(label < -1 or label >= EXPECTED_OBJECTS for label in histogram)
        or any(value <= 0 for value in histogram.values())
        or sum(histogram.values()) != counts["dense_weight_valid_samples"]
        or counts["mapped_object_exact_samples"]
        + counts["mapped_object_global_warp_surrogate_samples"]
        + counts["mapped_object_negative_samples"]
        + counts["mapped_object_other_samples"]
        != counts["dense_weight_valid_samples"]
        or not (
            counts["geometric_visible_samples"]
            >= counts["dense_weight_valid_samples"]
            >= counts["alignment_valid_samples"]
            >= counts["target_bounds_valid_samples"]
            >= counts["finite_color_samples"]
            >= counts["positive_score_samples"]
            == counts["accepted_samples"]
        )
        or counts["accepted_exact_mapped_object_samples"]
        + counts["accepted_global_warp_surrogate_samples"]
        > counts["accepted_samples"]
        or counts["accepted_exact_mapped_object_samples"]
        > counts["mapped_object_exact_samples"]
        or counts["accepted_global_warp_surrogate_samples"]
        > counts["mapped_object_global_warp_surrogate_samples"]
    ):
        return False
    piece_id = str(row.get("piece_id", ""))
    expected_policy = _global_warp_surrogate_policy_for_piece(piece_id)
    reported_policy = row.get("video_transfer_evidence", {}).get(
        "global_warp_coarse_label_surrogate_policy"
    )
    if reported_policy != expected_policy:
        return False
    surrogate_count = counts["accepted_global_warp_surrogate_samples"]
    if surrogate_count:
        if expected_policy is None:
            return False
        allowed = set(expected_policy["allowed_mapped_object_ids"])
        if sum(histogram.get(label, 0) for label in allowed) < surrogate_count:
            return False
    if split == "training":
        expected_accepted = int(row.get("accepted_projection_samples", -1))
    elif split == "holdout":
        expected_accepted = int(row.get("holdout_samples", -1))
    else:
        return False
    if (
        piece_id == "39:-1"
        and counts["accepted_samples"]
        != counts["accepted_exact_mapped_object_samples"]
        + counts["accepted_global_warp_surrogate_samples"]
    ):
        return False
    return counts["accepted_samples"] == expected_accepted


def _steering_dense_provenance_floor(
    row: dict, split: str, minimum_samples: int
) -> bool:
    if str(row.get("piece_id", "")) != "39:-1":
        return True
    if not _projection_gate_audit_is_consistent(row, split):
        return False
    audit = row["projection_gate_audit"][split]
    exact = int(audit["accepted_exact_mapped_object_samples"])
    surrogate = int(audit["accepted_global_warp_surrogate_samples"])
    return bool(exact >= minimum_samples or surrogate >= minimum_samples)


def _winner_evidence_classification(winner_texels: dict) -> str:
    try:
        exact = int(winner_texels["exact_dense_label"])
        surrogate = int(winner_texels["global_warp_coarse_label_surrogate"])
        nonlocal_dense = int(winner_texels["permitted_nonlocal_dense"])
    except (KeyError, TypeError, ValueError):
        return "invalid"
    if min(exact, surrogate, nonlocal_dense) < 0:
        return "invalid"
    if exact and surrogate:
        return "mixed_exact_label_and_global_warp_coarse_label_surrogate_video_transfer"
    if surrogate:
        return "future_material_cohort_global_warp_coarse_label_surrogate_video_transfer"
    if exact and nonlocal_dense:
        return "mixed_exact_label_and_permitted_nonlocal_dense_video_transfer"
    if exact:
        return "direct_exact_piece_video_transfer"
    if nonlocal_dense:
        return "permitted_nonlocal_dense_video_transfer"
    return "none"


def _winner_evidence_is_supported_by_projection(
    piece_id: str, winner_texels: dict, training_projection: dict
) -> bool:
    try:
        exact_winners = int(winner_texels["exact_dense_label"])
        surrogate_winners = int(
            winner_texels["global_warp_coarse_label_surrogate"]
        )
        nonlocal_winners = int(winner_texels["permitted_nonlocal_dense"])
        accepted = int(training_projection["accepted_samples"])
        exact_accepted = int(
            training_projection["accepted_exact_mapped_object_samples"]
        )
        surrogate_accepted = int(
            training_projection["accepted_global_warp_surrogate_samples"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    nonlocal_accepted = accepted - exact_accepted - surrogate_accepted
    return bool(
        min(
            exact_winners,
            surrogate_winners,
            nonlocal_winners,
            exact_accepted,
            surrogate_accepted,
            nonlocal_accepted,
        )
        >= 0
        and (exact_winners == 0 or exact_accepted > 0)
        and (surrogate_winners == 0 or surrogate_accepted > 0)
        and (nonlocal_winners == 0 or nonlocal_accepted > 0)
        and (str(piece_id) == "39:-1" or surrogate_winners == 0)
        and (str(piece_id) != "39:-1" or nonlocal_winners == 0)
    )


def _frame_piece_candidates(
    np,
    torch,
    F,
    dr,
    ctx,
    vertices,
    faces,
    face_object,
    face_piece,
    page_face_normals,
    transparent_object_ids,
    visibility_piece_count,
    piece_target_full_z,
    atlas_surface,
    occupied_indices,
    intrinsics,
    payload,
    target,
    native_source_width,
    native_source_height,
    rgb_width,
    rgb_height,
    width,
    height,
    chunk_size=1_000_000,
    visibility_audit=None,
    visibility_audit_key=None,
    piece_owner_pixel_audit=None,
    opportunity_mask_flat=None,
    projection_gate_audit=None,
):
    """Yield exact piece-owned inverse-projection samples from one frame."""

    device = vertices.device
    if tuple(target.shape[1:3]) != (int(rgb_height), int(rgb_width)):
        raise ValueError("RGB tensor dimensions do not match radiometry contract")
    corrected_w2c = torch.from_numpy(
        np.asarray(payload["corrected_w2c_cv"], dtype=np.float32)
    ).to(device)
    corrected_c2w = torch.from_numpy(
        np.asarray(payload["corrected_c2w_cv"], dtype=np.float32)
    ).to(device)
    scaled_k = legacy._scale_intrinsics(
        torch,
        intrinsics,
        native_source_width,
        native_source_height,
        width,
        height,
    )
    page_piece_ids = torch.unique(
        atlas_surface["piece"].reshape(-1).index_select(0, occupied_indices)
    ).long()
    self_depth_piece_ids = [
        int(value)
        for value in page_piece_ids.detach().cpu().tolist()
        if int(value) in COUNTERFACTUAL_SELF_DEPTH_PIECE_INDICES
    ]
    screen = _raster_visibility_v8(
        np,
        torch,
        F,
        dr,
        ctx,
        vertices,
        faces,
        face_object,
        face_piece,
        corrected_w2c,
        scaled_k,
        width,
        height,
        transparent_object_ids,
        visibility_piece_count,
        self_depth_piece_ids,
    )
    if visibility_audit is not None:
        if visibility_audit_key is None:
            raise ValueError("visibility_audit_key is required with visibility_audit")
        body_full_pixels = int(
            (screen["full"]["object"] == bake_contract.BODY_OBJECT_ID).sum().item()
        )
        body_opaque_pixels = int(
            (screen["opaque"]["object"] == bake_contract.BODY_OBJECT_ID).sum().item()
        )
        visibility_audit[str(visibility_audit_key)] = {
            "body_full_z_raster_pixels": body_full_pixels,
            "body_opaque_z_raster_pixels": body_opaque_pixels,
        }
    if piece_owner_pixel_audit is not None:
        if visibility_audit_key is None:
            raise ValueError(
                "visibility_audit_key is required with piece_owner_pixel_audit"
            )
        counts_by_buffer = {}
        for name in ("full", "opaque"):
            owner = screen[name]["piece"].reshape(-1).long()
            valid_owner = (owner >= 0) & (owner < visibility_piece_count)
            counts_by_buffer[name] = torch.bincount(
                owner[valid_owner], minlength=visibility_piece_count
            )[:visibility_piece_count]
        selected_counts = torch.where(
            piece_target_full_z,
            counts_by_buffer["full"],
            counts_by_buffer["opaque"],
        )
        # Transmissive parts are removable layers.  Count each exact target
        # against opaque occlusion plus its own nearest surface, deliberately
        # excluding sibling transmissive layers that hide it in frontmost-z.
        for piece_index, self_buffer in screen["piece_self"].items():
            self_piece = self_buffer["piece"].reshape(-1)
            self_depth = self_buffer["depth"].reshape(-1)
            self_valid = (self_piece == int(piece_index)) & (
                self_buffer["interior"].reshape(-1) > 0.5
            )
            crop_height = int(self_buffer["piece"].shape[1])
            crop_width = int(self_buffer["piece"].shape[2])
            crop_y, crop_x = torch.meshgrid(
                (
                    2.0
                    * (torch.arange(crop_height, device=device) + 0.5)
                    / float(crop_height)
                    - 1.0,
                    2.0
                    * (torch.arange(crop_width, device=device) + 0.5)
                    / float(crop_width)
                    - 1.0,
                ),
                indexing="ij",
            )
            crop_grid = torch.stack([crop_x, crop_y], dim=-1).reshape(-1, 2)
            full_grid = (
                self_buffer["sampling_center"][None]
                + crop_grid * self_buffer["sampling_half_extent"][None]
            )
            opaque_valid_weight = legacy._sample_grid(
                F,
                screen["opaque"]["valid"].permute(0, 3, 1, 2),
                full_grid,
            )[:, 0]
            opaque_depth = (
                legacy._sample_grid(
                    F,
                    (
                        screen["opaque"]["depth"] * screen["opaque"]["valid"]
                    ).permute(0, 3, 1, 2),
                    full_grid,
                )[:, 0]
                / opaque_valid_weight.clamp_min(1e-8)
            )
            opaque_piece = legacy._sample_grid(
                F,
                screen["opaque"]["piece"].float().permute(0, 3, 1, 2),
                full_grid,
                mode="nearest",
            )[:, 0].round().long()
            opaque_valid = opaque_piece >= 0
            opaque_tolerance = torch.maximum(
                torch.full_like(self_depth, 0.003), self_depth.abs() * 0.0015
            )
            opaque_clear = (~opaque_valid) | (
                self_depth <= opaque_depth + opaque_tolerance
            )
            full_x = (
                (full_grid[:, 0] + 1.0) * float(width) * 0.5 - 0.5
            ).round().long()
            full_y = (
                (full_grid[:, 1] + 1.0) * float(height) * 0.5 - 0.5
            ).round().long()
            in_frame = (
                (full_x >= 0)
                & (full_x < width)
                & (full_y >= 0)
                & (full_y < height)
            )
            selected_screen_pixels = torch.unique(
                full_y[self_valid & opaque_clear & in_frame] * width
                + full_x[self_valid & opaque_clear & in_frame]
            )
            selected_counts[int(piece_index)] = int(
                selected_screen_pixels.numel()
            )
        piece_owner_pixel_audit[str(visibility_audit_key)] = [
            int(value) for value in selected_counts.detach().cpu().tolist()
        ]

    dense_xy = torch.from_numpy(payload["dense_render_to_target_xy"]).to(
        device=device, dtype=torch.float32
    )
    dense_valid = torch.from_numpy(payload["dense_valid"]).to(
        device=device, dtype=torch.float32
    )
    dense_confidence = torch.from_numpy(payload["dense_confidence"]).to(
        device=device, dtype=torch.float32
    )
    dense_object = torch.from_numpy(payload["dense_object_id"].astype("int32")).to(
        device
    )
    dense_xy_nchw = (dense_xy * dense_valid[..., None]).permute(2, 0, 1).unsqueeze(0)
    dense_valid_nchw = dense_valid[None, None]
    dense_confidence_nchw = (dense_confidence * dense_valid)[None, None]
    dense_object_nchw = dense_object.float()[None, None]
    target_nchw = target.permute(0, 3, 1, 2)
    sharpness_nchw = legacy._sharpness_map(torch, F, target).permute(0, 3, 1, 2)
    camera_position = corrected_c2w[:3, 3]
    atlas_world = atlas_surface["world"].reshape(-1, 3)
    atlas_face = atlas_surface["face"].reshape(-1)
    atlas_visibility_face = atlas_surface["visibility_face"].reshape(-1)
    atlas_object = atlas_surface["object"].reshape(-1)
    atlas_piece = atlas_surface["piece"].reshape(-1)
    atlas_chart = atlas_surface["chart"].reshape(-1)

    for start in range(0, len(occupied_indices), chunk_size):
        atlas_index = occupied_indices[start : start + chunk_size]
        world = atlas_world.index_select(0, atlas_index)
        local_face_id = atlas_face.index_select(0, atlas_index).long()
        target_visibility_face = atlas_visibility_face.index_select(
            0, atlas_index
        ).long()
        object_id = atlas_object.index_select(0, atlas_index).to(torch.int32)
        piece_id = atlas_piece.index_select(0, atlas_index).to(torch.int32)
        chart_id = atlas_chart.index_select(0, atlas_index).long()
        homo = torch.cat([world, torch.ones_like(world[:, :1])], dim=-1)
        camera = homo @ corrected_w2c.transpose(0, 1)
        z = camera[:, 2]
        render_x = scaled_k[0, 0] * camera[:, 0] / z.clamp_min(1e-8) + scaled_k[0, 2]
        render_y = scaled_k[1, 1] * camera[:, 1] / z.clamp_min(1e-8) + scaled_k[1, 2]
        render_grid = torch.stack(
            [
                2.0 * (render_x + 0.5) / float(width) - 1.0,
                2.0 * (render_y + 0.5) / float(height) - 1.0,
            ],
            dim=-1,
        )
        in_bounds = (
            (z > 0.05)
            & (render_x >= -0.5)
            & (render_x <= width - 0.5)
            & (render_y >= -0.5)
            & (render_y <= height - 0.5)
        )

        sampled = {}
        for name in ("full", "opaque"):
            buffer = screen[name]
            valid_weight = legacy._sample_grid(
                F, buffer["valid"].permute(0, 3, 1, 2), render_grid
            )[:, 0]
            sampled[name] = {
                "depth": (
                    legacy._sample_grid(
                        F,
                        (buffer["depth"] * buffer["valid"]).permute(0, 3, 1, 2),
                        render_grid,
                    )[:, 0]
                    / valid_weight.clamp_min(1e-8)
                ),
                "valid_weight": valid_weight,
                "object": legacy._sample_grid(
                    F,
                    buffer["object"].float().permute(0, 3, 1, 2),
                    render_grid,
                    mode="nearest",
                )[:, 0].round().to(torch.int32),
                "piece": legacy._sample_grid(
                    F,
                    buffer["piece"].float().permute(0, 3, 1, 2),
                    render_grid,
                    mode="nearest",
                )[:, 0].round().to(torch.int32),
                "interior": legacy._sample_grid(
                    F,
                    buffer["interior"].permute(0, 3, 1, 2),
                    render_grid,
                    mode="nearest",
                )[:, 0],
                "radius": legacy._sample_grid(
                    F,
                    buffer["radius"].float().permute(0, 3, 1, 2),
                    render_grid,
                    mode="nearest",
                )[:, 0].round().to(torch.int8),
            }

        target_is_transmissive = torch.isin(object_id, transparent_object_ids)
        sampled_depth = torch.where(
            target_is_transmissive,
            sampled["full"]["depth"],
            sampled["opaque"]["depth"],
        )
        sampled_object = torch.where(
            target_is_transmissive,
            sampled["full"]["object"],
            sampled["opaque"]["object"],
        )
        sampled_piece = torch.where(
            target_is_transmissive,
            sampled["full"]["piece"],
            sampled["opaque"]["piece"],
        )
        sampled_interior = torch.where(
            target_is_transmissive,
            sampled["full"]["interior"],
            sampled["opaque"]["interior"],
        )
        sampled_radius = torch.where(
            target_is_transmissive,
            sampled["full"]["radius"],
            sampled["opaque"]["radius"],
        )
        self_depth_target = torch.zeros_like(target_is_transmissive)
        self_depth_opaque_clear = torch.ones_like(target_is_transmissive)
        self_depth_valid_weight = torch.ones_like(z)
        self_depth_plane_valid = torch.ones_like(target_is_transmissive)
        self_depth_normal_cosine = torch.ones_like(z)
        for piece_index, self_buffer in screen["piece_self"].items():
            target_mask = piece_id == int(piece_index)
            if not bool(target_mask.any()):
                continue
            target_grid_subset = (
                render_grid[target_mask]
                - self_buffer["sampling_center"][None]
            ) / self_buffer["sampling_half_extent"][None]
            self_valid_weight = legacy._sample_grid(
                F,
                self_buffer["valid"].permute(0, 3, 1, 2),
                target_grid_subset,
            )[:, 0]
            self_values = {
                "depth": (
                    legacy._sample_grid(
                        F,
                        (
                            self_buffer["depth"] * self_buffer["valid"]
                        ).permute(0, 3, 1, 2),
                        target_grid_subset,
                    )[:, 0]
                    / self_valid_weight.clamp_min(1e-8)
                ),
                "object": legacy._sample_grid(
                    F,
                    self_buffer["object"].float().permute(0, 3, 1, 2),
                    target_grid_subset,
                    mode="nearest",
                )[:, 0].round().to(torch.int32),
                "piece": legacy._sample_grid(
                    F,
                    self_buffer["piece"].float().permute(0, 3, 1, 2),
                    target_grid_subset,
                    mode="nearest",
                )[:, 0].round().to(torch.int32),
                "interior": legacy._sample_grid(
                    F,
                    self_buffer["interior"].permute(0, 3, 1, 2),
                    target_grid_subset,
                )[:, 0],
                "radius": legacy._sample_grid(
                    F,
                    self_buffer["radius"].float().permute(0, 3, 1, 2),
                    target_grid_subset,
                    mode="nearest",
                )[:, 0].round().to(torch.int8),
                "face": legacy._sample_grid(
                    F,
                    self_buffer["face"].float().permute(0, 3, 1, 2),
                    target_grid_subset,
                    mode="nearest",
                )[:, 0].round().long(),
            }
            plane_depth, normal_cosine, plane_valid = _front_face_plane_depth_at_target(
                torch,
                faces,
                screen["camera_vertices"],
                self_values["face"],
                target_visibility_face[target_mask],
                camera[target_mask],
            )
            # Keep the normalised raster-depth calculation above as an
            # explicit coverage diagnostic, but compare the exact target ray
            # against the selected front triangle plane.  This avoids both
            # zero-background dilution and pixel-centre depth error.
            sampled_depth[target_mask] = plane_depth
            sampled_object[target_mask] = self_values["object"]
            sampled_piece[target_mask] = self_values["piece"]
            sampled_interior[target_mask] = self_values["interior"]
            sampled_radius[target_mask] = self_values["radius"]
            self_depth_target[target_mask] = True
            self_depth_valid_weight[target_mask] = self_valid_weight
            self_depth_plane_valid[target_mask] = plane_valid
            self_depth_normal_cosine[target_mask] = normal_cosine
            opaque_depth_subset = sampled["opaque"]["depth"][target_mask]
            opaque_valid_subset = sampled["opaque"]["piece"][target_mask] >= 0
            target_z = z[target_mask]
            opaque_tolerance = torch.maximum(
                torch.full_like(target_z, 0.003), target_z.abs() * 0.0015
            )
            self_depth_opaque_clear[target_mask] = (~opaque_valid_subset) | (
                target_z <= opaque_depth_subset + opaque_tolerance
            )
        tolerance = torch.maximum(torch.full_like(z, 0.003), z.abs() * 0.0015)
        visibility_depth_tolerance = torch.where(
            self_depth_target,
            torch.full_like(z, COUNTERFACTUAL_SELF_DEPTH_TOLERANCE_METERS),
            tolerance,
        )
        visible = (
            in_bounds
            & (sampled_piece == piece_id)
            & (sampled_object == object_id)
            & (sampled_interior > 0.5)
            & ((sampled_depth - z).abs() <= visibility_depth_tolerance)
            & self_depth_opaque_clear
            & self_depth_plane_valid
            & (
                self_depth_valid_weight
                > COUNTERFACTUAL_SELF_DEPTH_MIN_VALID_WEIGHT
            )
        )
        if opportunity_mask_flat is not None and bool(visible.any()):
            # This mask is deliberately recorded before dense-warp validity,
            # target bounds, colour sampling and score checks.  It is an
            # independent geometric opportunity, so a broken projection path
            # cannot label its own failures as "not visible".
            opportunity_mask_flat[atlas_index[visible]] = True

        dense_weight = legacy._sample_grid(F, dense_valid_nchw, render_grid)[:, 0]
        mapped_xy = legacy._sample_grid(
            F, dense_xy_nchw, render_grid
        ) / dense_weight[:, None].clamp_min(1e-8)
        mapped_confidence = (
            legacy._sample_grid(F, dense_confidence_nchw, render_grid)[:, 0]
            / dense_weight.clamp_min(1e-8)
        ).clamp(0, 1)
        mapped_object = legacy._sample_grid(
            F, dense_object_nchw, render_grid, mode="nearest"
        )[:, 0].round().to(torch.int32)
        mapped_is_transmissive = torch.isin(mapped_object, transparent_object_ids)
        glass_warp_proxy = (~target_is_transmissive) & mapped_is_transmissive
        relaxed_boundary = sampled_radius < 2
        mapped_object_surrogate = torch.zeros_like(visible)
        for piece_key, surrogate_ids in DENSE_OBJECT_SURROGATE_POLICY.items():
            piece_index = DENSE_OBJECT_SURROGATE_PIECE_INDEX[piece_key]
            surrogate_label = torch.zeros_like(visible)
            for surrogate_object_id in surrogate_ids:
                if _alignment_surrogate_label_has_no_local_flow(
                    np, payload, surrogate_object_id
                ):
                    surrogate_label |= mapped_object == int(surrogate_object_id)
            mapped_object_surrogate |= (
                (piece_id == piece_index)
                & surrogate_label
                & (mapped_confidence >= DENSE_OBJECT_SURROGATE_MIN_CONFIDENCE)
            )
        normal_alignment_valid = (
            (mapped_object < 0) | (mapped_object == object_id) | glass_warp_proxy
        )
        # The explicit coarse-label surrogate is boundary-only. At higher resolution, if the
        # same piece has a stable radius-2 interior, exact mapped-object
        # identity remains mandatory.
        relaxed_alignment_valid = (
            (mapped_object == object_id) | mapped_object_surrogate
        ) & (mapped_confidence >= 0.5)
        alignment_valid = (dense_weight > 0.75) & torch.where(
            relaxed_boundary, relaxed_alignment_valid, normal_alignment_valid
        )
        target_bounds = (
            (mapped_xy[:, 0] >= -0.5)
            & (mapped_xy[:, 0] <= native_source_width - 0.5)
            & (mapped_xy[:, 1] >= -0.5)
            & (mapped_xy[:, 1] <= native_source_height - 0.5)
        )
        rgb_x = _native_to_rgb_pixel_center(
            mapped_xy[:, 0], native_source_width, rgb_width
        )
        rgb_y = _native_to_rgb_pixel_center(
            mapped_xy[:, 1], native_source_height, rgb_height
        )
        target_grid = torch.stack(
            [
                2.0 * (rgb_x + 0.5) / float(rgb_width) - 1.0,
                2.0 * (rgb_y + 0.5) / float(rgb_height) - 1.0,
            ],
            dim=-1,
        )
        color = legacy._sample_grid(F, target_nchw, target_grid)
        sharpness = legacy._sample_grid(F, sharpness_nchw, target_grid)[:, 0]
        normal = page_face_normals.index_select(0, local_face_id)
        view = F.normalize(camera_position[None] - world, dim=-1, eps=1e-8)
        facing = (normal * view).sum(dim=-1).abs().clamp(0, 1)
        score = mapped_confidence * (0.08 + 0.92 * facing.square()) * (
            0.82 + 0.18 * sharpness
        )
        accepted = (
            visible
            & alignment_valid
            & target_bounds
            & torch.isfinite(color).all(dim=-1)
            & (score > 1e-5)
        )
        dense_weight_valid = visible & (dense_weight > 0.75)
        alignment_funnel = visible & alignment_valid
        target_bounds_funnel = alignment_funnel & target_bounds
        finite_funnel = target_bounds_funnel & torch.isfinite(color).all(dim=-1)
        _accumulate_projection_gate_audit(
            torch,
            projection_gate_audit,
            piece_id,
            mapped_object,
            {
                "geometric_visible_samples": visible,
                "dense_weight_valid_samples": dense_weight_valid,
                "mapped_object_exact_samples": dense_weight_valid
                & (mapped_object == object_id),
                "mapped_object_global_warp_surrogate_samples": dense_weight_valid
                & relaxed_boundary
                & mapped_object_surrogate,
                "mapped_object_negative_samples": dense_weight_valid
                & (mapped_object < 0),
                "mapped_object_other_samples": dense_weight_valid
                & (mapped_object >= 0)
                & (mapped_object != object_id)
                & ~(relaxed_boundary & mapped_object_surrogate),
                "alignment_valid_samples": alignment_funnel,
                "target_bounds_valid_samples": target_bounds_funnel,
                "finite_color_samples": finite_funnel,
                "positive_score_samples": finite_funnel & (score > 1e-5),
                "accepted_samples": accepted,
                "accepted_exact_mapped_object_samples": accepted
                & (mapped_object == object_id),
                "accepted_global_warp_surrogate_samples": accepted
                & relaxed_boundary
                & mapped_object_surrogate,
            },
        )
        if bool(accepted.any()):
            video_evidence_class = torch.where(
                mapped_object_surrogate,
                torch.full_like(mapped_object, 2, dtype=torch.uint8),
                torch.where(
                    mapped_object == object_id,
                    torch.ones_like(mapped_object, dtype=torch.uint8),
                    torch.full_like(mapped_object, 3, dtype=torch.uint8),
                ),
            )
            yield {
                "atlas_index": atlas_index[accepted],
                "face_id": local_face_id[accepted],
                "object_id": object_id[accepted],
                "piece_id": piece_id[accepted],
                "chart_id": chart_id[accepted],
                "color": color[accepted],
                "score": score[accepted],
                "glass_warp_proxy": glass_warp_proxy[accepted],
                "interior_radius": sampled_radius[accepted],
                "target_used_full_z": (
                    target_is_transmissive & ~self_depth_target
                )[accepted],
                "target_used_piece_self_z": self_depth_target[accepted],
                "target_self_front_face_normal_cosine": self_depth_normal_cosine[
                    accepted
                ],
                "mapped_object_surrogate": mapped_object_surrogate[accepted],
                "video_evidence_class": video_evidence_class[accepted],
            }

    _release_visibility(screen)
    del (
        corrected_w2c,
        corrected_c2w,
        scaled_k,
        dense_xy,
        dense_valid,
        dense_confidence,
        dense_object,
        dense_xy_nchw,
        dense_valid_nchw,
        dense_confidence_nchw,
        dense_object_nchw,
        target_nchw,
        sharpness_nchw,
        atlas_world,
        atlas_face,
        atlas_object,
        atlas_piece,
        atlas_chart,
    )


def _shift_tensor(torch, value, dy: int, dx: int, fill):
    output = torch.full_like(value, fill)
    height, width = value.shape[:2]
    source_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_y = slice(max(0, dy), min(height, height + dy))
    target_x = slice(max(0, dx), min(width, width + dx))
    output[target_y, target_x] = value[source_y, source_x]
    return output


MATERIAL_NEUTRAL_PRIORS = {
    "v8_glass_transmission": (0.10, 0.12, 0.14),
    "v8_mirror_chrome": (0.18, 0.18, 0.18),
    "v8_lamp_lens": (0.12, 0.08, 0.08),
    "v8_lamp_emissive": (0.18, 0.07, 0.04),
    "v8_metal": (0.15, 0.15, 0.16),
    "v8_rubber": (0.035, 0.035, 0.035),
    "v8_carbon": (0.055, 0.055, 0.06),
    "v8_fabric": (0.09, 0.09, 0.095),
    "v8_leather": (0.08, 0.07, 0.065),
    "v8_plastic": (0.075, 0.075, 0.08),
    "v8_paint_detail": (0.12, 0.12, 0.13),
    "v8_exterior_opaque": (0.12, 0.12, 0.13),
}


def _pull_push_fill_domain(torch, F, color, known, domain_mask, neutral_rgb):
    """Fill exactly one ``(piece_id, chart_id)`` domain.

    Both the multiscale numerator and weight are masked by ``domain_mask``.
    Calling this once per chart is important: pooling two charts owned by the
    same piece together can otherwise leak colour across a one-pixel gap.
    """

    if not bool(domain_mask.any()):
        return color
    if not bool((known & domain_mask).any()):
        return torch.where(domain_mask[..., None], neutral_rgb.view(1, 1, 3), color)
    value = color.permute(2, 0, 1).unsqueeze(0)
    weight = (known & domain_mask).float().unsqueeze(0).unsqueeze(0)
    domain = domain_mask.float().unsqueeze(0).unsqueeze(0)
    pyramid = [(value, weight, domain)]
    while min(pyramid[-1][0].shape[-2:]) > 2 and len(pyramid) < 14:
        previous_value, previous_weight, previous_domain = pyramid[-1]
        next_weight = F.avg_pool2d(previous_weight, 2, 2)
        numerator = F.avg_pool2d(previous_value * previous_weight, 2, 2)
        next_value = numerator / next_weight.clamp_min(1e-8)
        next_domain = F.max_pool2d(previous_domain, 2, 2)
        pyramid.append((next_value, next_weight, next_domain))
    coarse_value, coarse_weight, _coarse_domain = pyramid[-1]
    owned_rgb = (value * weight).sum(dim=(0, 2, 3)) / weight.sum().clamp_min(1)
    filled = torch.where(
        (coarse_weight > 1e-8).expand_as(coarse_value),
        coarse_value,
        owned_rgb.view(1, 3, 1, 1),
    )
    for current_value, current_weight, current_domain in reversed(pyramid[:-1]):
        filled = F.interpolate(
            filled, current_value.shape[-2:], mode="bilinear", align_corners=False
        )
        filled = torch.where(
            (current_weight > 1e-8).expand_as(current_value),
            current_value,
            filled,
        )
        filled = torch.where(
            current_domain.expand_as(filled) > 0,
            filled,
            torch.zeros_like(filled),
        )
    output = filled[0].permute(1, 2, 0)
    return torch.where(domain_mask[..., None], output, color)


def _fill_by_piece_chart(
    torch,
    F,
    color,
    known,
    piece_map,
    chart_map,
    piece_material,
    piece_count: int,
):
    """Strict ``(piece_id, chart_id)`` fill with material-local neutral priors.

    No observed RGB statistic is shared between charts or pieces.  A chart
    without video samples receives only its own piece's fixed semantic material
    prior; it never receives a page/global mean or a neighbouring chart colour.
    """

    output = color.clone()
    generated = torch.zeros_like(known)
    modes = []
    for piece_index in range(piece_count):
        piece_mask = piece_map == piece_index
        if not bool(piece_mask.any()):
            modes.append(
                {
                    "mode": "no_uv_occupancy",
                    "chart_domains": 0,
                    "observed_chart_domains": 0,
                    "neutral_chart_domains": 0,
                }
            )
            continue
        material_name = str(piece_material[piece_index])
        if material_name not in MATERIAL_NEUTRAL_PRIORS:
            raise ValueError(f"No deterministic neutral prior for {material_name}")
        neutral = torch.tensor(
            MATERIAL_NEUTRAL_PRIORS[material_name],
            device=color.device,
            dtype=color.dtype,
        )
        chart_ids = torch.unique(chart_map[piece_mask]).detach().cpu().tolist()
        observed_domains = 0
        for chart_id in sorted(map(int, chart_ids)):
            domain_mask = piece_mask & (chart_map == chart_id)
            coordinates = torch.nonzero(domain_mask, as_tuple=False)
            y0, x0 = coordinates.min(dim=0).values.tolist()
            y1, x1 = (coordinates.max(dim=0).values + 1).tolist()
            crop_mask = domain_mask[y0:y1, x0:x1]
            crop_known = known[y0:y1, x0:x1]
            crop_color = output[y0:y1, x0:x1]
            if bool((crop_known & crop_mask).any()):
                observed_domains += 1
            output[y0:y1, x0:x1] = _pull_push_fill_domain(
                torch, F, crop_color, crop_known, crop_mask, neutral
            )
            generated[y0:y1, x0:x1] |= crop_mask & ~crop_known
        domain_count = len(chart_ids)
        modes.append(
            {
                "mode": (
                    "all_chart_domains_from_owned_video"
                    if observed_domains == domain_count
                    else "all_chart_domains_neutral"
                    if observed_domains == 0
                    else "mixed_observed_and_neutral_chart_domains"
                ),
                "chart_domains": domain_count,
                "observed_chart_domains": observed_domains,
                "neutral_chart_domains": domain_count - observed_domains,
                "material_neutral_prior": list(MATERIAL_NEUTRAL_PRIORS[material_name]),
            }
        )
    return output, generated, modes


def _pad_gutters_by_piece_chart(torch, color, occupied, piece_map, chart_map, padding):
    """Deterministic nearest-domain gutters with zero cross-domain averaging."""

    chart_slots = int(chart_map[occupied].max().item()) + 1
    domain = piece_map.to(torch.int32) * chart_slots + chart_map.to(torch.int32)
    owner = torch.where(occupied, domain, torch.full_like(domain, -1))
    value = color.clone()
    max_owner = torch.iinfo(torch.int32).max
    directions = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )
    for _ in range(max(0, int(padding))):
        empty = owner < 0
        best = torch.full_like(owner, max_owner)
        best_color = torch.zeros_like(value)
        for dy, dx in directions:
            candidate_owner = _shift_tensor(torch, owner, dy, dx, -1)
            choose = empty & (candidate_owner >= 0) & (candidate_owner < best)
            best = torch.where(choose, candidate_owner, best)
            candidate_color = _shift_tensor(torch, value, dy, dx, 0.0)
            best_color = torch.where(choose[..., None], candidate_color, best_color)
        frontier = best != max_owner
        if not bool(frontier.any()):
            break
        owner = torch.where(frontier, best, owner)
        value = torch.where(frontier[..., None], best_color, value)
    padded = owner >= 0
    return value.contiguous(), padded, owner, chart_slots


def _evaluate_page_holdouts(
    np,
    torch,
    F,
    dr,
    ctx,
    runtime,
    alignment,
    targets,
    holdout_ids,
    page,
    texture,
    surface,
    occupied_indices,
    vertices,
    faces,
    face_object,
    face_piece,
    piece_target_full_z,
    page_normals,
    transparent,
    camera_lookup,
    width,
    height,
    PIL_Image,
    visibility_audit,
    piece_owner_pixel_audit,
    opportunity_mask_flat,
    projection_gate_audit,
):
    texture_flat = texture.reshape(-1, 3)
    frame_l1: list[float | None] = []
    frame_samples: list[int] = []
    per_piece_samples: dict[str, dict[str, int]] = {}
    piece_sample_totals = {key: 0 for key in page.piece_keys}
    for frame_id in holdout_ids:
        target = legacy._load_rgb(
            PIL_Image, np, torch, targets[frame_id], vertices.device
        )
        numerator = 0.0
        denominator = 0
        piece_counts = {key: 0 for key in page.piece_keys}
        candidates = _frame_piece_candidates(
            np,
            torch,
            F,
            dr,
            ctx,
            vertices,
            faces,
            face_object,
            face_piece,
            page_normals,
            transparent,
            runtime.visibility_piece_count,
            piece_target_full_z,
            surface,
            occupied_indices,
            torch.from_numpy(runtime.source.K[camera_lookup[frame_id]]).to(vertices.device),
            alignment.frames[frame_id],
            target,
            runtime.source.width,
            runtime.source.height,
            bake_contract.SEEDVR_RGB_WIDTH,
            bake_contract.SEEDVR_RGB_HEIGHT,
            width,
            height,
            visibility_audit=visibility_audit,
            visibility_audit_key=f"holdout:{int(frame_id)}",
            piece_owner_pixel_audit=piece_owner_pixel_audit,
            opportunity_mask_flat=opportunity_mask_flat,
            projection_gate_audit=projection_gate_audit,
        )
        for candidate in candidates:
            predicted = texture_flat.index_select(0, candidate["atlas_index"])
            numerator += float(
                (predicted - candidate["color"]).abs().mean(dim=-1).sum().item()
            )
            denominator += int(len(predicted))
            values, counts = torch.unique(candidate["piece_id"], return_counts=True)
            for value, count in zip(values.cpu().tolist(), counts.cpu().tolist()):
                piece_counts[runtime.piece_keys[int(value)]] += int(count)
        frame_l1.append(numerator / denominator if denominator else None)
        frame_samples.append(denominator)
        per_piece_samples[str(frame_id)] = piece_counts
        for key, count in piece_counts.items():
            piece_sample_totals[key] += int(count)
        del target, candidates
    valid = [value for value in frame_l1 if value is not None]
    return {
        "method": (
            "exact_geometry_piece_owned_projection_with_reported_dense_label_provenance"
        ),
        "mean_aligned_l1": float(sum(valid) / len(valid)) if valid else None,
        "per_frame_l1": frame_l1,
        "per_frame_samples": frame_samples,
        "frames_with_samples": int(sum(value > 0 for value in frame_samples)),
        "per_piece_samples_by_frame": per_piece_samples,
        "per_piece_sample_totals": piece_sample_totals,
    }


def _bake_one_page(
    np,
    torch,
    F,
    dr,
    ctx,
    runtime,
    alignment,
    page,
    train_ids,
    holdout_ids,
    targets,
    camera_lookup,
    vertices,
    faces,
    face_object,
    face_piece,
    piece_target_full_z,
    transparent,
    alignment_supported,
    training_alignment_evidence,
    width,
    height,
    atlas_size,
    atlas_padding,
    output_dir,
    PIL_Image,
):
    diagnostic_dir = output_dir / "diagnostics" / page.page_id
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    texture_dir = output_dir / "textures"
    device = vertices.device
    page_normals = F.normalize(
        torch.from_numpy(page.face_normals_world).to(device=device, dtype=torch.float32),
        dim=-1,
        eps=1e-8,
    )
    surface = _uv_surface_map_v8(torch, dr, ctx, page, atlas_size, device)
    occupied = surface["occupied"]
    atlas_piece = surface["piece"]
    occupied_flat = occupied.reshape(-1)
    occupied_indices = torch.nonzero(occupied_flat, as_tuple=False)[:, 0]
    if not len(occupied_indices):
        raise ValueError(f"{page.page_id} rasterises to zero occupied texels")
    total_texels = atlas_size * atlas_size
    best_color = torch.zeros((total_texels, 3), device=device)
    best_score = torch.full((total_texels,), -1.0, device=device)
    second_score = torch.full_like(best_score, -1.0)
    best_frame = torch.full(
        (total_texels,), 65535, device=device, dtype=torch.int32
    )
    best_evidence_class = torch.zeros(
        (total_texels,), device=device, dtype=torch.uint8
    )
    support = torch.zeros((total_texels,), device=device, dtype=torch.int16)
    chart_count = page.chart_count
    occupied_chart = surface["chart"].reshape(-1).index_select(
        0, occupied_indices
    ).long()
    chart_occupancy = torch.zeros((chart_count,), device=device)
    chart_occupancy.scatter_add_(
        0, occupied_chart, torch.ones(len(occupied_indices), device=device)
    )
    best_chart_score = torch.full((chart_count,), -1.0, device=device)
    best_chart_frame = torch.full(
        (chart_count,), 65535, device=device, dtype=torch.int32
    )
    visible_frames = [set() for _ in range(runtime.nonbody_piece_count)]
    training_opportunity_flat = torch.zeros(
        (total_texels,), device=device, dtype=torch.bool
    )
    holdout_opportunity_flat = torch.zeros_like(training_opportunity_flat)
    erosion_by_piece = torch.zeros(
        (runtime.nonbody_piece_count, 3), device=device, dtype=torch.int64
    )
    samples_per_frame: dict[str, int] = {}
    glass_proxy_per_frame: dict[str, int] = {}
    zbuffer_samples_per_frame: dict[str, dict[str, int]] = {}
    raw_statistics = {}
    body_visibility_audit: dict[str, dict[str, int]] = {}
    training_piece_owner_pixel_audit: dict[str, list[int]] = {}
    holdout_piece_owner_pixel_audit: dict[str, list[int]] = {}
    training_projection_gate_audit = _new_projection_gate_audit(
        torch, runtime.nonbody_piece_count, device
    )
    holdout_projection_gate_audit = _new_projection_gate_audit(
        torch, runtime.nonbody_piece_count, device
    )

    for position, frame_id in enumerate(train_ids, start=1):
        target = legacy._load_rgb(PIL_Image, np, torch, targets[frame_id], device)
        frame_chart_sum = torch.zeros((chart_count,), device=device)
        frame_chart_samples = torch.zeros((chart_count,), device=device)
        sample_count = 0
        glass_count = 0
        full_z_count = 0
        piece_self_z_count = 0
        color_sum = torch.zeros(3, device=device)
        color_square_sum = torch.zeros(3, device=device)
        color_count = 0
        seen_pieces: set[int] = set()
        candidates = _frame_piece_candidates(
            np,
            torch,
            F,
            dr,
            ctx,
            vertices,
            faces,
            face_object,
            face_piece,
            page_normals,
            transparent,
            runtime.visibility_piece_count,
            piece_target_full_z,
            surface,
            occupied_indices,
            torch.from_numpy(runtime.source.K[camera_lookup[frame_id]]).to(device),
            alignment.frames[frame_id],
            target,
            runtime.source.width,
            runtime.source.height,
            bake_contract.SEEDVR_RGB_WIDTH,
            bake_contract.SEEDVR_RGB_HEIGHT,
            width,
            height,
            visibility_audit=body_visibility_audit,
            visibility_audit_key=f"train:{int(frame_id)}",
            piece_owner_pixel_audit=training_piece_owner_pixel_audit,
            opportunity_mask_flat=training_opportunity_flat,
            projection_gate_audit=training_projection_gate_audit,
        )
        for candidate in candidates:
            atlas_index = candidate["atlas_index"]
            score = candidate["score"]
            previous_best = best_score.index_select(0, atlas_index)
            previous_second = second_score.index_select(0, atlas_index)
            winner = score > previous_best
            second_score[atlas_index] = torch.where(
                winner, previous_best, torch.maximum(previous_second, score)
            )
            winning_index = atlas_index[winner]
            if len(winning_index):
                best_score[winning_index] = score[winner]
                best_color[winning_index] = candidate["color"][winner]
                best_frame[winning_index] = int(frame_id)
                best_evidence_class[winning_index] = candidate[
                    "video_evidence_class"
                ][winner]
            support[atlas_index] += 1
            chart_id = candidate["chart_id"]
            frame_chart_sum.scatter_add_(0, chart_id, score)
            frame_chart_samples.scatter_add_(0, chart_id, torch.ones_like(score))
            seen_pieces.update(
                int(value) for value in torch.unique(candidate["piece_id"]).cpu().tolist()
            )
            radius = candidate["interior_radius"].long()
            if bool(((radius < 0) | (radius > 2)).any()):
                raise ValueError("Accepted candidate has invalid piece erosion radius")
            index = candidate["piece_id"].long() * 3 + radius
            erosion_by_piece += torch.bincount(
                index, minlength=runtime.nonbody_piece_count * 3
            )[: runtime.nonbody_piece_count * 3].reshape(runtime.nonbody_piece_count, 3)
            sample_count += int(len(atlas_index))
            glass_count += int(candidate["glass_warp_proxy"].sum().item())
            full_z_count += int(candidate["target_used_full_z"].sum().item())
            piece_self_z_count += int(
                candidate["target_used_piece_self_z"].sum().item()
            )
            color_sum += candidate["color"].sum(dim=0)
            color_square_sum += candidate["color"].square().sum(dim=0)
            color_count += int(len(atlas_index))
        observed_chart = frame_chart_samples > 0
        chart_coverage = frame_chart_samples / chart_occupancy.clamp_min(1)
        chart_score = (
            frame_chart_sum / frame_chart_samples.clamp_min(1)
        ) * chart_coverage.sqrt()
        chart_winner = observed_chart & (chart_score > best_chart_score)
        best_chart_score[chart_winner] = chart_score[chart_winner]
        best_chart_frame[chart_winner] = int(frame_id)
        samples_per_frame[str(frame_id)] = sample_count
        glass_proxy_per_frame[str(frame_id)] = glass_count
        zbuffer_samples_per_frame[str(frame_id)] = {
            "full_z": full_z_count,
            "piece_self_z": piece_self_z_count,
            "opaque_z": sample_count - full_z_count - piece_self_z_count,
        }
        if color_count:
            mean = color_sum / color_count
            deviation = (
                color_square_sum / color_count - mean.square()
            ).clamp_min(1e-6).sqrt()
        else:
            mean = torch.full((3,), 0.5, device=device)
            deviation = torch.full((3,), 0.2, device=device)
        raw_statistics[frame_id] = (mean, deviation)
        for piece_index in seen_pieces:
            visible_frames[piece_index].add(int(frame_id))
        del target, candidates, frame_chart_sum, frame_chart_samples, chart_score
        if position == 1 or position % 10 == 0 or position == len(train_ids):
            _log(
                f"{page.page_id} pass 1: {position}/{len(train_ids)}; "
                f"latest={sample_count} piece-owned texels"
            )

    means = torch.stack([raw_statistics[frame_id][0] for frame_id in train_ids])
    deviations = torch.stack([raw_statistics[frame_id][1] for frame_id in train_ids])
    reference_mean = means.median(dim=0).values
    reference_deviation = deviations.median(dim=0).values
    leveling = {}
    for frame_id in train_ids:
        mean, deviation = raw_statistics[frame_id]
        gain = (reference_deviation / deviation.clamp_min(1e-3)).clamp(0.95, 1.05)
        bias = (reference_mean - gain * mean).clamp(-0.02, 0.02)
        leveling[frame_id] = (gain, bias)

    coherent_color = torch.zeros_like(best_color)
    coherent_score = torch.full_like(best_score, -1.0)
    coherent_frame = torch.full_like(best_frame, 65535)
    coherent_evidence_class = torch.zeros_like(best_evidence_class)
    for position, frame_id in enumerate(train_ids, start=1):
        assigned_indices = occupied_indices[
            best_chart_frame.index_select(0, occupied_chart) == int(frame_id)
        ]
        if not len(assigned_indices):
            continue
        target = legacy._load_rgb(PIL_Image, np, torch, targets[frame_id], device)
        candidates = _frame_piece_candidates(
            np,
            torch,
            F,
            dr,
            ctx,
            vertices,
            faces,
            face_object,
            face_piece,
            page_normals,
            transparent,
            runtime.visibility_piece_count,
            piece_target_full_z,
            surface,
            assigned_indices,
            torch.from_numpy(runtime.source.K[camera_lookup[frame_id]]).to(device),
            alignment.frames[frame_id],
            target,
            runtime.source.width,
            runtime.source.height,
            bake_contract.SEEDVR_RGB_WIDTH,
            bake_contract.SEEDVR_RGB_HEIGHT,
            width,
            height,
        )
        for candidate in candidates:
            gain, bias = leveling[frame_id]
            color = (candidate["color"] * gain + bias).clamp(0, 1)
            assigned = (
                best_chart_frame.index_select(0, candidate["chart_id"])
                == int(frame_id)
            )
            atlas_index = candidate["atlas_index"][assigned]
            if len(atlas_index):
                coherent_color[atlas_index] = color[assigned]
                coherent_score[atlas_index] = candidate["score"][assigned]
                coherent_frame[atlas_index] = int(frame_id)
                coherent_evidence_class[atlas_index] = candidate[
                    "video_evidence_class"
                ][assigned]
        del target, candidates
        if position == 1 or position % 10 == 0 or position == len(train_ids):
            _log(f"{page.page_id} pass 2: {position}/{len(train_ids)}")

    per_texel_known = (best_score > 0) & occupied_flat
    coherent_known = (coherent_score > 0) & occupied_flat
    texel_fallback = per_texel_known & ~coherent_known
    for frame_id in train_ids:
        selected = texel_fallback & (best_frame == int(frame_id))
        if bool(selected.any()):
            gain, bias = leveling[frame_id]
            best_color[selected] = (best_color[selected] * gain + bias).clamp(0, 1)
    color_flat = torch.where(coherent_known[:, None], coherent_color, best_color)
    score_flat = torch.where(coherent_known, coherent_score, best_score)
    frame_flat = torch.where(coherent_known, coherent_frame, best_frame)
    evidence_class_flat = torch.where(
        coherent_known, coherent_evidence_class, best_evidence_class
    )
    known_flat = coherent_known | per_texel_known
    if bool((known_flat != (evidence_class_flat > 0)).any()):
        raise ValueError("Video evidence class does not exactly cover observed texels")
    color = color_flat.reshape(atlas_size, atlas_size, 3)
    known = known_flat.reshape(atlas_size, atlas_size)
    score_image = score_flat.clamp_min(0).reshape(atlas_size, atlas_size)
    frame_image = frame_flat.reshape(atlas_size, atlas_size)
    evidence_class_image = evidence_class_flat.reshape(atlas_size, atlas_size)
    steering_piece_index = DENSE_OBJECT_SURROGATE_PIECE_INDEX["39:-1"]
    if bool(
        (
            (evidence_class_image == 2)
            & (atlas_piece != steering_piece_index)
        ).any()
    ) or bool(
        (
            (evidence_class_image == 3)
            & (atlas_piece == steering_piece_index)
        ).any()
    ):
        raise ValueError("Video evidence winner class violates steering surrogate policy")
    texel_fallback_image = texel_fallback.reshape(atlas_size, atlas_size)
    support_image = support.float().reshape(atlas_size, atlas_size)
    positive_support = support_image[support_image > 0]
    support_scale = (
        torch.quantile(
            positive_support[:: max(1, positive_support.numel() // 1_000_000)],
            0.95,
        )
        if positive_support.numel()
        else torch.tensor(1.0, device=device)
    ).clamp_min(1)
    support_display = torch.log1p(support_image) / torch.log1p(support_scale)
    color, generated, piece_fill_modes = _fill_by_piece_chart(
        torch,
        F,
        color,
        known,
        atlas_piece,
        surface["chart"],
        runtime.piece_material,
        runtime.nonbody_piece_count,
    )
    padded_color, padded, gutter_domain, chart_slots = _pad_gutters_by_piece_chart(
        torch,
        color,
        occupied,
        surface["piece"],
        surface["chart"],
        atlas_padding,
    )
    if bool((gutter_domain[occupied] < 0).any()):
        raise ValueError("An occupied texel lost its piece/chart gutter owner")

    texture_path = texture_dir / f"{page.page_id}.png"
    legacy._save_rgb(PIL_Image, torch, padded_color, texture_path)
    legacy._save_gray(PIL_Image, torch, occupied.float(), diagnostic_dir / "occupied_mask.png")
    legacy._save_gray(PIL_Image, torch, known.float(), diagnostic_dir / "observed_mask.png")
    legacy._save_gray(PIL_Image, torch, generated.float(), diagnostic_dir / "generated_fill_mask.png")
    legacy._save_gray(PIL_Image, torch, texel_fallback_image.float(), diagnostic_dir / "per_texel_fallback_mask.png")
    evidence_class_path = diagnostic_dir / "video_evidence_class_id.png"
    PIL_Image.fromarray(
        evidence_class_image.detach().cpu().numpy()[::-1].astype(np.uint8),
        mode="L",
    ).save(evidence_class_path)
    legacy._save_gray(PIL_Image, torch, score_image, diagnostic_dir / "alignment_confidence.png")
    legacy._save_gray(PIL_Image, torch, support_display, diagnostic_dir / "view_support.png")
    legacy._save_u16(PIL_Image, torch, frame_image, diagnostic_dir / "source_frame_id.png")
    legacy._save_u16(
        PIL_Image,
        torch,
        torch.where(occupied, surface["piece"].to(torch.int32) + 1, torch.zeros_like(surface["piece"], dtype=torch.int32)),
        diagnostic_dir / "piece_owner_id.png",
    )
    legacy._save_u16(
        PIL_Image,
        torch,
        torch.where(
            occupied,
            surface["chart"].to(torch.int32) + 1,
            torch.zeros_like(surface["chart"], dtype=torch.int32),
        ),
        diagnostic_dir / "chart_owner_id.png",
    )

    holdout = _evaluate_page_holdouts(
        np,
        torch,
        F,
        dr,
        ctx,
        runtime,
        alignment,
        targets,
        holdout_ids,
        page,
        padded_color,
        surface,
        occupied_indices,
        vertices,
        faces,
        face_object,
        face_piece,
        piece_target_full_z,
        page_normals,
        transparent,
        camera_lookup,
        width,
        height,
        PIL_Image,
        body_visibility_audit,
        holdout_piece_owner_pixel_audit,
        holdout_opportunity_flat,
        holdout_projection_gate_audit,
    )

    training_opportunity = training_opportunity_flat.reshape(
        atlas_size, atlas_size
    )
    holdout_opportunity = holdout_opportunity_flat.reshape(atlas_size, atlas_size)
    training_opportunity_file = (
        diagnostic_dir / "training_visibility_opportunity_mask.png"
    )
    holdout_opportunity_file = (
        diagnostic_dir / "holdout_visibility_opportunity_mask.png"
    )
    legacy._save_gray(
        PIL_Image, torch, training_opportunity.float(), training_opportunity_file
    )
    legacy._save_gray(
        PIL_Image, torch, holdout_opportunity.float(), holdout_opportunity_file
    )

    def owner_pixels_by_frame(raw, prefix, frame_ids):
        output = {}
        for frame_id in frame_ids:
            audit_key = f"{prefix}:{int(frame_id)}"
            counts = raw.get(audit_key)
            if counts is None or len(counts) != runtime.visibility_piece_count:
                raise ValueError(
                    f"Missing piece-owner screen audit for {page.page_id}/{audit_key}"
                )
            output[str(int(frame_id))] = {
                runtime.piece_keys[int(piece_index)]: int(counts[int(piece_index)])
                for piece_index in page.piece_indices
            }
        return output

    training_owner_by_frame = owner_pixels_by_frame(
        training_piece_owner_pixel_audit, "train", train_ids
    )
    holdout_owner_by_frame = owner_pixels_by_frame(
        holdout_piece_owner_pixel_audit, "holdout", holdout_ids
    )

    piece_reports = []
    source_count_lookup = {
        int(piece_index): int(count)
        for piece_index, count in zip(
            page.piece_indices, page.piece_source_renderable_face_counts
        )
    }
    proxy_count_lookup = {
        int(piece_index): int(count)
        for piece_index, count in zip(page.piece_indices, page.piece_proxy_face_counts)
    }
    for piece_index in map(int, page.piece_indices):
        piece_key = runtime.piece_keys[piece_index]
        object_id = int(runtime.piece_object[piece_index])
        mask = atlas_piece == piece_index
        observed = known & mask
        occupied_count = int(mask.sum().item())
        observed_count = int(observed.sum().item())
        training_opportunity_count = int((training_opportunity & mask).sum().item())
        holdout_opportunity_count = int((holdout_opportunity & mask).sum().item())
        training_owner_pixel_count = sum(
            int(row[piece_key]) for row in training_owner_by_frame.values()
        )
        holdout_owner_pixel_count = sum(
            int(row[piece_key]) for row in holdout_owner_by_frame.values()
        )
        evidence = training_alignment_evidence[object_id]
        alignment_has_pixels = object_id in alignment_supported
        training_sufficient = (
            int(evidence["pixels"]) >= MIN_REQUIRED_TRAIN_ALIGNMENT_PIXELS
            and int(evidence["frames"]) >= MIN_REQUIRED_TRAIN_ALIGNMENT_FRAMES
        )
        neutral_allowed = observed_count == 0 and (
            (not alignment_has_pixels and object_id in EXPECTED_NO_ALIGNMENT_OBJECT_IDS)
            or (
                not training_sufficient
                and object_id in EXPECTED_INSUFFICIENT_MULTIVIEW_OBJECT_IDS
            )
        )
        integration_eligible = occupied_count > 0 and (
            observed_count > 0 or neutral_allowed
        )
        source_frames = sorted(
            int(value)
            for value in torch.unique(frame_image[observed]).detach().cpu().tolist()
        )
        radius_counts = erosion_by_piece[piece_index].detach().cpu().tolist()
        chart_fill = piece_fill_modes[piece_index]
        training_projection_audit = _projection_gate_audit_row(
            training_projection_gate_audit, piece_index
        )
        holdout_projection_audit = _projection_gate_audit_row(
            holdout_projection_gate_audit, piece_index
        )
        surrogate_policy = _global_warp_surrogate_policy_for_piece(piece_key)
        surrogate_accepted = int(
            training_projection_audit[
                "accepted_global_warp_surrogate_samples"
            ]
        )
        exact_accepted = int(
            training_projection_audit["accepted_exact_mapped_object_samples"]
        )
        piece_evidence_classes = evidence_class_image[mask]
        exact_winner_texels = int((piece_evidence_classes == 1).sum().item())
        surrogate_winner_texels = int((piece_evidence_classes == 2).sum().item())
        nonlocal_winner_texels = int((piece_evidence_classes == 3).sum().item())
        if (
            exact_winner_texels + surrogate_winner_texels + nonlocal_winner_texels
            != observed_count
        ):
            raise ValueError(f"Winner evidence classes do not cover {piece_key}")
        winner_texels = {
            "exact_dense_label": exact_winner_texels,
            "global_warp_coarse_label_surrogate": surrogate_winner_texels,
            "permitted_nonlocal_dense": nonlocal_winner_texels,
        }
        video_transfer_classification = _winner_evidence_classification(
            winner_texels
        )
        if video_transfer_classification == "invalid" or not (
            _winner_evidence_is_supported_by_projection(
                piece_key, winner_texels, training_projection_audit
            )
        ):
            raise ValueError(
                f"Winner evidence is unsupported by accepted samples for {piece_key}"
            )
        uses_piece_self_z = (
            piece_index in COUNTERFACTUAL_SELF_DEPTH_PIECE_INDICES
        )
        piece_reports.append(
            {
                "piece_index": piece_index,
                "piece_id": piece_key,
                "component_role": runtime.piece_roles[piece_index],
                "object_id": object_id,
                "object_name": runtime.source.object_names[object_id],
                "page_id": page.page_id,
                "material_template": runtime.piece_material[piece_index],
                "visibility_class": bake_contract.visibility_class_for_piece(
                    runtime.piece_ids[piece_index]
                ),
                "target_z_buffer": bake_contract.target_z_buffer_for_piece(
                    runtime.piece_ids[piece_index]
                ),
                "projection_visibility_buffer": (
                    "counterfactual_target_piece_self_z_against_opaque_z"
                    if uses_piece_self_z
                    else bake_contract.target_z_buffer_for_piece(
                        runtime.piece_ids[piece_index]
                    )
                ),
                "counterfactual_self_depth_policy": (
                    _counterfactual_self_depth_policy_for_piece(piece_key)
                    if uses_piece_self_z
                    else None
                ),
                "source_renderable_faces": source_count_lookup[piece_index],
                "raster_triangles": proxy_count_lookup[piece_index],
                "occupied_texels": occupied_count,
                "direct_observed_texels": observed_count,
                "direct_coverage": observed_count / max(1, occupied_count),
                "alignment_has_parent_object_pixels": alignment_has_pixels,
                "training_alignment_pixel_count": int(evidence["pixels"]),
                "training_alignment_frame_count": int(evidence["frames"]),
                "training_alignment_sufficient_for_required_texture": training_sufficient,
                "special_evidence_policy": SPECIAL_EVIDENCE_POLICY.get(object_id),
                "visible_training_frames": sorted(visible_frames[piece_index]),
                "winning_source_frames": source_frames,
                "selected_training_screen_owner_pixels": training_owner_pixel_count,
                "selected_training_visibility_opportunity_texels": training_opportunity_count,
                "selected_holdout_screen_owner_pixels": holdout_owner_pixel_count,
                "selected_holdout_visibility_opportunity_texels": holdout_opportunity_count,
                "holdout_samples": int(
                    holdout["per_piece_sample_totals"].get(piece_key, 0)
                ),
                "accepted_projection_samples_by_piece_erosion_radius": {
                    str(radius): int(radius_counts[radius]) for radius in range(3)
                },
                "accepted_projection_samples": int(sum(radius_counts)),
                "projection_gate_audit": {
                    "training": training_projection_audit,
                    "holdout": holdout_projection_audit,
                },
                "video_transfer_evidence": {
                    "classification": video_transfer_classification,
                    "winner_class_mask_file": (
                        f"diagnostics/{page.page_id}/video_evidence_class_id.png"
                    ),
                    "winner_texels": winner_texels,
                    "direct_exact_mapped_object_samples": exact_accepted,
                    "global_warp_coarse_label_surrogate_samples": surrogate_accepted,
                    "holdout_global_warp_coarse_label_surrogate_samples": int(
                        holdout_projection_audit[
                            "accepted_global_warp_surrogate_samples"
                        ]
                    ),
                    "global_warp_coarse_label_surrogate_policy": surrogate_policy,
                },
                "fill_policy": "neutral" if observed_count == 0 else "video_multiscale",
                "fill_policy_detail": (
                    "multiscale_from_seedance_with_global_warp_coarse_label_surrogate"
                    if surrogate_winner_texels
                    else "multiscale_from_seedance"
                    if observed_count
                    else "neutral_invisible_or_insufficient_multiview"
                    if neutral_allowed
                    else "prohibited_supported_unobserved"
                ),
                "neutral_fill_allowed": neutral_allowed,
                "piece_chart_fill": chart_fill,
                "observed_chart_domain_ratio": (
                    chart_fill["observed_chart_domains"]
                    / max(1, chart_fill["chart_domains"])
                ),
                "integration_eligible": integration_eligible,
                "source_pbr_used": False,
            }
        )

    occupied_count = int(occupied.sum().item())
    observed_count = int((known & occupied).sum().item())
    page_report = {
        "page_id": page.page_id,
        "atlas_id": page.page_id,
        "texture_file": f"textures/{page.page_id}.png",
        "texture_sha256": legacy._sha256(texture_path),
        "size": [atlas_size, atlas_size],
        "canonical_size": [page.size, page.size],
        "padding_pixels": atlas_padding,
        "objects": list(map(int, page.object_ids)),
        "pieces": list(page.piece_keys),
        "source_faces": page.source_face_count,
        "source_degenerates_quarantined": page.source_degenerate_count,
        "raster_faces": page.triangle_count,
        "charts": page.chart_count,
        "source_chart_slots": page.source_chart_slots,
        "sidecar_sha256": list(page.sidecar_sha256),
        "occupied_texels": occupied_count,
        "direct_observed_texels": observed_count,
        "direct_coverage": observed_count / max(1, occupied_count),
        "generated_fill_texels": int(generated.sum().item()),
        "chart_coherent_observed_texels": int(coherent_known.sum().item()),
        "per_texel_fallback_texels": int(texel_fallback.sum().item()),
        "charts_with_winning_view": int((best_chart_frame != 65535).sum().item()),
        "samples_per_frame": samples_per_frame,
        "glass_warp_proxy_samples_per_frame": glass_proxy_per_frame,
        "zbuffer_samples_per_frame": zbuffer_samples_per_frame,
        "video_evidence_winner_classes": {
            "file": f"diagnostics/{page.page_id}/video_evidence_class_id.png",
            "native_png_mode": "L",
            "legend": {
                "0": "unobserved",
                "1": "exact_dense_label_winner",
                "2": "piece39_global_warp_coarse_label36_surrogate_winner",
                "3": "permitted_nonlocal_dense_winner",
            },
            "counts": {
                str(class_id): int((evidence_class_image == class_id).sum().item())
                for class_id in range(4)
            },
        },
        "selected_visibility": {
            "method": (
                "exact_piece_owner_screen_pixels_and_pre_dense_warp_uv_visibility;"
                "layered_glass_targets_use_counterfactual_piece_self_z_against_opaque_z"
            ),
            "training_opportunity_mask_file": (
                f"diagnostics/{page.page_id}/training_visibility_opportunity_mask.png"
            ),
            "holdout_opportunity_mask_file": (
                f"diagnostics/{page.page_id}/holdout_visibility_opportunity_mask.png"
            ),
            "training_opportunity_texels": int(training_opportunity.sum().item()),
            "holdout_opportunity_texels": int(holdout_opportunity.sum().item()),
            "training_screen_owner_pixels_by_frame": training_owner_by_frame,
            "holdout_screen_owner_pixels_by_frame": holdout_owner_by_frame,
        },
        "holdout": holdout,
        "body_visibility_raster": {
            "per_selected_view": body_visibility_audit,
            "views": len(body_visibility_audit),
            "all_views_have_full_z_body_pixels": all(
                row["body_full_z_raster_pixels"] > 0
                for row in body_visibility_audit.values()
            ),
            "all_views_have_opaque_z_body_pixels": all(
                row["body_opaque_z_raster_pixels"] > 0
                for row in body_visibility_audit.values()
            ),
            "minimum_full_z_body_pixels": min(
                (
                    row["body_full_z_raster_pixels"]
                    for row in body_visibility_audit.values()
                ),
                default=0,
            ),
            "minimum_opaque_z_body_pixels": min(
                (
                    row["body_opaque_z_raster_pixels"]
                    for row in body_visibility_audit.values()
                ),
                default=0,
            ),
        },
        "gutter": {
            "algorithm": "deterministic_nearest_piece_chart_domain_no_average",
            "piece_chart_domain_slots": int(chart_slots),
            "padding_pixels": atlas_padding,
            "padded_texels": int((padded & ~occupied).sum().item()),
            "cross_piece_or_chart_blends": 0,
        },
        "source_pbr_used": False,
    }
    del (
        page_normals,
        surface,
        occupied,
        atlas_piece,
        occupied_indices,
        best_color,
        best_score,
        second_score,
        best_frame,
        support,
        coherent_color,
        coherent_score,
        coherent_frame,
        color,
        padded_color,
        padded,
        gutter_domain,
        generated,
        training_opportunity_flat,
        holdout_opportunity_flat,
        training_opportunity,
        holdout_opportunity,
    )
    torch.cuda.empty_cache()
    return page_report, piece_reports


def _write_inventory(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "hashes.sha256"
    )
    lines = [
        f"{legacy._sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in paths
    ]
    (output_dir / "hashes.sha256").write_text("\n".join(lines) + "\n")


def _validate_zip_container(archive: Path, required_anchor: str) -> None:
    """Reject duplicate, unsafe or symlink members before extraction."""

    import stat
    import zipfile
    from pathlib import PurePosixPath

    with zipfile.ZipFile(archive) as handle:
        names: set[str] = set()
        anchors = 0
        for info in handle.infolist():
            raw = info.filename
            path = PurePosixPath(raw)
            if (
                not raw
                or raw.startswith("/")
                or "\\" in raw
                or any(part in ("", ".", "..") for part in path.parts)
            ):
                raise ValueError(f"Unsafe ZIP member {raw!r}")
            canonical = path.as_posix().rstrip("/")
            if canonical in names:
                raise ValueError(f"Duplicate ZIP member {canonical!r}")
            names.add(canonical)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"Symlink ZIP member forbidden: {canonical}")
            if not info.is_dir() and path.name == required_anchor:
                anchors += 1
        if anchors != 1:
            raise ValueError(
                f"ZIP must contain exactly one {required_anchor!r}, got {anchors}"
            )


def _verify_minimal_unwrap_payload(unwrap_root: Path) -> dict:
    """Verify exact canonical report+blend+50-sidecar archive inventory."""

    report_path = unwrap_root / "unwrap_pack_report.json"
    report = json.loads(report_path.read_text())
    declared_sidecars: list[str] = []
    for page in report.get("pages", []):
        for row in page.get("object_sidecars", []):
            filename = Path(str(row.get("path", ""))).name
            if not filename.endswith(".npz"):
                raise ValueError(f"Malformed unwrap sidecar name {filename!r}")
            declared_sidecars.append(f"sidecars/{filename}")
    if len(declared_sidecars) != bake_contract.EXPECTED_SIDECARS or len(
        set(declared_sidecars)
    ) != bake_contract.EXPECTED_SIDECARS:
        raise ValueError("Unwrap archive report must declare 50 unique sidecars")
    expected_payload = {
        "unwrap_pack_report.json",
        "v8_nonbody_uv_proxy.blend",
        *declared_sidecars,
    }
    inventory_path = unwrap_root / "hashes.sha256"
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise ValueError("Minimal unwrap archive has no regular hashes.sha256")
    actual_payload: set[str] = set()
    for path in unwrap_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink forbidden in unwrap payload: {path}")
        if path.is_file() and path != inventory_path:
            actual_payload.add(path.relative_to(unwrap_root).as_posix())
    if actual_payload != expected_payload:
        raise ValueError(
            "Unwrap archive is not the minimal canonical payload: "
            f"missing={sorted(expected_payload-actual_payload)}, "
            f"extra={sorted(actual_payload-expected_payload)}"
        )
    declared: set[str] = set()
    for raw in inventory_path.read_text().splitlines():
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("Malformed unwrap hashes.sha256 line")
        expected_sha, relative = parts
        relative = relative.lstrip("* ")
        candidate = (unwrap_root / relative).resolve()
        if (
            relative in declared
            or unwrap_root.resolve() not in candidate.parents
            or not candidate.is_file()
            or candidate.is_symlink()
        ):
            raise ValueError(f"Unsafe/duplicate unwrap inventory path {relative!r}")
        if legacy._sha256(candidate) != expected_sha.lower():
            raise ValueError(f"Unwrap inventory hash mismatch for {relative}")
        declared.add(relative)
    if declared != expected_payload:
        raise ValueError("Unwrap inventory does not exactly cover the minimal payload")
    return {
        "inventory_sha256": legacy._sha256(inventory_path),
        "verified_payload_files": len(declared),
        "sidecar_files": sorted(declared_sidecars),
    }


def _runtime_sidecar_hashes(runtime: RuntimeV8Contract) -> dict[str, str]:
    rows = {}
    for page in runtime.pages:
        for path, digest in zip(page.sidecar_paths, page.sidecar_sha256):
            rows[f"sidecars/{path.name}"] = digest
    return dict(sorted(rows.items()))


def _numpy_shift(np, value, dy: int, dx: int, fill):
    output = np.full(value.shape, fill, dtype=value.dtype)
    height, width = value.shape[:2]
    source_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_y = slice(max(0, dy), min(height, height + dy))
    target_x = slice(max(0, dx), min(width, width + dx))
    output[target_y, target_x] = value[source_y, source_x]
    return output


def _recompute_page_artifact_evidence(
    report: dict,
    proof_root: Path,
    smoke_size: int,
    piece_rows_by_key: dict[str, dict],
) -> dict:
    """Decode raw PNG/masks and recompute page/piece/gutter evidence."""

    import numpy as np
    from PIL import Image as PIL_Image

    slug = str(report.get("page_id", ""))
    expected_texture_relative = f"textures/{slug}.png"
    if report.get("texture_file") != expected_texture_relative:
        raise ValueError(f"V8 smoke proof texture route drift for {slug}")

    def load_png(relative: str, expected_mode: str):
        path = (proof_root / relative).resolve()
        if proof_root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"V8 smoke proof PNG missing/unsafe: {relative}")
        try:
            with PIL_Image.open(path) as image:
                if (
                    image.format != "PNG"
                    or image.size != (smoke_size, smoke_size)
                    or image.mode != expected_mode
                ):
                    raise ValueError("format/dimensions/native-mode")
                # All runner diagnostics use legacy._save_* which writes
                # ``array[::-1]``. Restore the pre-save orientation before
                # recomputing the directional gutter tie-break.
                array = np.asarray(image)[::-1].copy()
        except Exception as error:
            raise ValueError(f"V8 smoke proof PNG invalid: {relative}: {error}") from error
        return path, array

    texture_path, texture = load_png(expected_texture_relative, "RGB")
    if report.get("texture_sha256") != legacy._sha256(texture_path):
        raise ValueError(f"V8 smoke proof texture SHA does not match bytes for {slug}")
    diagnostic_root = f"diagnostics/{slug}"
    _path, occupied_raw = load_png(f"{diagnostic_root}/occupied_mask.png", "L")
    _path, observed_raw = load_png(f"{diagnostic_root}/observed_mask.png", "L")
    _path, generated_raw = load_png(f"{diagnostic_root}/generated_fill_mask.png", "L")
    _path, fallback_raw = load_png(f"{diagnostic_root}/per_texel_fallback_mask.png", "L")
    _path, piece_owner = load_png(f"{diagnostic_root}/piece_owner_id.png", "I;16")
    _path, chart_owner = load_png(f"{diagnostic_root}/chart_owner_id.png", "I;16")
    _path, source_frame = load_png(f"{diagnostic_root}/source_frame_id.png", "I;16")
    _path, _confidence = load_png(f"{diagnostic_root}/alignment_confidence.png", "L")
    _path, _support = load_png(f"{diagnostic_root}/view_support.png", "L")
    evidence_metadata = report.get("video_evidence_winner_classes", {})
    expected_evidence_file = f"{diagnostic_root}/video_evidence_class_id.png"
    if evidence_metadata.get("file") != expected_evidence_file:
        raise ValueError(f"V8 smoke proof evidence-class route drift for {slug}")
    _path, evidence_class = load_png(expected_evidence_file, "L")
    selected_visibility = report.get("selected_visibility", {})
    expected_training_opportunity_file = (
        f"{diagnostic_root}/training_visibility_opportunity_mask.png"
    )
    expected_holdout_opportunity_file = (
        f"{diagnostic_root}/holdout_visibility_opportunity_mask.png"
    )
    if (
        selected_visibility.get("training_opportunity_mask_file")
        != expected_training_opportunity_file
        or selected_visibility.get("holdout_opportunity_mask_file")
        != expected_holdout_opportunity_file
    ):
        raise ValueError(f"V8 smoke proof opportunity mask route drift for {slug}")
    _path, training_opportunity_raw = load_png(
        expected_training_opportunity_file, "L"
    )
    _path, holdout_opportunity_raw = load_png(
        expected_holdout_opportunity_file, "L"
    )
    piece_owner = np.asarray(piece_owner, dtype=np.int64)
    chart_owner = np.asarray(chart_owner, dtype=np.int64)
    source_frame = np.asarray(source_frame, dtype=np.int64)
    if any(array.ndim != 2 for array in (piece_owner, chart_owner, source_frame)):
        raise ValueError(f"V8 smoke proof owner/frame mask is not scalar for {slug}")
    for name, raw in (
        ("occupied", occupied_raw),
        ("observed", observed_raw),
        ("generated", generated_raw),
        ("fallback", fallback_raw),
        ("training opportunity", training_opportunity_raw),
        ("holdout opportunity", holdout_opportunity_raw),
    ):
        if set(np.unique(raw).tolist()) - {0, 255}:
            raise ValueError(f"V8 smoke proof {name} mask is not binary for {slug}")
    if set(np.unique(evidence_class).tolist()) - {0, 1, 2, 3}:
        raise ValueError(f"V8 smoke proof evidence class is not categorical for {slug}")
    occupied = occupied_raw > 0
    observed = observed_raw > 0
    generated = generated_raw > 0
    fallback = fallback_raw > 0
    training_opportunity = training_opportunity_raw > 0
    holdout_opportunity = holdout_opportunity_raw > 0
    if (
        np.any(observed & ~occupied)
        or np.any(generated & ~occupied)
        or np.any(fallback & ~observed)
        or np.any(training_opportunity & ~occupied)
        or np.any(holdout_opportunity & ~occupied)
        or np.any(observed & ~training_opportunity)
        or np.any(observed & generated)
        or not np.array_equal(occupied, observed | generated)
        or not np.array_equal(evidence_class > 0, observed)
    ):
        raise ValueError(f"V8 smoke proof mask set relations failed for {slug}")
    if not np.array_equal(piece_owner > 0, occupied):
        raise ValueError(f"V8 smoke proof piece owner/occupancy relation failed for {slug}")
    if not np.array_equal(chart_owner > 0, occupied):
        raise ValueError(f"V8 smoke proof chart owner/occupancy relation failed for {slug}")
    evidence_counts = {
        str(class_id): int(np.count_nonzero(evidence_class == class_id))
        for class_id in range(4)
    }
    if (
        evidence_metadata.get("native_png_mode") != "L"
        or evidence_metadata.get("legend")
        != {
            "0": "unobserved",
            "1": "exact_dense_label_winner",
            "2": "piece39_global_warp_coarse_label36_surrogate_winner",
            "3": "permitted_nonlocal_dense_winner",
        }
        or {
            str(key): int(value)
            for key, value in evidence_metadata.get("counts", {}).items()
        }
        != evidence_counts
    ):
        raise ValueError(f"V8 smoke proof evidence-class metadata drift for {slug}")
    occupied_count = int(occupied.sum())
    observed_count = int(observed.sum())
    generated_count = int(generated.sum())
    fallback_count = int(fallback.sum())
    training_opportunity_count = int(training_opportunity.sum())
    holdout_opportunity_count = int(holdout_opportunity.sum())
    if (
        int(report.get("occupied_texels", -1)) != occupied_count
        or int(report.get("direct_observed_texels", -1)) != observed_count
        or int(report.get("generated_fill_texels", -1)) != generated_count
        or int(report.get("per_texel_fallback_texels", -1)) != fallback_count
    ):
        raise ValueError(f"V8 smoke proof page mask counts disagree with PNG for {slug}")
    if (
        int(selected_visibility.get("training_opportunity_texels", -1))
        != training_opportunity_count
        or int(selected_visibility.get("holdout_opportunity_texels", -1))
        != holdout_opportunity_count
    ):
        raise ValueError(
            f"V8 smoke proof page opportunity counts disagree with PNG for {slug}"
        )
    expected_page_coverage = observed_count / max(1, occupied_count)
    if abs(float(report.get("direct_coverage", -1.0)) - expected_page_coverage) > 1e-12:
        raise ValueError(f"V8 smoke proof page direct coverage disagrees with PNG for {slug}")

    if slug not in EXPECTED_TEXTURES:
        raise ValueError(f"V8 smoke proof has unknown page slug {slug!r}")
    expected_piece_keys = [
        bake_contract.piece_key(piece)
        for piece in bake_contract.EXPECTED_PIECE_IDS
        if bake_contract.page_for_object(piece[0]).page_id == slug
    ]
    if list(map(str, report.get("pieces", []))) != expected_piece_keys:
        raise ValueError(f"V8 smoke proof canonical page piece list failed for {slug}")
    expected_piece_indices = {
        bake_contract.piece_key(piece): index
        for index, piece in enumerate(bake_contract.EXPECTED_PIECE_IDS)
    }
    allowed_owner_values = {
        expected_piece_indices[key] + 1 for key in expected_piece_keys
    }
    actual_owner_values = set(np.unique(piece_owner[occupied]).tolist())
    if actual_owner_values != allowed_owner_values:
        raise ValueError(f"V8 smoke proof page piece-owner inventory failed for {slug}")
    steering_owner_value = expected_piece_indices["39:-1"] + 1
    if np.any((evidence_class == 2) & (piece_owner != steering_owner_value)) or np.any(
        (evidence_class == 3) & (piece_owner == steering_owner_value)
    ):
        raise ValueError(f"V8 smoke proof surrogate winner owner failed for {slug}")
    for chart_value in np.unique(chart_owner[occupied]).tolist():
        chart_piece_owners = set(
            np.unique(piece_owner[occupied & (chart_owner == chart_value)]).tolist()
        )
        if len(chart_piece_owners) != 1:
            raise ValueError(
                f"V8 smoke proof chart {chart_value} crosses piece owners on {slug}"
            )
    gutter = report.get("gutter", {})
    canonical_chart_slots = int(
        bake_contract.EXPECTED_RENDERABLE_CHARTS_BY_PAGE[slug]
    )
    chart_slots = int(gutter.get("piece_chart_domain_slots", 0))
    actual_chart_values = set(np.unique(chart_owner[occupied]).tolist())
    if (
        int(report.get("charts", -1)) != canonical_chart_slots
        or not actual_chart_values
        or chart_slots != max(actual_chart_values)
        or not actual_chart_values.issubset(
            set(range(1, canonical_chart_slots + 1))
        )
    ):
        raise ValueError(f"V8 smoke proof chart owner range failed for {slug}")
    for key in expected_piece_keys:
        row = piece_rows_by_key.get(key)
        if row is None:
            raise ValueError(f"V8 smoke proof page references missing piece {key}")
        owner_value = expected_piece_indices[key] + 1
        piece_mask = piece_owner == owner_value
        piece_observed = piece_mask & observed
        piece_training_opportunity = piece_mask & training_opportunity
        piece_holdout_opportunity = piece_mask & holdout_opportunity
        piece_occupied_count = int(piece_mask.sum())
        piece_observed_count = int(piece_observed.sum())
        piece_evidence_counts = {
            "exact_dense_label": int(
                np.count_nonzero(piece_mask & (evidence_class == 1))
            ),
            "global_warp_coarse_label_surrogate": int(
                np.count_nonzero(piece_mask & (evidence_class == 2))
            ),
            "permitted_nonlocal_dense": int(
                np.count_nonzero(piece_mask & (evidence_class == 3))
            ),
        }
        if (
            int(row.get("occupied_texels", -1)) != piece_occupied_count
            or int(row.get("direct_observed_texels", -1)) != piece_observed_count
            or abs(
                float(row.get("direct_coverage", -1.0))
                - piece_observed_count / max(1, piece_occupied_count)
            )
            > 1e-12
        ):
            raise ValueError(f"V8 smoke proof piece counts disagree with PNG for {key}")
        transfer = row.get("video_transfer_evidence", {})
        if (
            transfer.get("winner_class_mask_file") != expected_evidence_file
            or {
                str(name): int(value)
                for name, value in transfer.get("winner_texels", {}).items()
            }
            != piece_evidence_counts
            or sum(piece_evidence_counts.values()) != piece_observed_count
        ):
            raise ValueError(
                f"V8 smoke proof piece winner evidence disagrees with PNG for {key}"
            )
        if (
            int(
                row.get(
                    "selected_training_visibility_opportunity_texels", -1
                )
            )
            != int(piece_training_opportunity.sum())
            or int(
                row.get("selected_holdout_visibility_opportunity_texels", -1)
            )
            != int(piece_holdout_opportunity.sum())
        ):
            raise ValueError(
                f"V8 smoke proof piece opportunity counts disagree with PNG for {key}"
            )
        chart_domains = set(np.unique(chart_owner[piece_mask]).tolist())
        observed_chart_domains = set(
            np.unique(chart_owner[piece_observed]).tolist()
        )
        fill = row.get("piece_chart_fill", {})
        if (
            int(fill.get("chart_domains", -1)) != len(chart_domains)
            or int(fill.get("observed_chart_domains", -1))
            != len(observed_chart_domains)
            or int(fill.get("neutral_chart_domains", -1))
            != len(chart_domains - observed_chart_domains)
            or abs(
                float(row.get("observed_chart_domain_ratio", -1.0))
                - len(observed_chart_domains) / max(1, len(chart_domains))
            )
            > 1e-12
        ):
            raise ValueError(f"V8 smoke proof piece chart counts disagree with PNG for {key}")
        winning_frames = sorted(
            map(int, np.unique(source_frame[piece_observed]).tolist())
        )
        if winning_frames != list(map(int, row.get("winning_source_frames", []))):
            raise ValueError(f"V8 smoke proof winning frame IDs disagree with PNG for {key}")

    # Re-run the exact deterministic gutter propagation on quantized occupied
    # RGB. This verifies both owner-domain propagation and final gutter pixels.
    domain = (piece_owner - 1) * chart_slots + (chart_owner - 1)
    owner = np.where(occupied, domain, -1).astype(np.int64)
    value = np.where(occupied[..., None], texture, np.zeros_like(texture))
    directions = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )
    max_owner = np.iinfo(np.int64).max
    padding = int(report.get("padding_pixels", -1))
    if (
        padding != SMOKE_PADDING
        or int(gutter.get("padding_pixels", -1)) != SMOKE_PADDING
        or gutter.get("algorithm")
        != "deterministic_nearest_piece_chart_domain_no_average"
        or int(gutter.get("cross_piece_or_chart_blends", -1)) != 0
    ):
        raise ValueError(f"V8 smoke proof canonical gutter contract failed for {slug}")
    for _ in range(padding):
        empty = owner < 0
        best = np.full(owner.shape, max_owner, dtype=np.int64)
        best_color = np.zeros_like(value)
        for dy, dx in directions:
            candidate_owner = _numpy_shift(np, owner, dy, dx, -1)
            choose = empty & (candidate_owner >= 0) & (candidate_owner < best)
            best = np.where(choose, candidate_owner, best)
            candidate_color = _numpy_shift(np, value, dy, dx, 0)
            best_color = np.where(choose[..., None], candidate_color, best_color)
        frontier = best != max_owner
        if not np.any(frontier):
            break
        owner = np.where(frontier, best, owner)
        value = np.where(frontier[..., None], best_color, value)
    padded = owner >= 0
    expected_padded_texels = int(np.count_nonzero(padded & ~occupied))
    if int(gutter.get("padded_texels", -1)) != expected_padded_texels:
        raise ValueError(f"V8 smoke proof gutter count disagrees with recompute for {slug}")
    if not np.array_equal(texture, value):
        raise ValueError(f"V8 smoke proof RGB gutters disagree with deterministic recompute for {slug}")
    return {
        "occupied_texels": occupied_count,
        "direct_observed_texels": observed_count,
        "generated_fill_texels": generated_count,
        "per_texel_fallback_texels": fallback_count,
        "training_visibility_opportunity_texels": training_opportunity_count,
        "holdout_visibility_opportunity_texels": holdout_opportunity_count,
        "padded_texels": expected_padded_texels,
    }


def _verify_gpu_smoke_payload(report: dict, proof_root: Path, smoke_size: int) -> None:
    """Recompute structural/quality gates from a claimed GPU smoke archive."""

    if report.get("gpu_execution") is not True:
        raise ValueError("V8 smoke proof is not marked as GPU execution")
    runtime = report.get("runtime", {})
    if (
        not str(runtime.get("gpu", "")).strip()
        or not str(runtime.get("torch", "")).strip()
        or not str(runtime.get("torch_cuda", "")).strip()
        or float(runtime.get("elapsed_seconds", 0.0)) <= 0.0
        or float(runtime.get("peak_vram_allocated_gib", 0.0)) <= 0.0
        or float(runtime.get("peak_vram_reserved_gib", 0.0)) <= 0.0
    ):
        raise ValueError("V8 smoke proof has no credible GPU/CUDA runtime metrics")

    quality = report.get("quality_gate", {})
    checks = quality.get("checks", {})
    required_true = {
        "exact_six_page_ids_and_order",
        "exact_six_texture_files",
        "all_63_piece_ids_reported_once",
        "all_50_nonbody_parents_reported_once",
        "canonical_dataset_manifest_hash",
        "canonical_ferrari_body_geometry_hash",
        "canonical_seedance_video_hash",
        "canonical_alignment_archive_hash",
        "canonical_alignment_report_hash",
        "canonical_minimal_unwrap_archive_hash",
        "canonical_minimal_unwrap_inventory_hash",
        "body_is_actual_full_z_raster_geometry",
        "body_is_actual_opaque_z_raster_geometry",
        "body_raster_faces_have_exact_body_piece_owner",
        "pinned_seedvr_rgb_only_hash",
        "seedvr_rgb_only_exact_104_frame_identity_mapping",
        "seedvr_rgb_pixel_center_mapping_3x_plus_1",
        "native_720p_remains_geometry_camera_depth_owner_alignment_authority",
        "body_has_nonzero_full_and_opaque_raster_pixels_in_every_selected_view",
        "piece_chart_gutters_have_zero_cross_domain_blends",
        "exact_no_alignment_policy_object_21",
        "exact_insufficient_multiview_policy_objects_0_21_37",
        "all_63_pieces_have_uv_occupancy",
        "all_page_holdout_l1_below_fixed_threshold",
        "neutral_fill_never_authorized_outside_objects_0_21_37",
        "fill_is_strict_piece_chart_domain",
        "projection_gate_audits_are_piece_exact_and_funnel_consistent",
        "projection_visibility_buffer_matches_piece_contract",
        "global_warp_coarse_label_surrogate_exactly_whitelisted_39_to_36",
        "steering_dense_provenance_meets_exact_or_surrogate_floors",
        "emissive_20_and_38_share_shader_but_not_visibility",
        "cockpit_revealed_behind_glass_by_dense_warp_proxy",
        "cockpit_samples_use_opaque_z_behind_transmissive_exclusions",
        "layered_glass_exercises_piece_self_z_and_lenses_exercise_full_z",
        "coverage_aware_split_is_exact_48_51_swap_without_semantic_alias",
        "canonical_full_104_view_smoke",
        "smoke_split_83_21",
        "smoke_all_63_pieces_have_training_geometric_opportunity_or_exact_piece_na",
        "smoke_all_63_pieces_have_holdout_geometric_opportunity_or_exact_piece_na",
        "smoke_no_owner_visible_piece_lacks_uv_projection_opportunity",
        "smoke_no_colour_evidence_without_geometric_opportunity",
        "smoke_all_selected_visible_recoverable_pieces_have_video_transfer",
        "smoke_all_selected_visible_recoverable_pieces_have_holdout",
        "coverage_aware_split_minor_glass_15_0_has_exact_holdout",
        "frontmost_hidden_glass_is_exercised_by_counterfactual_piece_self_depth",
        "canonical_smoke_render_size_1280x720",
        "source_pbr_not_consumed",
    }
    false_or_missing = sorted(key for key in required_true if checks.get(key) is not True)
    if false_or_missing:
        raise ValueError(f"V8 smoke proof quality checks false/missing: {false_or_missing}")

    evidence_policy = report.get("alignment", {}).get("evidence_policy", {})
    if evidence_policy.get("no_alignment_object_ids") != list(
        EXPECTED_NO_ALIGNMENT_OBJECT_IDS
    ):
        raise ValueError("V8 smoke proof no-alignment exception list drift")
    if evidence_policy.get("insufficient_multiview_object_ids") != list(
        EXPECTED_INSUFFICIENT_MULTIVIEW_OBJECT_IDS
    ):
        raise ValueError("V8 smoke proof insufficient-multiview exception list drift")
    if evidence_policy.get("exact_low_evidence_piece_ids") != list(
        EXPECTED_INSUFFICIENT_MULTIVIEW_PIECE_IDS
    ):
        raise ValueError("V8 smoke proof low-evidence piece list drift")
    if evidence_policy.get("frontmost_hidden_glass_piece_ids") != list(
        EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS
    ):
        raise ValueError("V8 smoke proof layered-glass evidence list drift")
    if evidence_policy.get("dense_object_surrogate_policy") != {
        key: list(value) for key, value in DENSE_OBJECT_SURROGATE_POLICY.items()
    }:
        raise ValueError("V8 smoke proof dense-object surrogate policy drift")
    if evidence_policy.get("dense_object_surrogate_minimum_confidence") != (
        DENSE_OBJECT_SURROGATE_MIN_CONFIDENCE
    ):
        raise ValueError("V8 smoke proof dense-object surrogate confidence drift")

    thresholds = report.get("configuration", {}).get("quality_thresholds", {})
    expected_thresholds = {
        "minimum_direct_texels_per_recoverable_piece":
        MIN_SMOKE_DIRECT_TEXELS_PER_RECOVERABLE_PIECE,
        "minimum_direct_coverage_per_recoverable_piece":
        MIN_DIRECT_COVERAGE_PER_RECOVERABLE_PIECE,
        "minimum_observed_chart_domain_ratio": MIN_OBSERVED_CHART_DOMAIN_RATIO,
        "minimum_holdout_samples_per_recoverable_piece":
        MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE,
        "minimum_smoke_geometric_opportunity_texels_per_piece":
        MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE,
        "maximum_page_holdout_mean_l1": MAX_PAGE_HOLDOUT_MEAN_L1,
    }
    if thresholds != expected_thresholds:
        raise ValueError("V8 smoke proof quality threshold contract drift")
    if report.get("configuration", {}).get("render_size") != list(
        SMOKE_RENDER_SIZE
    ):
        raise ValueError("V8 smoke proof did not use canonical 1280x720 render size")
    configuration = report.get("configuration", {})
    if configuration.get("train_holdout_policy") != dict(
        bake_contract.V8_TRAIN_HOLDOUT_POLICY
    ):
        raise ValueError("V8 smoke proof train/holdout policy drift")
    selected_ids = list(map(int, configuration.get("selected_frame_ids", [])))
    expected_train_ids, expected_holdout_ids = _split_smoke_views(selected_ids)
    if list(map(int, configuration.get("training_frame_ids", []))) != list(
        expected_train_ids
    ) or list(map(int, configuration.get("holdout_frame_ids", []))) != list(
        expected_holdout_ids
    ):
        raise ValueError("V8 smoke proof training/holdout frame split drift")
    train_keys = {str(value) for value in expected_train_ids}
    holdout_keys = {str(value) for value in expected_holdout_ids}

    pieces = report.get("pieces", [])
    if [row.get("piece_id") for row in pieces] != [
        bake_contract.piece_key(piece) for piece in bake_contract.EXPECTED_PIECE_IDS
    ]:
        raise ValueError("V8 smoke proof does not contain the exact ordered 63 pieces")
    piece_rows_by_key = {str(row["piece_id"]): row for row in pieces}
    for piece_id, row in zip(bake_contract.EXPECTED_PIECE_IDS, pieces):
        object_id = int(piece_id[0])
        if (
            int(row.get("object_id", -1)) != object_id
            or row.get("page_id") != bake_contract.page_for_object(object_id).page_id
            or row.get("material_template")
            != bake_contract.material_template_for_object(object_id)
            or row.get("visibility_class")
            != bake_contract.visibility_class_for_piece(piece_id)
            or row.get("target_z_buffer")
            != bake_contract.target_z_buffer_for_piece(piece_id)
            or row.get("source_pbr_used") is not False
            or int(row.get("occupied_texels", 0)) <= 0
        ):
            raise ValueError(f"V8 smoke proof piece metadata/occupancy failed: {piece_id}")
        uses_piece_self_z = (
            bake_contract.piece_key(piece_id)
            in EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS
        )
        expected_self_policy = _counterfactual_self_depth_policy_for_piece(
            piece_id
        )
        if (
            row.get("projection_visibility_buffer")
            != (
                "counterfactual_target_piece_self_z_against_opaque_z"
                if uses_piece_self_z
                else bake_contract.target_z_buffer_for_piece(piece_id)
            )
            or row.get("counterfactual_self_depth_policy")
            != expected_self_policy
        ):
            raise ValueError(
                f"V8 smoke proof projection visibility policy drift: {piece_id}"
            )
        exception = _explicit_low_evidence_exception(row)
        fill = row.get("piece_chart_fill", {})
        domain_count = int(fill.get("chart_domains", 0))
        observed_domains = int(fill.get("observed_chart_domains", -1))
        neutral_domains = int(fill.get("neutral_chart_domains", -1))
        if domain_count <= 0 or observed_domains + neutral_domains != domain_count:
            raise ValueError(f"V8 smoke proof chart-domain inventory failed: {piece_id}")
        direct_ok = (
            int(row.get("direct_observed_texels", 0))
            >= MIN_SMOKE_DIRECT_TEXELS_PER_RECOVERABLE_PIECE
            and float(row.get("direct_coverage", 0.0))
            >= MIN_DIRECT_COVERAGE_PER_RECOVERABLE_PIECE
            and int(row.get("accepted_projection_samples", 0))
            >= MIN_SMOKE_DIRECT_TEXELS_PER_RECOVERABLE_PIECE
            and len(row.get("winning_source_frames", [])) > 0
            and float(row.get("observed_chart_domain_ratio", 0.0))
            >= MIN_OBSERVED_CHART_DOMAIN_RATIO
        )
        holdout_ok = int(row.get("holdout_samples", 0)) >= (
            MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE
        )
        training_owner = int(row.get("selected_training_screen_owner_pixels", -1))
        training_opportunity = int(
            row.get("selected_training_visibility_opportunity_texels", -1)
        )
        holdout_owner = int(row.get("selected_holdout_screen_owner_pixels", -1))
        holdout_opportunity = int(
            row.get("selected_holdout_visibility_opportunity_texels", -1)
        )
        training_geometry_na = _smoke_geometry_not_visible_exception(
            row, "training"
        )
        holdout_geometry_na = _smoke_geometry_not_visible_exception(row, "holdout")
        training_geometry_ok = (
            training_owner > 0
            and training_opportunity
            >= MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE
        )
        holdout_geometry_ok = (
            holdout_owner > 0
            and holdout_opportunity
            >= MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE
        )
        if (
            (training_owner == 0) != (training_opportunity == 0)
            or (holdout_owner == 0) != (holdout_opportunity == 0)
            or not (training_geometry_ok or training_geometry_na)
            or not (holdout_geometry_ok or holdout_geometry_na)
        ):
            raise ValueError(
                f"V8 smoke proof geometric opportunity floor failed: {piece_id}"
            )
        if not (direct_ok or exception) or not (holdout_ok or exception):
            raise ValueError(f"V8 smoke proof piece coverage/holdout floor failed: {piece_id}")
        if not _steering_dense_provenance_floor(
            row, "training", MIN_SMOKE_DIRECT_TEXELS_PER_RECOVERABLE_PIECE
        ) or not _steering_dense_provenance_floor(
            row, "holdout", MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE
        ):
            raise ValueError(
                f"V8 smoke proof steering dense provenance floor failed: {piece_id}"
            )
        if (
            training_opportunity == 0
            and (
                int(row.get("direct_observed_texels", -1)) != 0
                or int(row.get("accepted_projection_samples", -1)) != 0
                or row.get("winning_source_frames") != []
            )
        ):
            raise ValueError(
                f"V8 smoke proof has direct evidence without training opportunity: {piece_id}"
            )
        if holdout_opportunity == 0 and int(row.get("holdout_samples", -1)) != 0:
            raise ValueError(
                f"V8 smoke proof has holdout samples without holdout opportunity: {piece_id}"
            )
        if row.get("neutral_fill_allowed") is True and not exception:
            raise ValueError(f"V8 smoke proof unauthorized neutral fill: {piece_id}")
        if not all(
            _projection_gate_audit_is_consistent(row, split)
            for split in ("training", "holdout")
        ):
            raise ValueError(
                f"V8 smoke proof projection gate audit drift: {piece_id}"
            )
        training_projection = row["projection_gate_audit"]["training"]
        holdout_projection = row["projection_gate_audit"]["holdout"]
        transfer = row.get("video_transfer_evidence", {})
        surrogate_accepted = int(
            training_projection["accepted_global_warp_surrogate_samples"]
        )
        exact_accepted = int(
            training_projection["accepted_exact_mapped_object_samples"]
        )
        winner_texels = transfer.get("winner_texels", {})
        expected_classification = _winner_evidence_classification(
            winner_texels
        )
        if (
            expected_classification == "invalid"
            or transfer.get("classification") != expected_classification
            or not _winner_evidence_is_supported_by_projection(
                bake_contract.piece_key(piece_id),
                winner_texels,
                training_projection,
            )
            or int(transfer.get("direct_exact_mapped_object_samples", -1))
            != exact_accepted
            or int(transfer.get("global_warp_coarse_label_surrogate_samples", -1))
            != surrogate_accepted
            or int(
                transfer.get(
                    "holdout_global_warp_coarse_label_surrogate_samples", -1
                )
            )
            != int(
                holdout_projection[
                    "accepted_global_warp_surrogate_samples"
                ]
            )
        ):
            raise ValueError(
                f"V8 smoke proof video-transfer classification drift: {piece_id}"
            )
        erosion = row.get(
            "accepted_projection_samples_by_piece_erosion_radius", {}
        )
        if set(erosion) != {"0", "1", "2"} or sum(
            int(value) for value in erosion.values()
        ) != int(row.get("accepted_projection_samples", -1)):
            raise ValueError(f"V8 smoke proof projection sample summary drift: {piece_id}")
        visible_frames = set(map(int, row.get("visible_training_frames", [])))
        winning_frames = set(map(int, row.get("winning_source_frames", [])))
        if not winning_frames.issubset(visible_frames) or not visible_frames.issubset(
            set(expected_train_ids)
        ):
            raise ValueError(f"V8 smoke proof training frame evidence drift: {piece_id}")

    pages = report.get("pages", [])
    if [row.get("page_id") for row in pages] != list(EXPECTED_TEXTURES):
        raise ValueError("V8 smoke proof page order drift")
    full_z_samples = 0
    piece_self_z_samples = 0
    opaque_z_samples = 0
    cockpit_glass_reveal_samples = 0
    cockpit_opaque_z_samples = 0
    glass_piece_self_z_samples = 0
    lens_full_z_samples = 0
    for slug, row in zip(EXPECTED_TEXTURES, pages):
        _recompute_page_artifact_evidence(
            row, proof_root, smoke_size, piece_rows_by_key
        )
        selected_visibility = row.get("selected_visibility", {})
        if selected_visibility.get("method") != (
            "exact_piece_owner_screen_pixels_and_pre_dense_warp_uv_visibility;"
            "layered_glass_targets_use_counterfactual_piece_self_z_against_opaque_z"
        ):
            raise ValueError(f"V8 smoke proof visibility method drift for {slug}")
        training_owner_by_frame = selected_visibility.get(
            "training_screen_owner_pixels_by_frame", {}
        )
        holdout_owner_by_frame = selected_visibility.get(
            "holdout_screen_owner_pixels_by_frame", {}
        )
        page_piece_keys = list(map(str, row.get("pieces", [])))
        if (
            set(training_owner_by_frame) != train_keys
            or set(holdout_owner_by_frame) != holdout_keys
        ):
            raise ValueError(
                f"V8 smoke proof visibility frame inventory drift for {slug}"
            )
        for frame_key, counts in (
            *training_owner_by_frame.items(),
            *holdout_owner_by_frame.items(),
        ):
            if set(counts) != set(page_piece_keys) or any(
                int(value) < 0 for value in counts.values()
            ):
                raise ValueError(
                    f"V8 smoke proof screen-owner inventory drift for {slug}/{frame_key}"
                )
        for piece_key in page_piece_keys:
            training_owner_total = sum(
                int(counts[piece_key])
                for counts in training_owner_by_frame.values()
            )
            holdout_owner_total = sum(
                int(counts[piece_key])
                for counts in holdout_owner_by_frame.values()
            )
            if (
                int(
                    piece_rows_by_key[piece_key].get(
                        "selected_training_screen_owner_pixels", -1
                    )
                )
                != training_owner_total
                or int(
                    piece_rows_by_key[piece_key].get(
                        "selected_holdout_screen_owner_pixels", -1
                    )
                )
                != holdout_owner_total
            ):
                raise ValueError(
                    f"V8 smoke proof screen-owner totals drift for {piece_key}"
                )
        samples_per_frame = row.get("samples_per_frame", {})
        glass_proxy_per_frame = row.get(
            "glass_warp_proxy_samples_per_frame", {}
        )
        zbuffer_per_frame = row.get("zbuffer_samples_per_frame", {})
        if (
            set(samples_per_frame) != train_keys
            or set(glass_proxy_per_frame) != train_keys
            or set(zbuffer_per_frame) != train_keys
        ):
            raise ValueError(f"V8 smoke proof training frame metric keys drift for {slug}")
        for frame_key in train_keys:
            total = int(samples_per_frame[frame_key])
            counts = zbuffer_per_frame[frame_key]
            if (
                total < 0
                or int(counts.get("full_z", -1)) < 0
                or int(counts.get("piece_self_z", -1)) < 0
                or int(counts.get("opaque_z", -1)) < 0
                or int(counts.get("full_z", 0))
                + int(counts.get("piece_self_z", 0))
                + int(counts.get("opaque_z", 0))
                != total
                or not 0 <= int(glass_proxy_per_frame[frame_key]) <= total
            ):
                raise ValueError(f"V8 smoke proof z/glass sample summary drift for {slug}/{frame_key}")

        holdout = row.get("holdout", {})
        if holdout.get("method") != (
            "exact_geometry_piece_owned_projection_with_reported_dense_label_provenance"
        ):
            raise ValueError(f"V8 smoke proof holdout method drift for {slug}")
        per_frame_l1 = list(holdout.get("per_frame_l1", []))
        per_frame_samples = list(map(int, holdout.get("per_frame_samples", [])))
        per_piece_by_frame = holdout.get("per_piece_samples_by_frame", {})
        per_piece_totals = holdout.get("per_piece_sample_totals", {})
        if (
            len(per_frame_l1) != len(expected_holdout_ids)
            or len(per_frame_samples) != len(expected_holdout_ids)
            or set(per_piece_by_frame) != holdout_keys
            or set(per_piece_totals) != set(row.get("pieces", []))
        ):
            raise ValueError(f"V8 smoke proof holdout inventory drift for {slug}")
        recomputed_piece_totals = {key: 0 for key in row.get("pieces", [])}
        valid_l1 = []
        for position, frame_id in enumerate(expected_holdout_ids):
            frame_key = str(frame_id)
            counts = per_piece_by_frame[frame_key]
            if set(counts) != set(recomputed_piece_totals):
                raise ValueError(f"V8 smoke proof holdout piece keys drift for {slug}/{frame_key}")
            sample_total = sum(int(value) for value in counts.values())
            if sample_total != per_frame_samples[position] or sample_total < 0:
                raise ValueError(f"V8 smoke proof holdout sample summary drift for {slug}/{frame_key}")
            l1 = per_frame_l1[position]
            if sample_total == 0:
                if l1 is not None:
                    raise ValueError(f"V8 smoke proof zero-sample holdout has L1 for {slug}/{frame_key}")
            else:
                if l1 is None or not 0.0 <= float(l1) <= 1.0:
                    raise ValueError(f"V8 smoke proof holdout L1 invalid for {slug}/{frame_key}")
                valid_l1.append(float(l1))
            for key, value in counts.items():
                recomputed_piece_totals[key] += int(value)
        if {
            key: int(value) for key, value in per_piece_totals.items()
        } != recomputed_piece_totals:
            raise ValueError(f"V8 smoke proof holdout piece totals drift for {slug}")
        recomputed_frames_with_samples = sum(value > 0 for value in per_frame_samples)
        recomputed_mean_l1 = (
            sum(valid_l1) / len(valid_l1) if valid_l1 else None
        )
        if (
            int(holdout.get("frames_with_samples", -1))
            != recomputed_frames_with_samples
            or (
                recomputed_mean_l1 is None
                and holdout.get("mean_aligned_l1") is not None
            )
            or (
                recomputed_mean_l1 is not None
                and abs(
                    float(holdout.get("mean_aligned_l1", -1.0))
                    - recomputed_mean_l1
                )
                > 1e-12
            )
        ):
            raise ValueError(f"V8 smoke proof holdout aggregate drift for {slug}")
        for key, expected_total in recomputed_piece_totals.items():
            if int(piece_rows_by_key[key].get("holdout_samples", -1)) != expected_total:
                raise ValueError(f"V8 smoke proof piece holdout total drift for {key}")

        holdout = row.get("holdout", {})
        mean_l1 = holdout.get("mean_aligned_l1")
        body = row.get("body_visibility_raster", {})
        gutter = row.get("gutter", {})
        body_per_view = body.get("per_selected_view", {})
        expected_body_keys = {
            *(f"train:{value}" for value in expected_train_ids),
            *(f"holdout:{value}" for value in expected_holdout_ids),
        }
        if set(body_per_view) != expected_body_keys:
            raise ValueError(f"V8 smoke proof BODY view keys drift for {slug}")
        full_body_pixels = [
            int(value.get("body_full_z_raster_pixels", -1))
            for value in body_per_view.values()
        ]
        opaque_body_pixels = [
            int(value.get("body_opaque_z_raster_pixels", -1))
            for value in body_per_view.values()
        ]
        if (
            int(body.get("views", -1)) != len(expected_body_keys)
            or body.get("all_views_have_full_z_body_pixels")
            != all(value > 0 for value in full_body_pixels)
            or body.get("all_views_have_opaque_z_body_pixels")
            != all(value > 0 for value in opaque_body_pixels)
            or int(body.get("minimum_full_z_body_pixels", -1))
            != min(full_body_pixels, default=0)
            or int(body.get("minimum_opaque_z_body_pixels", -1))
            != min(opaque_body_pixels, default=0)
        ):
            raise ValueError(f"V8 smoke proof BODY aggregate drift for {slug}")
        if (
            row.get("size") != [smoke_size, smoke_size]
            or int(row.get("occupied_texels", 0)) <= 0
            or int(row.get("direct_observed_texels", 0)) <= 0
            or int(holdout.get("frames_with_samples", 0)) <= 0
            or mean_l1 is None
            or float(mean_l1) > MAX_PAGE_HOLDOUT_MEAN_L1
            or gutter.get("cross_piece_or_chart_blends") != 0
            or body.get("all_views_have_full_z_body_pixels") is not True
            or body.get("all_views_have_opaque_z_body_pixels") is not True
            or int(body.get("minimum_full_z_body_pixels", 0)) <= 0
            or int(body.get("minimum_opaque_z_body_pixels", 0)) <= 0
        ):
            raise ValueError(f"V8 smoke proof page quality failed for {slug}")
        for counts in row.get("zbuffer_samples_per_frame", {}).values():
            full_z_samples += int(counts.get("full_z", 0))
            piece_self_z_samples += int(counts.get("piece_self_z", 0))
            opaque_z_samples += int(counts.get("opaque_z", 0))
            if slug == "cockpit":
                cockpit_opaque_z_samples += int(counts.get("opaque_z", 0))
            elif slug == "glass":
                glass_piece_self_z_samples += int(counts.get("piece_self_z", 0))
            elif slug == "lights_translucent":
                lens_full_z_samples += int(counts.get("full_z", 0))
        if slug == "cockpit":
            cockpit_glass_reveal_samples = sum(
                int(value)
                for value in row.get(
                    "glass_warp_proxy_samples_per_frame", {}
                ).values()
            )
    if piece_self_z_samples <= 0 or opaque_z_samples <= 0:
        raise ValueError(
            "V8 smoke proof did not exercise both piece-self-z and opaque-z samples"
        )
    if (
        cockpit_glass_reveal_samples <= 0
        or cockpit_opaque_z_samples <= 0
        or glass_piece_self_z_samples <= 0
        or lens_full_z_samples <= 0
    ):
        raise ValueError(
            "V8 smoke proof did not exercise layered-glass piece-self-z, cockpit opaque-z and lens full-z"
        )


def _verify_radiometry_proof_identity(actual, expected) -> None:
    if actual != expected:
        raise ValueError(
            "V8 smoke proof does not match the active pinned SeedVR RGB-only decode"
        )


def _verify_smoke_proof(
    smoke_proof_url: str,
    workspace: Path,
    *,
    dataset_archive_sha256: str,
    dataset_manifest_sha256: str,
    alignment_archive_sha256: str,
    alignment_report_sha256: str,
    unwrap_archive_sha256: str,
    unwrap_report_sha256: str,
    unwrap_blend_sha256: str,
    unwrap_inventory_sha256: str,
    unwrap_sidecar_sha256: dict[str, str],
    runner_sha256: str,
    implementation_identity: dict,
    expected_selected_ids: list[int],
    expected_radiometry_video: dict,
) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    proof_zip = workspace / "smoke_proof.zip"
    proof_extract = workspace / "smoke_proof"
    legacy._download(smoke_proof_url, proof_zip)
    _validate_zip_container(proof_zip, "multi_atlas_bake_report.json")
    legacy._safe_extract(proof_zip, proof_extract)
    proof_root = legacy._find_root(proof_extract, "multi_atlas_bake_report.json")
    inventory_path = proof_root / "hashes.sha256"
    if not inventory_path.is_file():
        raise ValueError("V8 smoke proof has no hashes.sha256 inventory")
    actual_files = {
        path.relative_to(proof_root).as_posix()
        for path in proof_root.rglob("*")
        if path.is_file() and path != inventory_path
    }
    declared: set[str] = set()
    for raw in inventory_path.read_text().splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        relative = relative.lstrip("* ")
        candidate = (proof_root / relative).resolve()
        if proof_root.resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError(f"Unsafe/missing v8 smoke proof path {relative}")
        if relative in declared or legacy._sha256(candidate) != expected.lower():
            raise ValueError(f"V8 smoke proof hash/duplicate failure for {relative}")
        declared.add(relative)
    if declared != actual_files:
        raise ValueError("V8 smoke proof inventory does not exactly cover payload")
    report = json.loads((proof_root / "multi_atlas_bake_report.json").read_text())
    required_equal = {
        "schema": RESULT_SCHEMA,
        "mode": "smoke",
        "status": "smoke_complete",
        "source_pbr_used": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "runner_sha256": runner_sha256,
        "implementation": implementation_identity,
        "body_v6_texture_sha256": bake_contract.BODY_V6_TEXTURE_SHA256,
        "runner_execution_policy": RUNNER_EXECUTION_POLICY,
    }
    for field, expected in required_equal.items():
        if report.get(field) != expected:
            raise ValueError(
                f"V8 smoke proof {field} mismatch: {report.get(field)!r} != {expected!r}"
            )
    if report.get("quality_gate", {}).get("passed") is not True:
        raise ValueError("V8 smoke proof quality gate did not pass")
    _verify_radiometry_proof_identity(
        report.get("radiometry_video"), expected_radiometry_video
    )
    if report.get("dataset", {}).get("archive_sha256") != dataset_archive_sha256:
        raise ValueError("V8 smoke proof used a different dataset archive")
    if report.get("dataset", {}).get("manifest_sha256") != bake_contract.DATASET_V3_MANIFEST_SHA256:
        raise ValueError("V8 smoke proof is not pinned to canonical dataset manifest")
    if report.get("dataset", {}).get("geometry_sha256") != bake_contract.DATASET_V3_GEOMETRY_SHA256:
        raise ValueError("V8 smoke proof is not pinned to canonical Ferrari/BODY geometry")
    if report.get("dataset", {}).get("source_video_sha256") != bake_contract.DATASET_V3_VIDEO_SHA256:
        raise ValueError("V8 smoke proof is not pinned to canonical Seedance video")
    if report.get("alignment", {}).get("archive_sha256") != alignment_archive_sha256:
        raise ValueError("V8 smoke proof used a different alignment archive")
    if report.get("alignment", {}).get("report_sha256") != alignment_report_sha256:
        raise ValueError("V8 smoke proof used a different alignment report")
    if alignment_archive_sha256 != bake_contract.ALIGNMENT_V1_ARCHIVE_SHA256:
        raise ValueError("V8 smoke proof alignment archive is not canonical")
    if alignment_report_sha256 != bake_contract.ALIGNMENT_V1_REPORT_SHA256:
        raise ValueError("V8 smoke proof alignment report is not canonical")
    if report.get("unwrap", {}).get("archive_sha256") != unwrap_archive_sha256:
        raise ValueError("V8 smoke proof used a different unwrap archive")
    if unwrap_archive_sha256 != bake_contract.MINIMAL_UNWRAP_ARCHIVE_SHA256:
        raise ValueError("V8 smoke proof unwrap archive is not canonical minimal v8")
    if report.get("unwrap", {}).get("report_sha256") != unwrap_report_sha256:
        raise ValueError("V8 smoke proof used a different unwrap report")
    if report.get("unwrap", {}).get("blend_sha256") != unwrap_blend_sha256:
        raise ValueError("V8 smoke proof used a different unwrap carrier blend")
    if report.get("unwrap", {}).get("inventory_sha256") != unwrap_inventory_sha256:
        raise ValueError("V8 smoke proof used a different unwrap hash inventory")
    if unwrap_inventory_sha256 != bake_contract.MINIMAL_UNWRAP_INVENTORY_SHA256:
        raise ValueError("V8 smoke proof unwrap inventory is not canonical minimal v8")
    if report.get("unwrap", {}).get("sidecar_sha256") != unwrap_sidecar_sha256:
        raise ValueError("V8 smoke proof used different unwrap sidecars")
    mesh = report.get("mesh", {})
    if (
        mesh.get("body_is_raster_geometry_not_nonraster_sentinel") is not True
        or int(mesh.get("body_full_z_triangles", -1))
        != bake_contract.BODY_V6_PROXY_TRIANGLES
        or int(mesh.get("body_opaque_z_triangles", -1))
        != bake_contract.BODY_V6_PROXY_TRIANGLES
    ):
        raise ValueError("V8 smoke proof did not raster BODY in both visibility buffers")
    configuration = report.get("configuration", {})
    if configuration.get("selected_frame_ids") != expected_selected_ids:
        raise ValueError("V8 smoke proof did not use all 104 canonical views")
    smoke_size = int(configuration.get("smoke_atlas_size", -1))
    if smoke_size not in SMOKE_ALLOWED_ATLAS_SIZES:
        raise ValueError("V8 smoke proof atlas size is not 512 or 1024")
    expected_sizes = {
        slug: [smoke_size, smoke_size] for slug in EXPECTED_TEXTURES
    }
    actual_sizes = {
        row.get("page_id"): row.get("size") for row in report.get("pages", [])
    }
    if actual_sizes != expected_sizes:
        raise ValueError("V8 smoke proof did not exercise all six reduced pages")
    if not all(
        row.get("body_visibility_raster", {}).get(
            "all_views_have_full_z_body_pixels"
        )
        is True
        and row.get("body_visibility_raster", {}).get(
            "all_views_have_opaque_z_body_pixels"
        )
        is True
        and int(
            row.get("body_visibility_raster", {}).get(
                "minimum_full_z_body_pixels", 0
            )
        )
        > 0
        and int(
            row.get("body_visibility_raster", {}).get(
                "minimum_opaque_z_body_pixels", 0
            )
        )
        > 0
        and int(row.get("body_visibility_raster", {}).get("views", -1))
        == len(expected_selected_ids)
        for row in report.get("pages", [])
    ):
        raise ValueError("V8 smoke proof did not observe BODY pixels in all dual-z views")
    _verify_gpu_smoke_payload(report, proof_root, smoke_size)
    return {
        "archive_sha256": legacy._sha256(proof_zip),
        "report_sha256": legacy._sha256(
            proof_root / "multi_atlas_bake_report.json"
        ),
        "runner_sha256": runner_sha256,
        "implementation": implementation_identity,
        "selected_frame_ids": expected_selected_ids,
        "smoke_atlas_size": smoke_size,
        "radiometry_video_sha256": expected_radiometry_video["sha256"],
        "decoded_radiometry_frames_sha256": dict(
            expected_radiometry_video["decoded_frame_png_sha256"]
        ),
        "verified_files": len(declared),
    }


def _run_v8(
    dataset_url: str,
    alignment_url: str,
    unwrap_url: str,
    radiometry_video_url: str,
    *,
    smoke: bool,
    smoke_proof_url: str,
    smoke_atlas_size: int,
    render_long_side: int,
    seed: int,
):
    if not smoke and not smoke_proof_url:
        raise ValueError(
            "Production v8 requires a passing v8 smoke archive via smoke_proof_url"
        )
    if smoke_atlas_size not in SMOKE_ALLOWED_ATLAS_SIZES:
        raise ValueError("smoke_atlas_size must be 512 or 1024")
    if not radiometry_video_url:
        raise ValueError(
            "This v8 freeze requires the pinned existing SeedVR RGB-only override"
        )
    import gc
    import shutil
    import tempfile
    import time

    import numpy as np
    from PIL import Image as PIL_Image
    import nvdiffrast.torch as dr
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if render_long_side < 128 or render_long_side > 1280:
        raise ValueError("render_long_side must be in [128,1280]")
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    workspace = Path(tempfile.mkdtemp(prefix="car_city_six_page_v8_"))
    dataset_zip = workspace / "dataset.zip"
    alignment_zip = workspace / "alignment.zip"
    unwrap_zip = workspace / "unwrap.zip"
    dataset_extract = workspace / "dataset"
    unwrap_extract = workspace / "unwrap"
    output_dir = workspace / "output"
    (output_dir / "textures").mkdir(parents=True)
    (output_dir / "diagnostics").mkdir(parents=True)

    legacy._download(dataset_url, dataset_zip)
    legacy._download(alignment_url, alignment_zip)
    legacy._download(unwrap_url, unwrap_zip)
    dataset_archive_sha = legacy._sha256(dataset_zip)
    alignment_archive_sha = legacy._sha256(alignment_zip)
    unwrap_archive_sha = legacy._sha256(unwrap_zip)
    if unwrap_archive_sha != bake_contract.MINIMAL_UNWRAP_ARCHIVE_SHA256:
        raise ValueError("Unwrap input is not the pinned minimal canonical v8 archive")
    _validate_zip_container(unwrap_zip, "unwrap_pack_report.json")
    legacy._safe_extract(dataset_zip, dataset_extract)
    legacy._safe_extract(unwrap_zip, unwrap_extract)
    dataset_root = legacy._find_root(dataset_extract, "manifest.json")
    unwrap_root = legacy._find_root(unwrap_extract, "unwrap_pack_report.json")
    unwrap_archive_contract = _verify_minimal_unwrap_payload(unwrap_root)
    if (
        unwrap_archive_contract["inventory_sha256"]
        != bake_contract.MINIMAL_UNWRAP_INVENTORY_SHA256
    ):
        raise ValueError("Unwrap input inventory hash is not canonical minimal v8")
    runtime = load_runtime_contract(dataset_root, unwrap_root)
    unwrap_sidecar_sha256 = _runtime_sidecar_hashes(runtime)
    alignment = load_alignment_contract(alignment_zip, runtime.source)
    if alignment.archive_sha256 != alignment_archive_sha:
        raise ValueError("Alignment archive changed while loading")
    if alignment.archive_sha256 != bake_contract.ALIGNMENT_V1_ARCHIVE_SHA256:
        raise ValueError("Alignment archive is not the pinned canonical v1 archive")
    if alignment.report_sha256 != bake_contract.ALIGNMENT_V1_REPORT_SHA256:
        raise ValueError("Alignment report is not the pinned canonical v1 report")
    coordinate_contract = alignment.report.get(
        "coordinate_contract", alignment.report.get("conventions", {})
    )
    dense_camera = coordinate_contract.get(
        "dense_render_camera", alignment.report.get("dense_render_camera")
    )
    if dense_camera != "corrected_w2c_cv":
        raise ValueError(
            f"Dense warp is indexed by {dense_camera!r}, expected corrected_w2c_cv"
        )
    if len(alignment.eligible_frame_ids) != EXPECTED_ELIGIBLE_ALIGNMENT_FRAMES:
        raise ValueError("V8 requires exactly 104 gate-passing alignments")
    canonical_smoke_ids = list(alignment.eligible_frame_ids)
    if smoke:
        all_ids = canonical_smoke_ids
        train_ids, holdout_ids = _split_smoke_views(all_ids)
    else:
        all_ids = list(alignment.eligible_frame_ids)
        train_ids, holdout_ids = _split_smoke_views(all_ids)
    implementation_identity = _implementation_identity()
    runner_sha256 = implementation_identity["components"]["runner"]
    targets, radiometry_video = _prepare_seedvr_radiometry(
        radiometry_video_url,
        workspace,
        all_ids,
        runtime.source,
        PIL_Image,
    )
    smoke_proof = None
    if not smoke:
        smoke_proof = _verify_smoke_proof(
            smoke_proof_url,
            workspace / "proof",
            dataset_archive_sha256=dataset_archive_sha,
            dataset_manifest_sha256=runtime.source.manifest_sha256,
            alignment_archive_sha256=alignment.archive_sha256,
            alignment_report_sha256=alignment.report_sha256,
            unwrap_archive_sha256=unwrap_archive_sha,
            unwrap_report_sha256=runtime.unwrap_report_sha256,
            unwrap_blend_sha256=runtime.unwrap_blend_sha256,
            unwrap_inventory_sha256=unwrap_archive_contract["inventory_sha256"],
            unwrap_sidecar_sha256=unwrap_sidecar_sha256,
            runner_sha256=runner_sha256,
            implementation_identity=implementation_identity,
            expected_selected_ids=canonical_smoke_ids,
            expected_radiometry_video=radiometry_video,
        )
    camera_lookup = {
        int(frame_id): index
        for index, frame_id in enumerate(runtime.source.frame_ids)
    }
    width, height = legacy._stage_dimensions(
        runtime.source.width, runtime.source.height, render_long_side
    )
    alignment_supported = _alignment_supported_objects(alignment)
    # Neutral-fill authorization is tied to the exact coverage-aware canonical
    # 83-view partition used by both smoke and production.
    training_evidence = _training_alignment_evidence(alignment, train_ids)
    _log(
        f"mode={'smoke' if smoke else 'production'}; selected={len(all_ids)} "
        f"({len(train_ids)} train+{len(holdout_ids)} holdout); raster={width}x{height}"
    )

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    vertices = torch.from_numpy(runtime.visibility_vertices).to(
        device=device, dtype=torch.float32
    ).contiguous()
    faces = torch.from_numpy(runtime.visibility_faces).to(
        device=device, dtype=torch.int32
    ).contiguous()
    face_object = torch.from_numpy(runtime.visibility_face_object).to(
        device=device, dtype=torch.int32
    ).contiguous()
    face_piece = torch.from_numpy(runtime.visibility_face_piece).to(
        device=device, dtype=torch.int32
    ).contiguous()
    transparent = torch.tensor(
        bake_contract.DUAL_Z_EXCLUDED_OBJECT_IDS,
        device=device,
        dtype=torch.int32,
    )
    piece_target_full_z = torch.zeros(
        (runtime.visibility_piece_count,), device=device, dtype=torch.bool
    )
    piece_target_full_z[: runtime.nonbody_piece_count] = torch.isin(
        torch.from_numpy(runtime.piece_object).to(device=device, dtype=torch.int32),
        transparent,
    )
    ctx = dr.RasterizeCudaContext(device=device)

    page_reports = []
    piece_reports = []
    for index, page in enumerate(runtime.pages, start=1):
        atlas_size = smoke_atlas_size if smoke else page.size
        atlas_padding = SMOKE_PADDING if smoke else page.padding
        _log(
            f"page {index}/6 {page.page_id}: {atlas_size}x{atlas_size}, "
            f"{page.triangle_count} triangles, {len(page.piece_keys)} pieces"
        )
        page_report, rows = _bake_one_page(
            np,
            torch,
            F,
            dr,
            ctx,
            runtime,
            alignment,
            page,
            train_ids,
            holdout_ids,
            targets,
            camera_lookup,
            vertices,
            faces,
            face_object,
            face_piece,
            piece_target_full_z,
            transparent,
            alignment_supported,
            training_evidence,
            width,
            height,
            atlas_size,
            atlas_padding,
            output_dir,
            PIL_Image,
        )
        page_reports.append(page_report)
        piece_reports.extend(rows)
        torch.cuda.empty_cache()

    piece_reports.sort(key=lambda row: int(row["piece_index"]))
    object_reports = []
    for object_id in sorted(bake_contract.NONBODY_OBJECT_IDS):
        rows = [row for row in piece_reports if row["object_id"] == object_id]
        object_reports.append(
            {
                "object_id": object_id,
                "object_name": runtime.source.object_names[object_id],
                "page_id": bake_contract.page_for_object(object_id).page_id,
                "piece_ids": [row["piece_id"] for row in rows],
                "occupied_texels": sum(row["occupied_texels"] for row in rows),
                "direct_observed_texels": sum(
                    row["direct_observed_texels"] for row in rows
                ),
                "integration_eligible": all(
                    row["integration_eligible"] for row in rows
                ),
                "source_pbr_used": False,
            }
        )

    body_mask = runtime.visibility_face_object == bake_contract.BODY_OBJECT_ID
    body_full_z_triangles = int(np.count_nonzero(body_mask))
    body_opaque_z_triangles = int(
        np.count_nonzero(
            body_mask
            & ~np.isin(
                runtime.visibility_face_object,
                bake_contract.DUAL_Z_EXCLUDED_OBJECT_IDS,
            )
        )
    )
    body_piece_is_exact = bool(
        np.all(runtime.visibility_face_piece[body_mask] == runtime.body_piece_index)
    )
    no_alignment_ids = sorted(
        set(range(EXPECTED_OBJECTS)) - set(alignment_supported)
    )
    insufficient_multiview_ids = sorted(
        object_id
        for object_id, evidence in training_evidence.items()
        if int(evidence["pixels"]) < MIN_REQUIRED_TRAIN_ALIGNMENT_PIXELS
        or int(evidence["frames"]) < MIN_REQUIRED_TRAIN_ALIGNMENT_FRAMES
    )
    def direct_piece_ok(row, minimum_direct_texels: int) -> bool:
        return bool(
            int(row["direct_observed_texels"]) >= minimum_direct_texels
            and float(row["direct_coverage"])
            >= MIN_DIRECT_COVERAGE_PER_RECOVERABLE_PIECE
            and int(row["accepted_projection_samples"]) >= minimum_direct_texels
            and len(row["winning_source_frames"]) > 0
            and float(row["observed_chart_domain_ratio"])
            >= MIN_OBSERVED_CHART_DOMAIN_RATIO
        )

    def holdout_piece_ok(row, minimum_samples: int) -> bool:
        return int(row["holdout_samples"]) >= minimum_samples

    material_20 = bake_contract.material_visibility_metadata_for_piece((20, -1))
    material_38 = bake_contract.material_visibility_metadata_for_piece((38, -1))
    page_by_id = {row["page_id"]: row for row in page_reports}
    cockpit_glass_reveal_samples = sum(
        int(value)
        for value in page_by_id["cockpit"][
            "glass_warp_proxy_samples_per_frame"
        ].values()
    )
    cockpit_opaque_z_samples = sum(
        int(value["opaque_z"])
        for value in page_by_id["cockpit"]["zbuffer_samples_per_frame"].values()
    )
    glass_piece_self_z_samples = sum(
        int(value["piece_self_z"])
        for value in page_by_id["glass"]["zbuffer_samples_per_frame"].values()
    )
    lens_full_z_samples = sum(
        int(value["full_z"])
        for value in page_by_id["lights_translucent"][
            "zbuffer_samples_per_frame"
        ].values()
    )
    minimum_direct_texels = (
        MIN_SMOKE_DIRECT_TEXELS_PER_RECOVERABLE_PIECE
        if smoke
        else MIN_PRODUCTION_DIRECT_TEXELS_PER_RECOVERABLE_PIECE
    )
    minimum_holdout_samples = (
        MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE
        if smoke
        else MIN_PRODUCTION_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE
    )
    output_textures = [
        output_dir / "textures" / f"{slug}.png" for slug in EXPECTED_TEXTURES
    ]
    quality_checks = {
        "exact_six_page_ids_and_order": [
            row["page_id"] for row in page_reports
        ] == list(EXPECTED_TEXTURES),
        "exact_six_texture_files": all(path.is_file() for path in output_textures),
        "all_63_piece_ids_reported_once": [
            row["piece_id"] for row in piece_reports
        ] == list(runtime.piece_keys),
        "all_50_nonbody_parents_reported_once": [
            row["object_id"] for row in object_reports
        ] == sorted(bake_contract.NONBODY_OBJECT_IDS),
        "body_v6_is_passthrough_not_bake_target": "body_hero_8k" not in EXPECTED_TEXTURES,
        "canonical_dataset_manifest_hash": runtime.source.manifest_sha256
        == bake_contract.DATASET_V3_MANIFEST_SHA256,
        "canonical_ferrari_body_geometry_hash": runtime.source.geometry_sha256
        == bake_contract.DATASET_V3_GEOMETRY_SHA256,
        "canonical_seedance_video_hash": runtime.source.source_video_sha256
        == bake_contract.DATASET_V3_VIDEO_SHA256,
        "pinned_seedvr_rgb_only_hash": radiometry_video["sha256"]
        == bake_contract.SEEDVR_RGB_VIDEO_SHA256,
        "seedvr_rgb_only_exact_104_frame_identity_mapping": (
            radiometry_video["selected_frame_ids"] == all_ids
            and len(radiometry_video["decoded_frame_png_sha256"])
            == EXPECTED_ELIGIBLE_ALIGNMENT_FRAMES
            and radiometry_video["selected_max_frame_id"]
            == bake_contract.SEEDVR_RGB_MAX_CANONICAL_V8_FRAME_ID
            and radiometry_video["frame_357_selected"] is False
        ),
        "seedvr_rgb_pixel_center_mapping_3x_plus_1": (
            radiometry_video["native_to_rgb_pixel_center"]
            == [
                list(row)
                for row in bake_contract.SEEDVR_RGB_NATIVE_TO_RGB_PIXEL_CENTER
            ]
            and radiometry_video["sampling"]
            == "grid_sample_align_corners_false"
        ),
        "native_720p_remains_geometry_camera_depth_owner_alignment_authority": (
            radiometry_video[
                "geometry_camera_depth_owner_alignment_authority"
            ]["dataset_manifest_sha256"]
            == runtime.source.manifest_sha256
            and radiometry_video[
                "geometry_camera_depth_owner_alignment_authority"
            ]["geometry_sha256"]
            == runtime.source.geometry_sha256
            and radiometry_video[
                "geometry_camera_depth_owner_alignment_authority"
            ]["native_video_sha256"]
            == runtime.source.source_video_sha256
            and radiometry_video[
                "geometry_camera_depth_owner_alignment_authority"
            ]["resolution"]
            == [1280, 720]
        ),
        "body_is_actual_full_z_raster_geometry": body_full_z_triangles
        == bake_contract.BODY_V6_PROXY_TRIANGLES,
        "body_is_actual_opaque_z_raster_geometry": body_opaque_z_triangles
        == bake_contract.BODY_V6_PROXY_TRIANGLES,
        "body_raster_faces_have_exact_body_piece_owner": body_piece_is_exact,
        "body_has_nonzero_full_and_opaque_raster_pixels_in_every_selected_view": all(
            row["body_visibility_raster"]["views"] == len(all_ids)
            and row["body_visibility_raster"]["all_views_have_full_z_body_pixels"]
            and row["body_visibility_raster"]["all_views_have_opaque_z_body_pixels"]
            for row in page_reports
        ),
        "every_page_has_direct_seedance_texels": all(
            row["direct_observed_texels"] > 0 for row in page_reports
        ),
        "every_page_has_holdout_samples": all(
            row["holdout"]["frames_with_samples"] > 0 for row in page_reports
        ),
        "piece_chart_gutters_have_zero_cross_domain_blends": all(
            row["gutter"]["cross_piece_or_chart_blends"] == 0
            for row in page_reports
        ),
        "exact_alignment_count_104": len(alignment.eligible_frame_ids) == 104,
        "canonical_alignment_archive_hash": alignment.archive_sha256
        == bake_contract.ALIGNMENT_V1_ARCHIVE_SHA256,
        "canonical_alignment_report_hash": alignment.report_sha256
        == bake_contract.ALIGNMENT_V1_REPORT_SHA256,
        "canonical_minimal_unwrap_archive_hash": unwrap_archive_sha
        == bake_contract.MINIMAL_UNWRAP_ARCHIVE_SHA256,
        "canonical_minimal_unwrap_inventory_hash": unwrap_archive_contract[
            "inventory_sha256"
        ]
        == bake_contract.MINIMAL_UNWRAP_INVENTORY_SHA256,
        "source_pbr_not_consumed": all(
            row["source_pbr_used"] is False for row in piece_reports
        ),
        "exact_no_alignment_policy_object_21": no_alignment_ids
        == list(EXPECTED_NO_ALIGNMENT_OBJECT_IDS),
        "exact_insufficient_multiview_policy_objects_0_21_37":
        insufficient_multiview_ids
        == list(EXPECTED_INSUFFICIENT_MULTIVIEW_OBJECT_IDS),
        "all_63_pieces_have_uv_occupancy": all(
            row["occupied_texels"] > 0 for row in piece_reports
        ),
        "all_page_holdout_l1_below_fixed_threshold": all(
            row["holdout"]["mean_aligned_l1"] is not None
            and row["holdout"]["mean_aligned_l1"] <= MAX_PAGE_HOLDOUT_MEAN_L1
            for row in page_reports
        ),
        "neutral_fill_never_authorized_outside_objects_0_21_37": all(
            not row["neutral_fill_allowed"] or _explicit_low_evidence_exception(row)
            for row in piece_reports
        ),
        "fill_is_strict_piece_chart_domain": all(
            row["piece_chart_fill"]["chart_domains"] > 0
            and row["piece_chart_fill"]["observed_chart_domains"]
            + row["piece_chart_fill"]["neutral_chart_domains"]
            == row["piece_chart_fill"]["chart_domains"]
            for row in piece_reports
        ),
        "projection_gate_audits_are_piece_exact_and_funnel_consistent": all(
            _projection_gate_audit_is_consistent(row, split)
            for row in piece_reports
            for split in ("training", "holdout")
        ),
        "projection_visibility_buffer_matches_piece_contract": all(
            row["projection_visibility_buffer"]
            == (
                "counterfactual_target_piece_self_z_against_opaque_z"
                if row["piece_id"]
                in EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS
                else row["target_z_buffer"]
            )
            for row in piece_reports
        ),
        "global_warp_coarse_label_surrogate_exactly_whitelisted_39_to_36": (
            DENSE_OBJECT_SURROGATE_POLICY == {"39:-1": (36,)}
            and bake_contract.material_template_for_object(39)
            == bake_contract.material_template_for_object(36)
            == "v8_leather"
            and runtime.source.object_source_material[39] == "Leather_red"
            and runtime.source.object_source_material[36] == "Leather"
            and runtime.source.object_source_material[39]
            != runtime.source.object_source_material[36]
            and all(
                int(
                    row["projection_gate_audit"][split][
                        "accepted_global_warp_surrogate_samples"
                    ]
                )
                == 0
                for row in piece_reports
                if row["piece_id"] != "39:-1"
                for split in ("training", "holdout")
            )
        ),
        "steering_dense_provenance_meets_exact_or_surrogate_floors": all(
            _steering_dense_provenance_floor(
                row, "training", minimum_direct_texels
            )
            and _steering_dense_provenance_floor(
                row, "holdout", minimum_holdout_samples
            )
            for row in piece_reports
        ),
        "emissive_20_and_38_share_shader_but_not_visibility": (
            material_20["material_template"]
            == material_38["material_template"]
            == "v8_lamp_emissive"
            and material_20["target_z_buffer"] == "opaque_z"
            and material_20["visibility_class"] == "opaque_z_participant"
            and material_38["target_z_buffer"] == "full_z"
            and material_38["visibility_class"]
            == "transmissive_excluded_from_opaque_z"
        ),
        "cockpit_revealed_behind_glass_by_dense_warp_proxy":
        cockpit_glass_reveal_samples > 0,
        "cockpit_samples_use_opaque_z_behind_transmissive_exclusions":
        cockpit_opaque_z_samples > 0,
        "layered_glass_exercises_piece_self_z_and_lenses_exercise_full_z":
        glass_piece_self_z_samples > 0 and lens_full_z_samples > 0,
        "coverage_aware_split_is_exact_48_51_swap_without_semantic_alias": (
            int(bake_contract.V8_COVERAGE_AWARE_TRAIN_TO_HOLDOUT_FRAME_ID)
            not in train_ids
            and int(bake_contract.V8_COVERAGE_AWARE_TRAIN_TO_HOLDOUT_FRAME_ID)
            in holdout_ids
            and int(bake_contract.V8_COVERAGE_AWARE_HOLDOUT_TO_TRAIN_FRAME_ID)
            in train_ids
            and int(bake_contract.V8_COVERAGE_AWARE_HOLDOUT_TO_TRAIN_FRAME_ID)
            not in holdout_ids
            and bake_contract.V8_TRAIN_HOLDOUT_POLICY["semantic_alias_added"]
            is False
        ),
    }
    if smoke:
        quality_checks.update(
            {
                "canonical_full_104_view_smoke": all_ids == canonical_smoke_ids
                and len(all_ids) == EXPECTED_ELIGIBLE_ALIGNMENT_FRAMES,
                "smoke_split_83_21": len(train_ids) == 83
                and len(holdout_ids) == 21,
                "allowed_smoke_size": smoke_atlas_size in SMOKE_ALLOWED_ATLAS_SIZES,
                "all_six_pages_use_same_reduced_size": all(
                    row["size"] == [smoke_atlas_size, smoke_atlas_size]
                    for row in page_reports
                ),
                "canonical_smoke_render_size_1280x720": [width, height]
                == list(SMOKE_RENDER_SIZE),
                "smoke_all_63_pieces_have_training_geometric_opportunity_or_exact_piece_na": all(
                    _smoke_geometry_exercised(row, "training")
                    or _smoke_geometry_not_visible_exception(row, "training")
                    for row in piece_reports
                ),
                "smoke_all_63_pieces_have_holdout_geometric_opportunity_or_exact_piece_na": all(
                    _smoke_geometry_exercised(row, "holdout")
                    or _smoke_geometry_not_visible_exception(row, "holdout")
                    for row in piece_reports
                ),
                "smoke_no_owner_visible_piece_lacks_uv_projection_opportunity": all(
                    (
                        int(row["selected_training_screen_owner_pixels"]) == 0
                        and int(row["selected_training_visibility_opportunity_texels"])
                        == 0
                    )
                    or (
                        int(row["selected_training_screen_owner_pixels"]) > 0
                        and int(row["selected_training_visibility_opportunity_texels"])
                        > 0
                    )
                    for row in piece_reports
                ) and all(
                    (
                        int(row["selected_holdout_screen_owner_pixels"]) == 0
                        and int(row["selected_holdout_visibility_opportunity_texels"])
                        == 0
                    )
                    or (
                        int(row["selected_holdout_screen_owner_pixels"]) > 0
                        and int(row["selected_holdout_visibility_opportunity_texels"])
                        > 0
                    )
                    for row in piece_reports
                ),
                "smoke_no_colour_evidence_without_geometric_opportunity": all(
                    int(row["selected_training_visibility_opportunity_texels"]) > 0
                    or (
                        int(row["direct_observed_texels"]) == 0
                        and int(row["accepted_projection_samples"]) == 0
                        and row["winning_source_frames"] == []
                    )
                    for row in piece_reports
                ) and all(
                    int(row["selected_holdout_visibility_opportunity_texels"]) > 0
                    or int(row["holdout_samples"]) == 0
                    for row in piece_reports
                ),
                "smoke_all_selected_visible_recoverable_pieces_have_video_transfer": all(
                    direct_piece_ok(
                        row, MIN_SMOKE_DIRECT_TEXELS_PER_RECOVERABLE_PIECE
                    )
                    or _explicit_low_evidence_exception(row)
                    or _smoke_geometry_not_visible_exception(row, "training")
                    for row in piece_reports
                ),
                "frontmost_hidden_glass_is_exercised_by_counterfactual_piece_self_depth": all(
                    row["projection_visibility_buffer"]
                    == "counterfactual_target_piece_self_z_against_opaque_z"
                    and row["counterfactual_self_depth_policy"]
                    == _counterfactual_self_depth_policy_for_piece(
                        row["piece_id"]
                    )
                    and int(row["selected_training_screen_owner_pixels"]) > 0
                    and int(row["selected_holdout_screen_owner_pixels"]) > 0
                    and int(
                        row["selected_training_visibility_opportunity_texels"]
                    ) >= MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE
                    and int(
                        row["selected_holdout_visibility_opportunity_texels"]
                    ) >= MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE
                    for row in piece_reports
                    if row["piece_id"]
                    in EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS
                ),
                "smoke_all_selected_visible_recoverable_pieces_have_holdout": all(
                    holdout_piece_ok(
                        row, MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE
                    )
                    or _explicit_low_evidence_exception(row)
                    or _smoke_geometry_not_visible_exception(row, "holdout")
                    for row in piece_reports
                ),
                "coverage_aware_split_minor_glass_15_0_has_exact_holdout": any(
                    row["piece_id"] == "15:0"
                    and int(row["holdout_samples"])
                    >= MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE
                    and int(
                        row["projection_gate_audit"]["holdout"][
                            "accepted_exact_mapped_object_samples"
                        ]
                    )
                    >= MIN_SMOKE_HOLDOUT_SAMPLES_PER_RECOVERABLE_PIECE
                    and int(
                        row["projection_gate_audit"]["holdout"][
                            "accepted_global_warp_surrogate_samples"
                        ]
                    )
                    == 0
                    for row in piece_reports
                ),
            }
        )
    else:
        quality_checks.update(
            {
                "all_pieces_have_uv_occupancy": all(
                    row["occupied_texels"] > 0 for row in piece_reports
                ),
                "all_63_pieces_have_video_transfer_or_explicit_low_evidence_policy": all(
                    direct_piece_ok(row, minimum_direct_texels)
                    or _explicit_low_evidence_exception(row)
                    for row in piece_reports
                ),
                "all_63_pieces_have_holdout_or_explicit_low_evidence_policy": all(
                    holdout_piece_ok(row, minimum_holdout_samples)
                    or _explicit_low_evidence_exception(row)
                    for row in piece_reports
                ),
                "all_recoverable_supported_pieces_have_video_transfer_texels": all(
                    row["direct_observed_texels"] > 0
                    for row in piece_reports
                    if row["alignment_has_parent_object_pixels"]
                    and row["training_alignment_sufficient_for_required_texture"]
                ),
                "canonical_split_83_21": len(train_ids) == 83
                and len(holdout_ids) == 21,
                "matching_smoke_proof_verified": smoke_proof is not None,
                "native_page_resolutions": all(
                    row["size"] == row["canonical_size"] for row in page_reports
                ),
            }
        )
    quality_gate_passed = all(quality_checks.values())
    integration_eligible = (
        not smoke
        and quality_gate_passed
        and all(row["integration_eligible"] for row in piece_reports)
    )
    elapsed = time.perf_counter() - started
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA,
        "mode": "smoke" if smoke else "production",
        "status": (
            "smoke_complete"
            if smoke and quality_gate_passed
            else "complete"
            if integration_eligible
            else "gate_failed"
        ),
        "integration_eligible": integration_eligible,
        "gpu_execution": True,
        "runner_sha256": runner_sha256,
        "implementation": implementation_identity,
        "runner_execution_policy": RUNNER_EXECUTION_POLICY,
        "source_pbr_used": False,
        "dataset_manifest_sha256": runtime.source.manifest_sha256,
        "body_v6_texture_sha256": bake_contract.BODY_V6_TEXTURE_SHA256,
        "body_v6_mode": "immutable_passthrough",
        "method": (
            "six sequential non-BODY RGB pages; corrected-camera dense-warp inverse "
            "projection; 179488-triangle representative whole-car full/opaque dual-z; "
            "exact piece_id visibility and adaptive erosion; chart-coherent two-pass "
            "winner; pre-warp geometric opportunity audit; piece-isolated fill; "
            "target-crop exact-ray self-depth for layered glass; categorical "
            "winner provenance; deterministic piece+chart gutters"
        ),
        "dataset": {
            "schema_version": runtime.source.manifest["schema_version"],
            "archive_sha256": dataset_archive_sha,
            "manifest_sha256": runtime.source.manifest_sha256,
            "geometry_sha256": runtime.source.geometry_sha256,
            "source_video_sha256": runtime.source.source_video_sha256,
        },
        "radiometry_video": radiometry_video,
        "alignment": {
            "schema_version": alignment.report["schema_version"],
            "archive_sha256": alignment.archive_sha256,
            "report_sha256": alignment.report_sha256,
            "reported_frames": len(alignment.reported_frame_ids),
            "eligible_frames": len(alignment.eligible_frame_ids),
            "excluded_frame_ids": list(alignment.excluded_frame_ids),
            "dense_render_camera": dense_camera,
            "evidence_policy": {
                "no_alignment_object_ids": no_alignment_ids,
                "insufficient_multiview_object_ids": insufficient_multiview_ids,
                "expected_no_alignment_object_ids": list(
                    EXPECTED_NO_ALIGNMENT_OBJECT_IDS
                ),
                "expected_insufficient_multiview_object_ids": list(
                    EXPECTED_INSUFFICIENT_MULTIVIEW_OBJECT_IDS
                ),
                "exact_low_evidence_piece_ids": list(
                    EXPECTED_INSUFFICIENT_MULTIVIEW_PIECE_IDS
                ),
                "frontmost_hidden_glass_piece_ids": list(
                    EXPECTED_FRONTMOST_HIDDEN_GLASS_PIECE_IDS
                ),
                "frontmost_hidden_glass_policy": (
                    "counterfactual_target_piece_self_z_against_opaque_z;"
                    "never_geometry_na;source_video_radiometry_remains_composite"
                ),
                "dense_object_surrogate_policy": {
                    key: list(value)
                    for key, value in DENSE_OBJECT_SURROGATE_POLICY.items()
                },
                "dense_object_surrogate_minimum_confidence": (
                    DENSE_OBJECT_SURROGATE_MIN_CONFIDENCE
                ),
                "special_object_policy": SPECIAL_EVIDENCE_POLICY,
            },
        },
        "unwrap": {
            "archive_sha256": unwrap_archive_sha,
            "report_sha256": runtime.unwrap_report_sha256,
            "blend_sha256": runtime.unwrap_blend_sha256,
            "inventory_sha256": unwrap_archive_contract["inventory_sha256"],
            "verified_payload_files": unwrap_archive_contract[
                "verified_payload_files"
            ],
            "sidecar_sha256": unwrap_sidecar_sha256,
            "sidecars": 50,
            "piece_ids": 63,
        },
        "configuration": {
            "seed": seed,
            "render_size": [width, height],
            "smoke_atlas_size": smoke_atlas_size if smoke else None,
            "eligible_frame_ids": list(alignment.eligible_frame_ids),
            "selected_frame_ids": all_ids,
            "training_frame_ids": list(train_ids),
            "holdout_frame_ids": list(holdout_ids),
            "train_holdout_policy": dict(bake_contract.V8_TRAIN_HOLDOUT_POLICY),
            "page_execution": "strictly_sequential",
            "opaque_z_excluded_object_ids": list(
                bake_contract.DUAL_Z_EXCLUDED_OBJECT_IDS
            ),
            "quality_thresholds": {
                "minimum_direct_texels_per_recoverable_piece": minimum_direct_texels,
                "minimum_direct_coverage_per_recoverable_piece":
                MIN_DIRECT_COVERAGE_PER_RECOVERABLE_PIECE,
                "minimum_observed_chart_domain_ratio":
                MIN_OBSERVED_CHART_DOMAIN_RATIO,
                "minimum_holdout_samples_per_recoverable_piece":
                minimum_holdout_samples,
                "minimum_smoke_geometric_opportunity_texels_per_piece": (
                    MIN_SMOKE_GEOMETRIC_OPPORTUNITY_TEXELS_PER_PIECE
                    if smoke
                    else None
                ),
                "maximum_page_holdout_mean_l1": MAX_PAGE_HOLDOUT_MEAN_L1,
            },
        },
        "mesh": {
            "semantic_objects": 51,
            "vertices": int(len(runtime.visibility_vertices)),
            "triangles_for_visibility": int(len(runtime.visibility_faces)),
            "body_representative_triangles": bake_contract.BODY_V6_PROXY_TRIANGLES,
            "nonbody_proxy_triangles": bake_contract.EXPECTED_PROXY_FACES,
            "nonbody_piece_ids": runtime.nonbody_piece_count,
            "body_visibility_piece_index": runtime.body_piece_index,
            "body_is_raster_geometry_not_nonraster_sentinel": True,
            "body_full_z_triangles": body_full_z_triangles,
            "body_opaque_z_triangles": body_opaque_z_triangles,
            "body_visibility_geometry_dataset_manifest_sha256":
            runtime.source.manifest_sha256,
            "body_visibility_geometry_sha256": runtime.source.geometry_sha256,
        },
        "future_materials": {
            "templates": bake_contract.MATERIAL_TEMPLATES,
            "visibility_overrides": bake_contract.MATERIAL_VISIBILITY_OVERRIDES,
            "object_20": material_20,
            "object_38": material_38,
            "routing_is_semantic_not_source_pbr": True,
        },
        "pages": page_reports,
        "atlases": page_reports,
        "pieces": piece_reports,
        "objects": object_reports,
        "source_pbr_textures_used": False,
        "source_pbr_texture_files_consumed": [],
        "postprocess_contract": {
            "runner_output_stage": "raw_seedvr_rgb_only_transfer_on_canonical_v8_uv",
            "seedream_invoked_by_runner": False,
            "seedream_may_run_only_after_raw_bake": True,
            "seedream_must_not_move_uv_islands": True,
            "required_per_page_diagnostics": [
                "occupied_mask.png",
                "observed_mask.png",
                "generated_fill_mask.png",
                "per_texel_fallback_mask.png",
                "chart_owner_id.png",
                "piece_owner_id.png",
                "source_frame_id.png",
                "alignment_confidence.png",
                "view_support.png",
                "video_evidence_class_id.png",
                "training_visibility_opportunity_mask.png",
                "holdout_visibility_opportunity_mask.png",
            ],
            "deterministic_gutters_owned_by_this_pipeline": True,
            "upscale_stage": "after_seedream_recomposition_and_owned_gutters",
        },
        "smoke_proof": smoke_proof,
        "quality_gate": {
            "passed": quality_gate_passed,
            "checks": quality_checks,
            "policy": (
                "production requires native six-page sizes, one report per 63 pieces, "
                "UV occupancy for every piece, full 83/21 geometric opportunity for "
                "every recoverable exact piece, direct Seedance colour and holdout "
                "floors, zero cross-domain gutter blends, and an exact matching v8 "
                "smoke proof"
            ),
        },
        "artifacts": {
            slug: f"textures/{slug}.png" for slug in EXPECTED_TEXTURES
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "peak_vram_allocated_gib": float(
                torch.cuda.max_memory_allocated(device) / 1024**3
            ),
            "peak_vram_reserved_gib": float(
                torch.cuda.max_memory_reserved(device) / 1024**3
            ),
        },
        "limitations": [
            "The six RGB pages encode captured appearance, not a recovered physical PBR decomposition.",
            "Future glass/lens/chrome shaders are new semantic templates; source PBR payload is forbidden.",
            "BODY v6 is referenced by immutable hash and is intentionally absent from the bake outputs.",
        ],
    }
    (output_dir / "multi_atlas_bake_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.txt").write_text(
        "Ferrari Seedance geometry + existing SeedVR RGB six-page bake v8\n\n"
        "Six non-BODY textures are under textures/. BODY remains the exact packed v6 "
        "texture referenced by SHA-256 in the report. Every colour sample comes from "
        "the pinned SeedVR RGB-only carrier; geometry, cameras, depth, owners and "
        "alignment remain the canonical native 720p dataset. Original Ferrari PBR "
        "textures are forbidden.\n"
    )
    _write_inventory(output_dir)
    result_zip = shutil.make_archive(
        str(workspace / "multi_atlas_bake_v8"), "zip", output_dir
    )
    artifact = File.from_path(result_zip, content_type="application/zip")
    _log(f"{report['status']} in {elapsed / 60:.1f} min; RESULT_URL={artifact.url}")
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "mode": "smoke" if smoke else "production",
        "status": report["status"],
        "integration_eligible": integration_eligible,
        "results_zip": artifact.url,
        "runner_sha256": runner_sha256,
        "implementation_sha256": implementation_identity["sha256"],
        "dataset_sha256": dataset_archive_sha,
        "alignment_sha256": alignment_archive_sha,
        "unwrap_sha256": unwrap_archive_sha,
        "radiometry_video_sha256": radiometry_video["sha256"],
        "quality_checks": quality_checks,
        "elapsed_seconds": elapsed,
    }


@fal.function(
    image=IMAGE,
    machine_type="GPU-H100",
    request_timeout=14_400,
    local_python_modules=[
        "bake_object_textures",
        "body_bake_v3",
        "contract",
        "runtime_contract",
    ],
)
def smoke_ferrari_atlases_v8(
    dataset_url: str = DEFAULT_DATASET_URL,
    alignment_url: str = DEFAULT_ALIGNMENT_URL,
    unwrap_url: str = DEFAULT_UNWRAP_URL,
    radiometry_video_url: str = DEFAULT_RADIOMETRY_VIDEO_URL,
    smoke_atlas_size: int = 1024,
    seed: int = 83,
):
    """Exercise all six pages on the full 104-view split at 512 or 1024."""
    return _run_v8(
        dataset_url,
        alignment_url,
        unwrap_url,
        radiometry_video_url,
        smoke=True,
        smoke_proof_url="",
        smoke_atlas_size=smoke_atlas_size,
        render_long_side=1280,
        seed=seed,
    )


@fal.function(
    image=IMAGE,
    machine_type="GPU-H100",
    request_timeout=43_200,
    local_python_modules=[
        "bake_object_textures",
        "body_bake_v3",
        "contract",
        "runtime_contract",
    ],
)
def bake_ferrari_atlases_v8(
    dataset_url: str = DEFAULT_DATASET_URL,
    alignment_url: str = DEFAULT_ALIGNMENT_URL,
    unwrap_url: str = DEFAULT_UNWRAP_URL,
    radiometry_video_url: str = DEFAULT_RADIOMETRY_VIDEO_URL,
    smoke_proof_url: str = DEFAULT_SMOKE_PROOF_URL,
    render_long_side: int = 1280,
    seed: int = 83,
):
    """Run strict native 4K/8K production; definition alone submits nothing."""
    return _run_v8(
        dataset_url,
        alignment_url,
        unwrap_url,
        radiometry_video_url,
        smoke=False,
        smoke_proof_url=smoke_proof_url,
        smoke_atlas_size=512,
        render_long_side=render_long_side,
        seed=seed,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Import this module or invoke a v8 entrypoint through fal; direct "
        "execution intentionally does not submit a job."
    )
