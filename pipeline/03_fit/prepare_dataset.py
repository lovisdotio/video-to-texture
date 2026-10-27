#!/usr/bin/env python3
"""Build the compact v03 dataset consumed by the differentiable texture fitter.

The script is deliberately local-only: it reads the reference video, Blender camera
poses and UniTEX mesh arrays, then writes cropped frames plus numerical metadata.  It
does not upload anything or call a paid service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


def _dependency_error(exc: BaseException) -> "NoReturn":
    here = Path(__file__).resolve()
    expected = here.parents[1] / "tools" / "venv" / "bin" / "python"
    raise SystemExit(
        "prepare_dataset.py requires numpy, Pillow, trimesh and xatlas.\n"
        f"Use the project environment:\n  {expected} {here}\n"
        f"Import error: {exc}"
    )


try:
    import numpy as np
    from PIL import Image
    import trimesh
    import xatlas

    # A partial namespace-only NumPy installation exists on this workstation.
    if not hasattr(np, "load") or not hasattr(np, "__version__"):
        raise ImportError("the active NumPy installation is incomplete")
except (ImportError, AttributeError) as dependency_exception:
    _dependency_error(dependency_exception)


SCHEMA_VERSION = "car-city-video-fit-dataset-v1"
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
EXPECTED_TRIANGLES = 179_498
EXPECTED_VERTICES = 111_972


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Command failed ({' '.join(command)}):\n{exc.stderr.strip()}"
        ) from exc
    return json.loads(completed.stdout)


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def video_info(video: Path, ffprobe: str) -> dict[str, Any]:
    payload = run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video),
        ]
    )
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {video}, got {len(streams)}")
    return streams[0]


def load_camera_data(
    transforms_path: Path, video_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    transforms = json.loads(transforms_path.read_text())
    required = ("fl_x", "fl_y", "cx", "cy", "w", "h", "frames")
    missing = [key for key in required if key not in transforms]
    if missing:
        raise ValueError(f"Missing transforms.json fields: {missing}")

    by_blender_frame: dict[int, dict[str, Any]] = {}
    for entry in transforms["frames"]:
        try:
            frame_number = int(Path(entry["file_path"]).stem)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Bad camera frame entry: {entry}") from exc
        by_blender_frame[frame_number] = entry

    blender_frames = video_indices + 1
    matrices = []
    for frame_number in blender_frames.tolist():
        if frame_number not in by_blender_frame:
            raise ValueError(
                f"No Blender pose for video frame {frame_number - 1}; "
                f"expected transforms entry {frame_number:04d}"
            )
        matrix = np.asarray(
            by_blender_frame[frame_number]["transform_matrix"], dtype=np.float64
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid pose matrix for Blender frame {frame_number}")
        matrices.append(matrix)

    c2w = np.stack(matrices)
    rotation = c2w[:, :3, :3]
    orthogonality_error = np.max(
        np.abs(rotation @ np.swapaxes(rotation, 1, 2) - np.eye(3)[None])
    )
    if orthogonality_error > 2e-5:
        raise ValueError(f"Camera rotations are not orthogonal (max error {orthogonality_error})")
    if not np.allclose(c2w[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("Camera transforms are not homogeneous rigid transforms")

    K = np.asarray(
        [
            [transforms["fl_x"], 0.0, transforms["cx"]],
            [0.0, transforms["fl_y"], transforms["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return c2w, K, transforms


def load_and_denormalize_mesh(
    vertices_path: Path,
    faces_path: Path,
    clean_mesh_path: Path,
    unitex_scale: float,
) -> dict[str, np.ndarray]:
    vertices_normalized = np.load(vertices_path, allow_pickle=False)
    faces = np.load(faces_path, allow_pickle=False)
    if vertices_normalized.shape != (EXPECTED_VERTICES, 3):
        raise ValueError(
            f"Unexpected vertex array {vertices_normalized.shape}; "
            f"expected {(EXPECTED_VERTICES, 3)}"
        )
    if faces.shape != (EXPECTED_TRIANGLES, 3):
        raise ValueError(
            f"Unexpected face array {faces.shape}; expected {(EXPECTED_TRIANGLES, 3)}"
        )
    if faces.min() < 0 or faces.max() >= len(vertices_normalized):
        raise ValueError("Face array contains out-of-range vertex indices")
    if not np.isfinite(vertices_normalized).all():
        raise ValueError("Mesh contains non-finite vertices")

    scene = trimesh.load(clean_mesh_path, force="scene", process=False)
    bounds_gltf = np.asarray(scene.bounds, dtype=np.float64)
    if bounds_gltf.shape != (2, 3) or not np.isfinite(bounds_gltf).all():
        raise ValueError(f"Could not obtain finite scene bounds from {clean_mesh_path}")

    # UniTEX preprocess_blank_mesh_o3d uses:
    #   sss = max_extent / (2 * scale)
    #   v_normalized = v_world / sss - center / sss
    # The stored mesh_V array is the normalized mesh after glTF -> Blender axes:
    #   (x, y, z)_gltf -> (x, -z, y)_blender.
    center_gltf = bounds_gltf.mean(axis=0)
    denormalization_scale = float(
        (bounds_gltf[1] - bounds_gltf[0]).max() / (2.0 * unitex_scale)
    )
    center_blender = np.asarray(
        [center_gltf[0], -center_gltf[2], center_gltf[1]], dtype=np.float64
    )
    vertices_world = (
        np.asarray(vertices_normalized, dtype=np.float64) * denormalization_scale
        + center_blender
    )

    normalized_to_world = np.eye(4, dtype=np.float64)
    normalized_to_world[:3, :3] *= denormalization_scale
    normalized_to_world[:3, 3] = center_blender
    world_to_normalized = np.linalg.inv(normalized_to_world)

    # Cross-check the derived affine in the original glTF coordinate system.  This
    # catches axis mistakes and a changed clean mesh before they poison every pose.
    normalized_gltf = np.stack(
        [vertices_normalized[:, 0], vertices_normalized[:, 2], -vertices_normalized[:, 1]],
        axis=1,
    )
    expected_normalized_bounds = (bounds_gltf - center_gltf) / denormalization_scale
    actual_normalized_bounds = np.stack(
        [normalized_gltf.min(axis=0), normalized_gltf.max(axis=0)]
    )
    bounds_error = float(np.max(np.abs(actual_normalized_bounds - expected_normalized_bounds)))
    if bounds_error > 5e-5:
        raise ValueError(
            "Derived UniTEX de-normalization does not match mesh_V.npy "
            f"(bounds error {bounds_error:.6g})"
        )

    triangles = vertices_world[faces]
    face_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(face_cross, axis=1) < 1e-12):
        raise ValueError("Mesh contains degenerate triangles")
    vertex_normals = np.zeros_like(vertices_world)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_cross)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-15)

    return {
        "vertices_normalized": np.asarray(vertices_normalized, dtype=np.float32),
        "vertices_world": np.asarray(vertices_world, dtype=np.float32),
        "vertex_normals_world": np.asarray(vertex_normals, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
        "bounds_gltf": bounds_gltf,
        "center_gltf": center_gltf,
        "center_blender": center_blender,
        "denormalization_scale": np.asarray(denormalization_scale, dtype=np.float64),
        "normalized_to_world": normalized_to_world,
        "world_to_normalized": world_to_normalized,
        "bounds_error": np.asarray(bounds_error, dtype=np.float64),
    }


def repack_uvs(
    vertices_normalized: np.ndarray,
    faces: np.ndarray,
    atlas_size: int,
    padding: int,
    texels_per_unit: float,
    input_scale: float,
) -> dict[str, np.ndarray | int | float]:
    atlas = xatlas.Atlas()
    # xatlas classifies a few valid but microscopic trim/detail triangles as
    # degenerate at the normalized car scale.  A uniform pre-scale avoids its
    # absolute epsilon and is UV-equivalent; compensate the requested density.
    atlas.add_mesh(
        np.ascontiguousarray(vertices_normalized * input_scale, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
    )
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = atlas_size
    pack_options.padding = padding
    pack_options.texels_per_unit = texels_per_unit / input_scale
    pack_options.bilinear = True
    pack_options.blockAlign = False
    pack_options.bruteForce = False
    pack_options.create_image = False
    pack_options.rotate_charts = True
    pack_options.rotate_charts_to_axis = True
    atlas.generate(chart_options, pack_options, verbose=False)

    vertex_remap, uv_faces, uv_coords = atlas.get_mesh(0)
    vertex_remap = np.asarray(vertex_remap, dtype=np.int32)
    uv_faces = np.asarray(uv_faces, dtype=np.int32)
    uv_coords = np.asarray(uv_coords, dtype=np.float32)
    if atlas.atlas_count != 1:
        raise ValueError(f"xatlas generated {atlas.atlas_count} atlases; exactly one is required")
    if atlas.width != atlas_size or atlas.height != atlas_size:
        raise ValueError(
            f"xatlas generated {atlas.width}x{atlas.height}, not {atlas_size}x{atlas_size}. "
            "Adjust --texels-per-unit for this mesh."
        )
    if uv_faces.shape != faces.shape:
        raise ValueError(f"xatlas changed face count: {uv_faces.shape} vs {faces.shape}")
    if not np.array_equal(vertex_remap[uv_faces], faces):
        raise ValueError("xatlas changed triangle topology or ordering")
    if not np.isfinite(uv_coords).all() or uv_coords.min() < -1e-6 or uv_coords.max() > 1.000001:
        raise ValueError("xatlas returned invalid normalized UV coordinates")
    uv_triangles = uv_coords[uv_faces]
    twice_area = (
        (uv_triangles[:, 1, 0] - uv_triangles[:, 0, 0])
        * (uv_triangles[:, 2, 1] - uv_triangles[:, 0, 1])
        - (uv_triangles[:, 1, 1] - uv_triangles[:, 0, 1])
        * (uv_triangles[:, 2, 0] - uv_triangles[:, 0, 0])
    )
    if np.any(np.abs(twice_area) < 1e-12):
        raise ValueError("xatlas returned degenerate UV triangles")

    return {
        "uv_vertex_remap": vertex_remap,
        "uv_faces": uv_faces,
        "uv_coords": uv_coords,
        "atlas_width": int(atlas.width),
        "atlas_height": int(atlas.height),
        "chart_count": int(atlas.chart_count),
        "utilization": float(atlas.utilization),
        "actual_texels_per_unit": float(atlas.texels_per_unit * input_scale),
        "raw_xatlas_texels_per_unit": float(atlas.texels_per_unit),
        "input_scale": float(input_scale),
    }


def project_mesh_bbox(
    vertices_world: np.ndarray, c2w: np.ndarray, K_full: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w2c_blender = np.linalg.inv(c2w)
    w2c_opencv = BLENDER_TO_OPENCV[None] @ w2c_blender
    vertices_h = np.concatenate(
        [np.asarray(vertices_world, dtype=np.float64), np.ones((len(vertices_world), 1))],
        axis=1,
    )
    bboxes = []
    projection_full = []
    for index in range(len(c2w)):
        camera_xyz = vertices_h @ w2c_opencv[index].T
        in_front = camera_xyz[:, 2] > 1e-6
        if not np.all(in_front):
            raise ValueError(f"Mesh crosses camera plane in sampled view {index}")
        pixels_h = camera_xyz[:, :3] @ K_full.T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        bboxes.append(
            [pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()]
        )
        projection_full.append(K_full @ w2c_opencv[index, :3, :])
    return (
        np.asarray(bboxes, dtype=np.float64),
        w2c_blender,
        w2c_opencv,
        np.asarray(projection_full, dtype=np.float64),
    )


def choose_square_crops(
    bboxes: np.ndarray, width: int, height: int, margin: float
) -> np.ndarray:
    crops = []
    for bbox in bboxes:
        bbox_width, bbox_height = bbox[2:] - bbox[:2]
        required = float(max(bbox_width, bbox_height))
        side = max(2, int(np.ceil(required * (1.0 + 2.0 * margin))))

        # Keep ordinary crops inside the source image.  If the projected mesh itself
        # is larger than the short image dimension, retain it and let Pillow pad.
        if required <= min(width, height):
            side = min(side, min(width, height))
        center_x = float((bbox[0] + bbox[2]) * 0.5)
        center_y = float((bbox[1] + bbox[3]) * 0.5)
        x0 = int(round(center_x - side * 0.5))
        y0 = int(round(center_y - side * 0.5))
        if side <= width:
            x0 = min(max(x0, 0), width - side)
        if side <= height:
            y0 = min(max(y0, 0), height - side)
        if bbox[0] < x0 - 1e-5 or bbox[2] > x0 + side + 1e-5:
            raise ValueError(f"Crop does not contain projected horizontal mesh bbox {bbox}")
        if bbox[1] < y0 - 1e-5 or bbox[3] > y0 + side + 1e-5:
            raise ValueError(f"Crop does not contain projected vertical mesh bbox {bbox}")
        crops.append([x0, y0, side, side])
    return np.asarray(crops, dtype=np.int32)


def crop_transforms(
    crops: np.ndarray, output_resolution: int, K_full: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_from_full = []
    for x0, y0, width, height in crops.tolist():
        if width != height:
            raise ValueError("Only square crops are supported")
        scale = output_resolution / float(width)
        # Pixel-center-aware equivalent of Pillow crop followed by resize:
        # out = scale * (full - crop_origin + 0.5) - 0.5.
        image_from_full.append(
            [
                [scale, 0.0, scale * (0.5 - x0) - 0.5],
                [0.0, scale, scale * (0.5 - y0) - 0.5],
                [0.0, 0.0, 1.0],
            ]
        )
    image_from_full_array = np.asarray(image_from_full, dtype=np.float64)
    full_from_image = np.linalg.inv(image_from_full_array)
    K_crop = image_from_full_array @ K_full[None]
    return image_from_full_array, full_from_image, K_crop


def laplacian_variance(rgb: np.ndarray) -> float:
    gray = (
        rgb[:, :, 0].astype(np.float32) * 0.299
        + rgb[:, :, 1].astype(np.float32) * 0.587
        + rgb[:, :, 2].astype(np.float32) * 0.114
    )
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def extract_and_crop_frames(
    video: Path,
    frames_dir: Path,
    video_indices: np.ndarray,
    crops: np.ndarray,
    full_width: int,
    full_height: int,
    output_resolution: int,
    jpeg_quality: int,
    ffmpeg: str,
) -> tuple[list[str], np.ndarray, list[str]]:
    if not np.array_equal(video_indices, np.arange(0, 358, 3, dtype=np.int32)):
        raise ValueError("The ffmpeg selector is fixed to required v03 indices 0,3,...,357")
    frames_dir.mkdir(parents=True, exist_ok=False)
    frame_bytes = full_width * full_height * 3
    selector = "select=not(mod(n\\,3))*lte(n\\,357)"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        selector,
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required executable not found: {ffmpeg}") from exc
    assert process.stdout is not None and process.stderr is not None

    frame_files: list[str] = []
    sharpness: list[float] = []
    frame_hashes: list[str] = []
    try:
        for dataset_index in range(len(video_indices)):
            data = process.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                stderr = process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"ffmpeg returned only {len(data)}/{frame_bytes} bytes for frame "
                    f"{dataset_index}: {stderr.strip()}"
                )
            full = Image.frombytes("RGB", (full_width, full_height), data)
            x0, y0, width, height = crops[dataset_index].tolist()
            cropped = full.crop((x0, y0, x0 + width, y0 + height)).resize(
                (output_resolution, output_resolution), Image.Resampling.LANCZOS
            )
            relative_path = f"frames/{dataset_index:04d}.jpg"
            output_path = frames_dir.parent / relative_path
            cropped.save(
                output_path,
                format="JPEG",
                quality=jpeg_quality,
                subsampling=0,
                optimize=True,
            )
            frame_files.append(relative_path)
            sharpness.append(laplacian_variance(np.asarray(cropped)))
            frame_hashes.append(sha256(output_path))

        extra = process.stdout.read(1)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with {return_code}: {stderr.strip()}")
        if extra:
            raise RuntimeError("ffmpeg selected more than the required 120 frames")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return frame_files, np.asarray(sharpness, dtype=np.float32), frame_hashes


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def make_archive(dataset_dir: Path, archive_path: Path) -> None:
    files = [dataset_dir / "manifest.json", dataset_dir / "dataset.npz"]
    files.extend(sorted((dataset_dir / "frames").glob("*.jpg")))
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(dataset_dir).as_posix())


def validate_dataset(dataset_dir: Path, archive_path: Path | None = None) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    npz_path = dataset_dir / "dataset.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema {manifest.get('schema_version')}")
    with np.load(npz_path, allow_pickle=False) as data:
        required_shapes = {
            "video_frame_indices": (120,),
            "blender_frame_numbers": (120,),
            "frame_files": (120,),
            "c2w": (120, 4, 4),
            "w2c": (120, 4, 4),
            "w2c_opencv": (120, 4, 4),
            "K_full": (120, 3, 3),
            "K_crop": (120, 3, 3),
            "crop_xywh": (120, 4),
            "image_from_full": (120, 3, 3),
            "full_from_image": (120, 3, 3),
            "projected_bbox_full": (120, 4),
            "vertices_normalized": (EXPECTED_VERTICES, 3),
            "vertices_world": (EXPECTED_VERTICES, 3),
            "vertex_normals_world": (EXPECTED_VERTICES, 3),
            "faces": (EXPECTED_TRIANGLES, 3),
            "uv_faces": (EXPECTED_TRIANGLES, 3),
            "sharpness_laplacian_var": (120,),
        }
        for key, expected in required_shapes.items():
            if key not in data or data[key].shape != expected:
                got = None if key not in data else data[key].shape
                raise ValueError(f"Bad dataset array {key}: {got}, expected {expected}")
        expected_indices = np.arange(0, 358, 3, dtype=np.int32)
        if not np.array_equal(data["video_frame_indices"], expected_indices):
            raise ValueError("Video frame indices are not exactly 0,3,...,357")
        if not np.array_equal(data["blender_frame_numbers"], expected_indices + 1):
            raise ValueError("Blender poses are not matched to video frame n + 1")
        if not np.allclose(data["w2c"] @ data["c2w"], np.eye(4)[None], atol=2e-5):
            raise ValueError("w2c is not the inverse of c2w")
        if not np.allclose(
            data["image_from_full"] @ data["full_from_image"],
            np.eye(3)[None],
            atol=2e-4,
        ):
            raise ValueError("Crop calibration transforms are not mutually inverse")
        if not np.allclose(
            data["K_crop"], data["image_from_full"] @ data["K_full"], atol=2e-4
        ):
            raise ValueError("K_crop does not preserve full-image calibration")
        if not np.array_equal(data["uv_vertex_remap"][data["uv_faces"]], data["faces"]):
            raise ValueError("UV topology no longer maps exactly to source faces")
        if not np.isfinite(data["vertices_world"]).all():
            raise ValueError("World mesh contains non-finite values")
        frame_files = [str(item) for item in data["frame_files"].tolist()]

    expected_hashes = manifest["files"]["frame_sha256"]
    if len(frame_files) != len(expected_hashes):
        raise ValueError("Frame hash list length mismatch")
    for relative_path, expected_hash in zip(frame_files, expected_hashes):
        frame_path = dataset_dir / relative_path
        if not frame_path.is_file():
            raise ValueError(f"Missing frame {relative_path}")
        with Image.open(frame_path) as image:
            if image.size != tuple(manifest["images"]["stored_size"]):
                raise ValueError(f"Bad image size for {relative_path}: {image.size}")
        if sha256(frame_path) != expected_hash:
            raise ValueError(f"Frame hash mismatch for {relative_path}")
    if sha256(npz_path) != manifest["files"]["dataset_npz_sha256"]:
        raise ValueError("dataset.npz hash mismatch")

    if archive_path is not None:
        with zipfile.ZipFile(archive_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Archive CRC failure: {bad_member}")
            names = set(archive.namelist())
            expected_names = {"manifest.json", "dataset.npz", *frame_files}
            if names != expected_names:
                raise ValueError(
                    f"Archive members differ: missing={expected_names - names}, extra={names - expected_names}"
                )

    return {
        "frames": len(frame_files),
        "dataset_bytes": npz_path.stat().st_size,
        "frames_bytes": sum((dataset_dir / path).stat().st_size for path in frame_files),
        "archive_bytes": archive_path.stat().st_size if archive_path else None,
    }


def build(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    root = args.root.resolve()
    video = args.video.resolve()
    transforms_path = args.transforms.resolve()
    vertices_path = args.vertices.resolve()
    faces_path = args.faces.resolve()
    clean_mesh_path = args.clean_mesh.resolve()
    output_dir = args.output.resolve()
    archive_path = args.archive.resolve()
    input_paths = (video, transforms_path, vertices_path, faces_path, clean_mesh_path)
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    if archive_path.exists():
        raise SystemExit(f"Archive already exists: {archive_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        video_indices = np.arange(0, 358, 3, dtype=np.int32)
        if len(video_indices) != 120 or video_indices[-1] != 357:
            raise AssertionError("Internal sampling error")
        info = video_info(video, args.ffprobe)
        full_width, full_height = int(info["width"]), int(info["height"])
        if (full_width, full_height) != (1280, 720):
            raise ValueError(f"v03 must be 1280x720, got {full_width}x{full_height}")
        if info.get("nb_frames") not in (None, "N/A") and int(info["nb_frames"]) <= 357:
            raise ValueError(f"v03 has only {info['nb_frames']} frames")

        c2w, K_single, transforms = load_camera_data(transforms_path, video_indices)
        if (int(transforms["w"]), int(transforms["h"])) != (full_width, full_height):
            raise ValueError("transforms.json calibration size differs from the reference video")
        mesh = load_and_denormalize_mesh(
            vertices_path, faces_path, clean_mesh_path, args.unitex_scale
        )
        uv = repack_uvs(
            mesh["vertices_normalized"],
            mesh["faces"],
            args.atlas_size,
            args.atlas_padding,
            args.texels_per_unit,
            args.xatlas_input_scale,
        )
        projected_bbox, w2c, w2c_opencv, projection_full = project_mesh_bbox(
            mesh["vertices_world"], c2w, K_single
        )
        crops = choose_square_crops(projected_bbox, full_width, full_height, args.margin)
        image_from_full, full_from_image, K_crop = crop_transforms(
            crops, args.output_resolution, K_single
        )
        projection_crop = image_from_full @ projection_full

        frame_files, sharpness, frame_hashes = extract_and_crop_frames(
            video,
            staging / "frames",
            video_indices,
            crops,
            full_width,
            full_height,
            args.output_resolution,
            args.jpeg_quality,
            args.ffmpeg,
        )
        K_full = np.repeat(K_single[None], len(video_indices), axis=0)
        arrays: dict[str, np.ndarray] = {
            "video_frame_indices": video_indices,
            "blender_frame_numbers": video_indices + 1,
            "frame_files": np.asarray(frame_files),
            "c2w": c2w.astype(np.float32),
            "w2c": w2c.astype(np.float32),
            "w2c_opencv": w2c_opencv.astype(np.float32),
            "blender_to_opencv": BLENDER_TO_OPENCV.astype(np.float32),
            "K_full": K_full.astype(np.float32),
            "K_crop": K_crop.astype(np.float32),
            "crop_xywh": crops,
            "image_from_full": image_from_full.astype(np.float32),
            "full_from_image": full_from_image.astype(np.float32),
            "projection_full": projection_full.astype(np.float32),
            "projection_crop": projection_crop.astype(np.float32),
            "projected_bbox_full": projected_bbox.astype(np.float32),
            "sharpness_laplacian_var": sharpness,
            "vertices_normalized": mesh["vertices_normalized"],
            "vertices_world": mesh["vertices_world"],
            "vertex_normals_world": mesh["vertex_normals_world"],
            "faces": mesh["faces"],
            "uv_vertex_remap": uv["uv_vertex_remap"],
            "uv_faces": uv["uv_faces"],
            "uv_coords": uv["uv_coords"],
            "normalized_to_world": mesh["normalized_to_world"],
            "world_to_normalized": mesh["world_to_normalized"],
        }
        npz_path = staging / "dataset.npz"
        save_npz(npz_path, arrays)

        input_hashes = {
            relative_or_absolute(path, root): sha256(path) for path in input_paths
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "description": "Seedance v03 frames 0,3,...,357 aligned to Blender frames n+1",
            "sources": {
                "video": relative_or_absolute(video, root),
                "transforms": relative_or_absolute(transforms_path, root),
                "vertices": relative_or_absolute(vertices_path, root),
                "faces": relative_or_absolute(faces_path, root),
                "clean_mesh_for_denormalization": relative_or_absolute(clean_mesh_path, root),
                "sha256": input_hashes,
                "video_probe": info,
            },
            "sampling": {
                "video_frame_indices": "0,3,...,357 (inclusive; zero-based)",
                "count": len(video_indices),
                "pose_rule": "Blender frame = zero-based video frame + 1",
            },
            "images": {
                "full_size": [full_width, full_height],
                "stored_size": [args.output_resolution, args.output_resolution],
                "format": "JPEG RGB",
                "jpeg_quality": args.jpeg_quality,
                "jpeg_subsampling": 0,
                "crop": "square projected-mesh bbox crop, shifted inside image when possible",
                "margin_each_side_fraction_of_max_bbox_dimension": args.margin,
                "resize": "Pillow LANCZOS with pixel-center transform stored in image_from_full",
                "target_masks": None,
            },
            "cameras": {
                "c2w": "Blender/OpenGL camera-to-world; local +X right, +Y up, -Z forward",
                "w2c": "inverse(c2w), same Blender/OpenGL camera axes",
                "w2c_opencv": "blender_to_opencv @ w2c; +X right, +Y down, +Z forward",
                "K_full_and_K_crop": "OpenCV pixel intrinsics; use with w2c_opencv",
                "distortion": {
                    key: float(transforms.get(key, 0.0)) for key in ("k1", "k2", "p1", "p2")
                },
                "calibration_note": "Full-image K and both crop homographies are retained in dataset.npz",
            },
            "mesh": {
                "coordinate_system": "Blender world axes: +X right, +Y forward/length, +Z up",
                "vertex_count": int(len(mesh["vertices_world"])),
                "triangle_count": int(len(mesh["faces"])),
                "unitex_preprocess_scale": args.unitex_scale,
                "clean_mesh_bounds_gltf": mesh["bounds_gltf"].tolist(),
                "clean_mesh_center_gltf": mesh["center_gltf"].tolist(),
                "clean_mesh_center_blender": mesh["center_blender"].tolist(),
                "denormalization_scale": float(mesh["denormalization_scale"]),
                "formula": "vertices_world = vertices_normalized * denormalization_scale + clean_mesh_center_blender",
                "normalized_bounds_crosscheck_max_abs_error": float(mesh["bounds_error"]),
            },
            "uv": {
                "library": "xatlas",
                "library_version": getattr(xatlas, "__version__", "0.0.11"),
                "atlas_size": [uv["atlas_width"], uv["atlas_height"]],
                "padding_pixels": args.atlas_padding,
                "requested_texels_per_unit": args.texels_per_unit,
                "actual_texels_per_unit": uv["actual_texels_per_unit"],
                "xatlas_input_scale": uv["input_scale"],
                "raw_xatlas_texels_per_unit": uv["raw_xatlas_texels_per_unit"],
                "chart_count": uv["chart_count"],
                "utilization": uv["utilization"],
                "convention": "raw normalized xatlas UVs; u right, v up",
                "topology": "vertices_world[uv_vertex_remap] with uv_faces; uv_vertex_remap[uv_faces] == faces",
            },
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
            "files": {
                "dataset_npz": "dataset.npz",
                "dataset_npz_sha256": sha256(npz_path),
                "frames": frame_files,
                "frame_sha256": frame_hashes,
            },
            "notes": [
                "Frames contain their original city background; no target mattes are bundled.",
                "projected_bbox_full is geometric and preserves known camera calibration; Seedance content may drift within the margin.",
                "sharpness_laplacian_var can be used for frame rejection or weighting.",
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        validation = validate_dataset(staging)
        os.replace(staging, output_dir)
        make_archive(output_dir, archive_path)
        validation = validate_dataset(output_dir, archive_path)
        return output_dir, archive_path, validation
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parser() -> argparse.ArgumentParser:
    script_root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=script_root)
    result.add_argument(
        "--video", type=Path, default=script_root / "after_effects" / "seedance_v03.mp4"
    )
    result.add_argument(
        "--transforms", type=Path, default=script_root / "brush_dataset" / "transforms.json"
    )
    result.add_argument(
        "--vertices", type=Path, default=script_root / "unitex_result5" / "mesh_V.npy"
    )
    result.add_argument(
        "--faces", type=Path, default=script_root / "unitex_result5" / "mesh_FV.npy"
    )
    result.add_argument("--clean-mesh", type=Path, default=script_root / "ferrari_clean2.glb")
    result.add_argument("--output", type=Path, default=script_root / "video_fit" / "dataset_v03")
    result.add_argument(
        "--archive", type=Path, default=script_root / "video_fit" / "dataset_v03.zip"
    )
    result.add_argument("--output-resolution", type=int, default=512)
    result.add_argument("--margin", type=float, default=0.15)
    result.add_argument("--jpeg-quality", type=int, default=95)
    result.add_argument("--atlas-size", type=int, default=4096)
    result.add_argument("--atlas-padding", type=int, default=8)
    result.add_argument(
        "--texels-per-unit",
        type=float,
        default=600.0,
        help="Mesh-specific xatlas density that yields one exact 4096x4096 atlas",
    )
    result.add_argument(
        "--xatlas-input-scale",
        type=float,
        default=100.0,
        help="Uniform epsilon workaround; density is compensated, world vertices are unchanged",
    )
    result.add_argument("--unitex-scale", type=float, default=0.95)
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--ffprobe", default="ffprobe")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output_resolution <= 0:
        raise SystemExit("--output-resolution must be positive")
    if not 0.0 <= args.margin <= 1.0:
        raise SystemExit("--margin must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")
    if args.atlas_size != 4096 or args.atlas_padding != 8:
        raise SystemExit("This fit dataset requires --atlas-size 4096 --atlas-padding 8")
    if args.texels_per_unit <= 0.0 or args.xatlas_input_scale <= 0.0:
        raise SystemExit("xatlas density and input scale must be positive")
    output_dir, archive_path, validation = build(args)
    total_mb = validation["archive_bytes"] / (1024 * 1024)
    print(f"OK: {validation['frames']} aligned frames")
    print(f"Dataset directory: {output_dir}")
    print(f"Runner archive:    {archive_path} ({total_mb:.1f} MiB)")
    print("Validation: hashes, shapes, camera inverses, crop calibration, UV topology and ZIP CRC passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
