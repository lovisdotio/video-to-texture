"""Augment an HQ v2 fitting dataset with lossless high-resolution target mattes.

This preserves the unmasked dataset and creates a separate ``*_masked`` dataset.
The matte video may be 720p, 1080p or 2160p as long as it has the same 16:9 crop and
24 fps frame order; crop coordinates are scaled exactly into its pixel grid.  For
production, use the restored 2160p source through BiRefNet rather than upscaling the
old 720p mask.

Run with the project environment:
    tools/venv/bin/python video_fit/hq_v2/add_masks_hq.py
"""

from __future__ import annotations

import hashlib
import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset_v03_hq"
TARGET = ROOT / "dataset_v03_hq_masked"
ARCHIVE = ROOT / "dataset_v03_hq_masked.zip"
MASK_VIDEO = ROOT / "source" / "seedance_v03_restored_2160p_mask.mp4"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def probe_video(video: Path, ffprobe: str) -> tuple[int, int, float, int | None]:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(video),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    frame_rate = float(Fraction(str(stream["avg_frame_rate"])))
    frame_count = None if stream.get("nb_frames") in (None, "N/A") else int(stream["nb_frames"])
    return int(stream["width"]), int(stream["height"]), frame_rate, frame_count


def extract_masks(
    mask_video: Path,
    crops: np.ndarray,
    video_indices: np.ndarray,
    source_size: tuple[int, int],
    output_size: int,
    destination: Path,
    ffmpeg: str,
    ffprobe: str,
) -> list[str]:
    mask_width, mask_height, mask_fps, mask_frame_count = probe_video(mask_video, ffprobe)
    source_width, source_height = source_size
    scale_x, scale_y = mask_width / source_width, mask_height / source_height
    if abs(scale_x - scale_y) > 1e-6:
        raise RuntimeError(
            f"Mask/source aspect mismatch: mask={mask_width}x{mask_height}, "
            f"source={source_width}x{source_height}"
        )
    if len(video_indices) < 2 or not np.all(np.diff(video_indices) == np.diff(video_indices)[0]):
        raise RuntimeError("Dataset indices must form one arithmetic progression")
    start, stop = int(video_indices[0]), int(video_indices[-1])
    step = int(video_indices[1] - video_indices[0])
    if abs(mask_fps - 24.0) > 1e-3:
        raise RuntimeError(f"Mask video is {mask_fps:g} fps, expected 24 fps")
    if mask_frame_count is not None and mask_frame_count <= stop:
        raise RuntimeError(f"Mask video has {mask_frame_count} frames, needs frame {stop}")
    frame_bytes = mask_width * mask_height
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mask_video),
        "-vf",
        f"select=gte(n\\,{start})*lte(n\\,{stop})*not(mod(n-{start}\\,{step}))",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    destination.mkdir(parents=True)
    files: list[str] = []
    invert: bool | None = None
    try:
        for index, (x0, y0, width, height) in enumerate(crops.tolist()):
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                error = process.stderr.read().decode(errors="replace")
                raise RuntimeError(f"Short mask frame {index}: {len(raw)} bytes; {error}")
            full_array = np.frombuffer(raw, np.uint8).reshape(mask_height, mask_width)
            if invert is None:
                border = np.concatenate(
                    [
                        full_array[:16].ravel(),
                        full_array[-16:].ravel(),
                        full_array[:, :16].ravel(),
                        full_array[:, -16:].ravel(),
                    ]
                )
                invert = float(np.median(border)) > 127.0
            if invert:
                full_array = 255 - full_array
            full = Image.fromarray(full_array, mode="L")
            mx0, my0 = x0 * scale_x, y0 * scale_y
            mx1, my1 = (x0 + width) * scale_x, (y0 + height) * scale_y
            cropped = full.crop((mx0, my0, mx1, my1)).resize(
                (output_size, output_size), Image.Resampling.LANCZOS
            )
            relative = f"masks/{index:04d}.png"
            cropped.save(destination.parent / relative, format="PNG", optimize=True)
            files.append(relative)
        extra = process.stdout.read(1)
        error = process.stderr.read().decode(errors="replace")
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg mask extraction failed ({code}): {error}")
        if extra:
            raise RuntimeError("Mask selector returned more than 120 frames")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return files


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=SOURCE)
    result.add_argument("--target", type=Path, default=TARGET)
    result.add_argument("--archive", type=Path, default=ARCHIVE)
    result.add_argument("--mask-video", type=Path, default=MASK_VIDEO)
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--ffprobe", default="ffprobe")
    return result


def main() -> None:
    args = parser().parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    archive_path = args.archive.resolve()
    mask_video = args.mask_video.resolve()
    if not source.is_dir() or not mask_video.is_file():
        raise SystemExit(f"Missing {source} or {mask_video}")
    if target.exists() or archive_path.exists():
        raise SystemExit(f"Refusing to overwrite existing {target} or {archive_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=".dataset_v03_masked-", dir=ROOT))
    try:
        shutil.copytree(source, staging / "dataset")
        dataset = staging / "dataset"
        manifest = json.loads((dataset / "manifest.json").read_text())
        if not str(manifest.get("schema_version", "")).startswith("car-city-video-fit-dataset-v2-hq"):
            raise RuntimeError(f"Not an HQ v2 dataset: {manifest.get('schema_version')}")
        source_size = tuple(int(value) for value in manifest["images"]["full_size"])
        stored_size = tuple(int(value) for value in manifest["images"]["stored_size"])
        if len(source_size) != 2 or stored_size[0] != stored_size[1]:
            raise RuntimeError("Invalid source/stored image dimensions in manifest")
        with np.load(dataset / "dataset.npz", allow_pickle=False) as arrays:
            crops = arrays["crop_xywh"]
            indices = arrays["video_frame_indices"]
        mask_files = extract_masks(
            mask_video,
            crops,
            indices,
            source_size,
            stored_size[0],
            dataset / "masks",
            args.ffmpeg,
            args.ffprobe,
        )

        # Sanity: BiRefNet should cover a meaningful but minority fraction of each crop.
        coverage = []
        for relative in mask_files:
            mask = np.asarray(Image.open(dataset / relative), np.float32) / 255.0
            coverage.append(float((mask > 0.5).mean()))
        if min(coverage) < 0.03 or max(coverage) > 0.90:
            raise RuntimeError(f"Implausible mask coverage range: {min(coverage):.3f}..{max(coverage):.3f}")

        manifest_path = dataset / "manifest.json"
        manifest["schema_version"] = "car-city-video-fit-dataset-v2-hq-masked"
        manifest["images"]["target_masks"] = {
            "format": "PNG grayscale soft alpha",
            "files": mask_files,
            "source": str(mask_video),
            "method": "BiRefNet HQ video matte; soft PNG alpha retained losslessly",
            "coverage_min": min(coverage),
            "coverage_median": float(np.median(coverage)),
            "coverage_max": max(coverage),
        }
        manifest["files"]["masks"] = mask_files
        manifest["files"]["mask_sha256"] = [digest(dataset / path) for path in mask_files]
        manifest["notes"] = [
            note for note in manifest["notes"] if "no target mattes" not in note.lower()
        ] + ["BiRefNet target mattes are soft; erode/intersect with rendered silhouette for RGB loss."]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        with zipfile.ZipFile(
            staging / "dataset_v03_masked.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(dataset.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(dataset).as_posix())
        with zipfile.ZipFile(staging / "dataset_v03_masked.zip") as archive:
            expected_members = 2 + 2 * len(indices)
            if archive.testzip() is not None or len(archive.namelist()) != expected_members:
                raise RuntimeError("Masked archive validation failed")

        # `--target` and `--archive` are useful for validation on another volume;
        # shutil.move keeps that supported while the production defaults remain
        # atomic same-volume renames in practice.
        shutil.move(str(dataset), str(target))
        shutil.move(str(staging / "dataset_v03_masked.zip"), str(archive_path))
        print(
            f"MASKED_DATASET_READY {archive_path} coverage="
            f"{min(coverage):.3f}/{np.median(coverage):.3f}/{max(coverage):.3f}",
            flush=True,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
