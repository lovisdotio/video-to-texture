"""Augment the v03 fitting dataset with BiRefNet target mattes.

This preserves the unmasked dataset and creates ``dataset_v03_masked/`` plus
``dataset_v03_masked.zip``. The mask video must retain the 24 fps frame order.

Run with the project environment:
    tools/venv/bin/python video_fit/add_masks_to_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset_v03"
TARGET = ROOT / "dataset_v03_masked"
ARCHIVE = ROOT / "dataset_v03_masked.zip"
MASK_VIDEO = ROOT / "v03_first15_mask.mp4"
FULL_WIDTH = 1280
FULL_HEIGHT = 720
OUTPUT_SIZE = 512


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def extract_masks(mask_video: Path, crops: np.ndarray, destination: Path) -> list[str]:
    frame_bytes = FULL_WIDTH * FULL_HEIGHT
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mask_video),
        "-vf",
        "select=not(mod(n\\,3))*lte(n\\,357)",
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
            full_array = np.frombuffer(raw, np.uint8).reshape(FULL_HEIGHT, FULL_WIDTH)
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
            cropped = full.crop((x0, y0, x0 + width, y0 + height)).resize(
                (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
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


def main() -> None:
    if not SOURCE.is_dir() or not MASK_VIDEO.is_file():
        raise SystemExit(f"Missing {SOURCE} or {MASK_VIDEO}")
    if TARGET.exists() or ARCHIVE.exists():
        raise SystemExit(f"Refusing to overwrite existing {TARGET} or {ARCHIVE}")

    staging = Path(tempfile.mkdtemp(prefix=".dataset_v03_masked-", dir=ROOT))
    try:
        shutil.copytree(SOURCE, staging / "dataset")
        dataset = staging / "dataset"
        with np.load(dataset / "dataset.npz", allow_pickle=False) as arrays:
            crops = arrays["crop_xywh"]
            indices = arrays["video_frame_indices"]
        if not np.array_equal(indices, np.arange(0, 358, 3, dtype=np.int32)):
            raise RuntimeError("Dataset frames are not 0,3,...,357")
        mask_files = extract_masks(MASK_VIDEO, crops, dataset / "masks")

        # Sanity: BiRefNet should cover a meaningful but minority fraction of each crop.
        coverage = []
        for relative in mask_files:
            mask = np.asarray(Image.open(dataset / relative), np.float32) / 255.0
            coverage.append(float((mask > 0.5).mean()))
        if min(coverage) < 0.03 or max(coverage) > 0.90:
            raise RuntimeError(f"Implausible mask coverage range: {min(coverage):.3f}..{max(coverage):.3f}")

        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = "car-city-video-fit-dataset-v1-masked"
        manifest["images"]["target_masks"] = {
            "format": "PNG grayscale soft alpha",
            "files": mask_files,
            "source": "fal-ai/birefnet/v2/video, General Use (Light 2K), 2048 operating resolution",
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
            if archive.testzip() is not None or len(archive.namelist()) != 242:
                raise RuntimeError("Masked archive validation failed")

        dataset.rename(TARGET)
        (staging / "dataset_v03_masked.zip").replace(ARCHIVE)
        print(
            f"MASKED_DATASET_READY {ARCHIVE} coverage="
            f"{min(coverage):.3f}/{np.median(coverage):.3f}/{max(coverage):.3f}",
            flush=True,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
