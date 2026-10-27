#!/usr/bin/env python3
"""Step 2: clay frame -> styled reference image -> styled orbit video, on fal.

    export FAL_KEY=...            # your fal API key, never hard-code it
    pip install fal-client
    python generate.py reference --frame renders/clay_frames/frame_0001.png
    python generate.py video --clay clay_orbit_720p24.mp4 --reference seedream.png

Both calls are paid. Seedance's input video is billed too.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import fal_client

HERE = Path(__file__).resolve().parent
SEEDREAM = "bytedance/seedream/v5/pro/edit"
SEEDANCE = "bytedance/seedance-2.5/reference-to-video"


def _download(url: str, out: Path) -> None:
    import urllib.request

    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)
    print("saved", out)


def reference(args: argparse.Namespace) -> None:
    """Seedream 5 Pro edit: turn clay frame 1 into the photoreal look reference."""
    prompt = (HERE / "seedream_prompt.txt").read_text().strip()
    result = fal_client.subscribe(
        SEEDREAM,
        arguments={
            "prompt": prompt,
            "image_urls": [fal_client.upload_file(args.frame)],
            "image_size": "landscape_16_9",
            "output_format": "png",
        },
        with_logs=True,
    )
    for i, image in enumerate(result["images"]):
        _download(image["url"], Path(args.out) / f"seedream_{i}.png")


def video(args: argparse.Namespace) -> None:
    """Seedance 2.5 reference-to-video: same camera move, new look."""
    prompt = (HERE / "seedance_prompt.txt").read_text().strip()
    result = fal_client.subscribe(
        SEEDANCE,
        arguments={
            "prompt": prompt,
            "video_urls": [fal_client.upload_file(args.clay)],
            "image_urls": [fal_client.upload_file(args.reference)],
            "duration": "30",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "generate_audio": False,
        },
        with_logs=True,
    )
    video = result.get("video")
    if not video:
        # Seedance can complete without a video when its classifier rejects the
        # prompt as "video editing". See README for the wording that avoids it.
        raise SystemExit(f"No video returned: {result}")
    _download(video["url"], Path(args.out) / "seedance.mp4")


def main() -> None:
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("Set FAL_KEY in the environment first.")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--frame", required=True, help="clay render of frame 1")
    r.add_argument("--out", default="out")
    v = sub.add_parser("video")
    v.add_argument("--clay", required=True, help="clay orbit mp4 (720p, 24 fps, 30 s)")
    v.add_argument("--reference", required=True, help="validated Seedream reference image")
    v.add_argument("--out", default="out")
    args = parser.parse_args()
    {"reference": reference, "video": video}[args.cmd](args)


if __name__ == "__main__":
    main()
