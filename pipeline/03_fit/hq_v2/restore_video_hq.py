"""Resumable HQ restoration request for the 358-frame fitting window.

This file performs no work when imported.  A new paid request is impossible unless
``--submit`` is passed explicitly.  SeedVR2 is the conservative default because it
offers target-resolution control and temporal consistency.  Topaz Starlight Precise
2.5 is available as an alternate restoration-oriented backend for AI video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from fal import apps
from fal.toolkit import File


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "seedance_v03_first15_native.mp4"
OUTPUT = ROOT / "source" / "seedance_v03_restored_2160p.mp4"
STATE = ROOT / "restore_video_hq_state.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=SOURCE)
    result.add_argument("--output", type=Path, default=OUTPUT)
    result.add_argument("--state", type=Path, default=STATE)
    result.add_argument("--backend", choices=("seedvr", "topaz"), default="seedvr")
    result.add_argument("--submit", action="store_true", help="authorize one new paid request")
    return result


def write_state(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def request_spec(backend: str, video_url: str) -> tuple[str, dict]:
    if backend == "seedvr":
        return (
            "fal-ai/seedvr/upscale/video",
            {
                "video_url": video_url,
                "upscale_mode": "target",
                "target_resolution": "2160p",
                # Match the user-approved six-second A/B exactly: the lower
                # setting favors temporal fidelity over invented micro-detail.
                "noise_scale": 0.05,
                "seed": 63,
                "output_format": "X264 (.mp4)",
                "output_quality": "maximum",
                "output_write_mode": "balanced",
            },
        )
    return (
        "fal-ai/topaz/upscale/video",
        {
            "video_url": video_url,
            "model": "Starlight Precise 2.5",
            "upscale_factor": 3.0,
            "recover_detail": 0.8,
            "compression": 0.25,
            "noise": 0.10,
            "halo": 0.05,
            "grain": 0.0,
            "H264_output": False,
        },
    )


def main() -> None:
    args = parser().parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    state_path = args.state.resolve()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    if "request_id" not in state:
        if not args.submit:
            raise SystemExit(
                f"No existing request in {state_path}. Re-run with --submit to authorize one."
            )
        if not source.is_file():
            raise SystemExit(f"Missing 358-frame native fitting window: {source}")
        video_url = File.from_path(source, content_type="video/mp4").url
        endpoint, payload = request_spec(args.backend, video_url)
        handle = apps.submit(endpoint, payload)
        state = {
            "backend": args.backend,
            "endpoint": endpoint,
            "request_id": handle.request_id,
            "input_url": video_url,
            "input_path": str(source),
            "payload": payload,
        }
        write_state(state_path, state)
        print(f"submitted {endpoint}: {handle.request_id}", flush=True)
    else:
        if state.get("backend") != args.backend:
            raise SystemExit(
                f"State belongs to backend {state.get('backend')!r}, not {args.backend!r}"
            )
        handle = apps.RequestHandle(state["endpoint"], state["request_id"])
        print(f"resuming {state['endpoint']}: {handle.request_id}", flush=True)

    if "result" not in state:
        state["result"] = handle.get()
        write_state(state_path, state)
    video = state["result"].get("video")
    if not video or not video.get("url"):
        raise RuntimeError(f"Restoration returned no video: {state['result']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or not output.stat().st_size:
        with httpx.stream("GET", video["url"], timeout=600, follow_redirects=True) as response:
            response.raise_for_status()
            with output.open("wb") as destination:
                for chunk in response.iter_bytes():
                    destination.write(chunk)
    print(f"RESTORED_VIDEO_READY {output} ({output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
