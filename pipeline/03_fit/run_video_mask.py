"""Run BiRefNet once on the 15 s fitting window and download its mask video.

The script is resumable and stores the fal request/result in ``mask_video_state.json``.
Run it with the authenticated fal CLI environment:

    python video_fit/run_video_mask.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from fal import apps
from fal.toolkit import File


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "v03_first15.mp4"
STATE = ROOT / "mask_video_state.json"
OUTPUT = ROOT / "v03_first15_mask.mp4"
ENDPOINT = "fal-ai/birefnet/v2/video"


def write_state(value: dict) -> None:
    STATE.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "request_id" not in state:
        video_url = File.from_path(INPUT, content_type="video/mp4").url
        handle = apps.submit(
            ENDPOINT,
            {
                "video_url": video_url,
                "model": "General Use (Light 2K)",
                "operating_resolution": "2048x2048",
                "output_mask": True,
                "refine_foreground": True,
                "video_output_type": "X264 (.mp4)",
                "video_quality": "maximum",
                "video_write_mode": "balanced",
            },
        )
        state = {"endpoint": ENDPOINT, "request_id": handle.request_id, "input_url": video_url}
        write_state(state)
        print(f"submitted: {handle.request_id}", flush=True)
    else:
        handle = apps.RequestHandle(ENDPOINT, state["request_id"])
        print(f"resuming: {handle.request_id}", flush=True)

    if "result" not in state:
        state["result"] = handle.get()
        write_state(state)
    mask = state["result"].get("mask_video")
    if not mask or not mask.get("url"):
        raise RuntimeError(f"BiRefNet did not return mask_video: {state['result']}")

    if not OUTPUT.exists() or not OUTPUT.stat().st_size:
        with httpx.stream("GET", mask["url"], timeout=300, follow_redirects=True) as response:
            response.raise_for_status()
            with OUTPUT.open("wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
    print(f"MASK_VIDEO_READY {OUTPUT} ({OUTPUT.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
