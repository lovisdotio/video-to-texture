#!/usr/bin/env python3
"""Blender stage: strict seven-image V11 full-bleed replacement and staging save."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR)) if str(SCRIPT_DIR) not in sys.path else None
from targeted_final_v11_blender import integrate  # noqa: E402
from targeted_final_v11_common import ASSEMBLY_SCHEMA, SOURCE_BLEND, SOURCE_PINS, WORK_ROOT, load_texture_set, require, sha256  # noqa: E402

def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--staged-blend", type=Path, required=True); parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv); staged = args.staged_blend.expanduser().resolve(); report_path = args.report.expanduser().resolve()
    for path in (staged, report_path):
        try: path.relative_to(WORK_ROOT.resolve())
        except ValueError as exc: raise RuntimeError(f"V11 staging path escapes work root: {path}") from exc
    require(staged.suffix == ".blend" and staged.parent == report_path.parent and staged.parent.is_dir(), "Invalid V11 staging layout")
    require(not staged.exists() and not report_path.exists(), "Refusing to overwrite V11 staging")
    require(Path(bpy.data.filepath).resolve() == SOURCE_BLEND, "Blender did not open pinned V10 source")
    textures = load_texture_set(); report = integrate(textures)
    require(report.get("schema") == ASSEMBLY_SCHEMA and report.get("status") == "passed" and all(report.get("hard_gates", {}).values()), "V11 in-memory assembly failed")
    report.update({"blender_version": bpy.app.version_string, "staged_blend": str(staged), "assembly_report": str(report_path), "saved": False, "rendered": False, "remote_generation_called": False})
    bpy.ops.wm.save_as_mainfile(filepath=str(staged), compress=True, relative_remap=False)
    require(Path(bpy.data.filepath).resolve() == staged and staged.is_file(), "V11 staging save failed")
    report["saved"] = True; report["staged_blend_sha256"] = sha256(staged); report["source_v10_sha256_after_save"] = sha256(SOURCE_BLEND)
    require(report["source_v10_sha256_after_save"] == SOURCE_PINS[SOURCE_BLEND], "Pinned V10 changed during V11 save")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TARGETED_FINAL_V11_STAGE=" + json.dumps({"status":"passed","staged_blend":str(staged),"sha256":report["staged_blend_sha256"],"meshes":report["state_after"]["meshes"],"triangles":report["state_after"]["triangles"],"image_bindings":len(report["v11_bindings"]["nodes"])}, sort_keys=True))

if __name__ == "__main__": main()
