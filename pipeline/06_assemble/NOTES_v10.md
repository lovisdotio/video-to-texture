# Targeted final V10 — texture-only integration

This additive stage starts only from the pinned, audited V9 final blend and swaps
exactly seven packed atlas image datablocks across exactly 32 existing image-node
bindings. It never edits geometry, topology, UVs, normals, transforms, material
slots, shader structure/sampling, the production camera, or the production world.

Pure-Python preflight (does not launch Blender):

```sh
python3 \
  <unwrap-root>/targeted_final_v10/build_targeted_final_v10.py \
  --preflight-only
```

Build, isolated staging save, reopen audit, and no-overwrite promotion:

```sh
python3 \
  <unwrap-root>/targeted_final_v10/build_targeted_final_v10.py \
  --output-name ferrari_targeted_reunwrap_seedream_v10_final.blend
```

Five read-only around views after a successful build:

```sh
blender -b \
  <unwrap-root>/targeted_final_v10/outputs/ferrari_targeted_reunwrap_seedream_v10_final.blend \
  --python <unwrap-root>/targeted_final_v10/render_targeted_final_v10_preview_blender.py -- \
  --output-dir <unwrap-root>/targeted_final_v10/outputs/renders/ferrari_targeted_reunwrap_seedream_v10_around5 \
  --build-report <unwrap-root>/targeted_final_v10/outputs/ferrari_targeted_reunwrap_seedream_v10_final.build_report.json \
  --view-set around5 --width 1920 --height 1080
```

Every final artifact and render directory is fail-closed and refuses overwrite.
