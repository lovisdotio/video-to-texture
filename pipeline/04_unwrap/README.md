# 04 · Re-unwrap into clean UV islands, rebake per material atlas

The step-03 fit works on one auto-generated atlas over a joined proxy mesh: thousands of
tiny charts, one material for glass, tyres and paint alike. This step goes back to the
original multi-part model, unwraps it into clean islands grouped by material page, and
re-projects the video into those pages.

- `ARCHITECTURE.md` – the v8 design: exact per-object proxies (weld/dedup with a full
  source→proxy face/corner map), hard-surface seams (UVgami at 66°), density weighted by
  video visibility, 16 px gutters, and the gates every unwrap must pass.
- `build_nonbody_proxies_v8.py`, `proxy_core_v8.py`, `contract_v8.py` – proxy build and
  pinned contract.
- `unwrap_pack_nonbody_v8_blender.py` – headless Blender unwrap + pack of the non-body
  parts into material pages (exterior, glass, lights, running metal, tyres, cockpit).
- `bake_v8/` – six-page inverse-projection baker on a fal H100 (full + opaque z-buffers so
  glass and what is behind it each get their own observations).
- `targeted_reunwrap_v9/` – re-unwraps only the 13 pieces whose charts were still too
  fragmented (e.g. wipers: 65 → 29 charts) and places them without moving the rest.
- `targeted_rebake_v9/` – rebakes the touched pages (`reproject_page_texture_v9.py`).

```bash
blender --background --factory-startup \
  --python targeted_reunwrap_v9/targeted_unwrap_v9_blender.py -- \
  --selectors '49:-1' --output-dir runs/smoke --methods MINIMUM_STRETCH --seam-modes v8
```

**Research snapshot.** These scripts are hash-pinned to our intermediate datasets
(not shipped) and import a few sibling modules from the original workspace that are not
part of this repo. Read them as a reference implementation of the method, not as a
turnkey tool.
