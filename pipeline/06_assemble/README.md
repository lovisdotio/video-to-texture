# 06 · Put the textures back on the mesh

Texture-only integration in Blender: starting from the audited step-04 scene, swap the
packed atlas images bound to the material image nodes. Geometry, UVs, normals, material
slots, shader structure, camera and world are never touched.

- `build_targeted_final_v11.py` – preflight, build in an isolated staging file, reopen
  audit, no-overwrite promotion.
- `assemble_targeted_final_v11_blender.py`, `targeted_final_v11_blender.py`,
  `targeted_final_v11_common.py` – the Blender side (v11 = full-bleed textures from
  step 05); `targeted_final_v10_*` are the base modules v11 extends.
- `render_targeted_final_v11_preview_blender.py` – five read-only "around" views.
- `NOTES_v10.md` – original run notes.

```bash
python3 build_targeted_final_v11.py --preflight-only
python3 build_targeted_final_v11.py --output-name ferrari_final_v11.blend
```

Same research-snapshot caveat as step 04.
