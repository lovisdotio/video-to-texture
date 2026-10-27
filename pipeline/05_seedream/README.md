# 05 · Fill seams and unseen areas with Seedream, upscale with Topaz

A video never sees every texel: under the car, inside wheel arches, seam gutters. Each
material atlas is completed by an image-edit model, then upscaled to its native size.

| Script | What |
|---|---|
| `prepare_seedream_inputs_v10.py` | per page: pre-guttered working atlas, sparse occupied-only layout, exact binary mask, pinned renders as identity references |
| `run_seedream_v10.py` (+ `targeted_seedream_v10_common.py`) | nine 2048² `bytedance/seedream/v5/pro/edit` requests (body 8K, exterior, glass, lights, running metal, tyres, cockpit leather / plastics / carbon), journaled and resumable |
| `run_topaz_v10.py` | `fal-ai/topaz/upscale/image` (High Fidelity V2) back to each page's native 4K / 8K |
| `composite_native_v10.py` | keeps every video-observed texel, takes generated pixels only inside the mask |
| `build_full_bleed_v11.py` | local: fills every empty texel from its nearest occupied donor so no black shows at mip levels |
| `NOTES_v10.md` | original run notes |

Prompt rules that mattered: ask for **neutral albedo** (no baked highlights, reflections,
shadows, AO, glow, scenery or black void); describe the renders as *identity* references,
never as lighting or colour references; give cockpit materials their own explicit
descriptions.

```bash
python run_seedream_v10.py --emit-preflight preflight.json   # offline
export FAL_KEY=...
python run_seedream_v10.py --execute --go-token SEEDREAM_V10_NETWORK_GO --run-id my_run --max-workers 3
python run_topaz_v10.py --seedream-report runs/my_run/seedream_v10_results_manifest.json \
  --execute --go-token TOPAZ_V10_NETWORK_GO
python composite_native_v10.py && python build_full_bleed_v11.py
```

Network calls need both `--execute` and the go token on purpose: the default is a dry
run. Same research-snapshot caveat as step 04 (pinned inputs not shipped).
