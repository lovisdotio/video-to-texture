# Targeted re-unwrap v9

This folder is an additive UV-repair experiment.  It does **not** replace or
edit v8, BODY v6/v5, the dataset, textures, a canonical `.blend`, or any
production authority.  No Fal, GitHub, texture bake, GUI Blender session or
canonical assembly belongs to this phase.

The governing design contract is the v8 `ARCHITECTURE.md` plus the immutable
hashes in `contract_v9.py`.

## Exact shortlist

`1:17`, `6:-1`, `7:-1`, `15:3`, `17:-1`, `18:-1`, `19:-1`, `23:-1`,
`26:-1`, `27:-1`, `28:-1`, `36:-1`, `49:-1`.

`object:component` uses `-1` for the entire non-BODY parent.  BODY `1:17` and
glass `15:3` are strict component selections.  The per-piece chart,
avoidable-microchart, seam, condition and density limits are data in
`contract_v9.py`; widening the shortlist or a limit is a contract change.

## Pipeline

1. `prepare_targets_v9.py` verifies all pinned hashes, the exact face mask and
   immutable geometry/topology/normal hashes.  It reads the passed v8 hero
   sidecars under `artifacts/unwrap_pack_hero_8k` and the BODY render proxy.
2. `targeted_unwrap_v9_blender.py` runs headlessly on at most two targets per
   smoke.  It tries bounded seam/method candidates and writes UV-only overlay
   NPZs.  Same-direction face winding is corrected only on a solver copy; the
   exact inverse corner permutation returns UVs to source-corner order.
3. `audit_patch_v9.py` independently reloads the authority and patch, proves
   the selected partition and immutable hashes, reconstructs charts, and
   reruns distortion/overlap gates.
4. `place_keep_fixed_v9_blender.py` is the fail-closed atlas stage.  It pins
   every KEEP loop and lets only target islands move.  It emits atlas-final
   patches only if KEEP UVs are bit-identical and the whole page proves zero
   overlap, 16px inter-chart/boundary margin, no occupancy regression, and
   unchanged resolution.  A solver-local patch is never a final atlas.

The placement smoke intentionally demonstrated that Blender's pinned pack can
fail to find space even while preserving KEEP exactly.  In that case the
status is `failed_no_final_patch` and **no** atlas-final NPZ is emitted.  The
next production allocator must place charts in the freed target domain (or a
new additive page) and pass the same whole-page gates; it may not silently
move the 53 KEEP pieces.

## Overlay schema

Local patches live at `runs/<run>/patches/<slug>.npz` and include:

- `source_object_id`, `piece_component_id`, `selected_proxy_face_indices`;
- `replacement_proxy_corner_uv` in original proxy-corner order;
- `local_chart_id`, `solver_corner_to_source_corner`;
- source/proxy SHA-256 and immutable array-hash proof;
- `atlas_coordinates_final=False`, `atlas_placement_required=True`,
  `keep_faces_may_move=False`.

Only the fixed-KEEP placement stage may flip the first two atlas flags.

## Reproducible checks

```bash
python3 -m py_compile \
  targeted_reunwrap_v9/*.py

python3 \
  targeted_reunwrap_v9/test_synthetic_v9.py

blender \
  --background --factory-startup \
  --python targeted_reunwrap_v9/targeted_unwrap_v9_blender.py -- \
  --selectors '49:-1' \
  --output-dir targeted_reunwrap_v9/runs/smoke_wipers49/repaired \
  --methods MINIMUM_STRETCH --seam-modes v8

python3 \
  targeted_reunwrap_v9/audit_patch_v9.py \
  --run targeted_reunwrap_v9/runs/smoke_wipers49/repaired
```

The real wiper smoke passes locally: 65→29 charts, 34→10 avoidable
microcharts, 22.42%→11.99% seam edges, condition p95 `1.2856`, density
p95:p05 `1.1121`, and zero final local overlap/UV degenerates.  Solver-only
orientation removes all 141 same-direction edges while the saved authority
geometry, topology and normals remain untouched.  Atlas placement remains a
separate hard gate, as recorded in every report and patch.

## Componentwise Minimum Stretch

The additive `componentwise_v8_minimum_stretch` strategy is scoped to objects
`17`, `18`, `19`, `28`, and `36`.  It partitions the selected proxy by stable
edge-connected topology component, installs only that component's v8 seams,
and runs `MINIMUM_STRETCH` for 100 iterations.  A component with a failed
solve, degenerate UV, fold, or positive-area overlap is replaced independently
by its byte-read v8 baseline UV; every patch records one provenance row per
component, including the fallback reason.

All resulting local islands are repacked together.  Blender `CONCAVE` is used
when its exact read-back overlap audit passes.  The object-18 smoke exposed one
repeatable intra-chart packer collision in stable component 334, so the bounded
fallback normalizes geometric density and packs chart AABBs on deterministic
shelves.  It uses only orientation-preserving per-chart similarities and a
16.5-pixel numerical guard.

```bash
blender \
  --background --factory-startup \
  --python targeted_reunwrap_v9/targeted_unwrap_v9_blender.py -- \
  --selectors '18:-1' \
  --output-dir targeted_reunwrap_v9/runs/smoke_componentwise_18_v4 \
  --strategies componentwise_v8_minimum_stretch

python3 \
  targeted_reunwrap_v9/audit_patch_v9.py \
  --run targeted_reunwrap_v9/runs/smoke_componentwise_18_v4 \
  --structural-only
```

The five-target comparison is
`runs/componentwise_comparative_v1/comparative_report.json`.  Across 58,286
faces and 490 topology components, 458 components use Minimum Stretch and 32
fall back to v8.  Charts decrease `1604→1369` and avoidable microcharts
`615→466`; every saved patch independently reconstructs with zero UV
degenerates, zero solver-orientation folds, and zero positive-area overlap.
These are structural candidates: some deliberately remain outside the strict
absolute chart/microchart limits, and all still require whole-page atlas
placement before baking.
