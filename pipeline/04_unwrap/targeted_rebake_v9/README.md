# Targeted rebake v9

This folder is an additive handoff for the 13 UV repairs. It does not modify
the canonical v8 unwrap, the approved Seedream v5 textures, or the current
Blender assembly. The original `strict_patch` path remains available, and a
second `page_repack` mode permits a complete repack/rebake of touched pages.

The exact repair set is:

- BODY: `1:17`;
- exterior: `49:-1`;
- glass: `15:3`;
- lights: `6:-1`;
- running metal: `26:-1`, `27:-1`;
- cockpit: `7:-1`, `17:-1`, `18:-1`, `19:-1`, `23:-1`, `28:-1`, `36:-1`.

The strict pipeline has four gates:

1. Validate the 13 local patches produced by `targeted_reunwrap_v9`.
   Each patch must explicitly declare `atlas_coordinates_final=False`,
   `atlas_placement_required=True`, and `keep_faces_may_move=False`.
2. Insert those patches into new full-page V9 layouts. Untargeted corner UVs
   must remain byte-identical; a patch may use only the space released by its
   former target domain or otherwise proven free space.
3. Raster full-layout owner maps, then use `slice_page_for_piece_indices()` so
   the existing video baker accumulates samples only for target faces. Whole-car
   visibility remains enabled.
4. Recompose target RGB over the approved texture. Untargeted occupied RGB is
   copied byte-for-byte. Piece/chart owner maps come from the new layout and
   16-pixel gutters are regenerated from one exact donor, never averaged.

BODY `1:17` is intentionally routed separately because baker v8 treats BODY as
one immutable visibility piece; it must not be assigned a non-BODY piece index.

## Page-repack mode

`page_repack` resolves cases where a valid local patch cannot fit while all
existing atlas coordinates are frozen. On the five touched non-BODY pages,
every chart may be translated, rotated and uniformly scaled during packing.
For each untargeted KEEP chart, source face membership and local UV shape must
remain unchanged up to an orientation-preserving similarity transform. Chart
splits, merges, reflections, shear and non-uniform scale fail the gate.

All RGB on those five pages is then rebaked; target-only compositing is not
used. Full piece/chart owner maps and 16-pixel gutters are regenerated. The
tires page remains byte-identical. BODY `1:17` stays in its separate workflow.
Local patches with `atlas_coordinates_final=False` are valid inputs, but both
modes block baking until a final placement report passes overlap, bounds,
degeneracy, margin and (for page repack) KEEP-similarity gates.

Local checks (no Blender, Fal or bake):

```sh
python3 -m py_compile \
  targeted_rebake_v9.py test_targeted_rebake_v9.py
python3 -m unittest -v \
  test_targeted_rebake_v9.py
```

Once a re-unwrap smoke exists:

```sh
python3 targeted_rebake_v9.py \
  --mode page_repack \
  --reunwrap-run ../targeted_reunwrap_v9/runs/<run> \
  --validate-patches \
  --write-plan runs/<run>/targeted_rebake_plan.json
```

The writer refuses to overwrite an existing plan.
