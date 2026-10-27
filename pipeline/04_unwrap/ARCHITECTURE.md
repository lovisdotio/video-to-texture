# Ferrari non-BODY full unwrap v8

## Scope and immutable inputs

This experiment replaces only the UV topology of object IDs other than BODY
(`1`). It reads the pinned v3 geometry and visibility report, and writes only
below `smart_uv_v8_full_unwrap`. It does not modify v7, production textures,
the v3 dataset or BODY v6. No Fal request or texture bake belongs to this
phase.

Pinned source inventory:

- 50 non-BODY source parents and 278,222 source triangles;
- 139,253 geometrically unique triangles after exact-coordinate weld/dedup
  when source-declared degenerates are included;
- 84 source-declared degenerate faces, representing 42 of those unique
  triangles, are excluded from the parameterization solver;
- 139,211 renderable proxy triangles enter Blender;
- 27,720 current UV charts across the three legacy 4K pages;
- no two objects have byte-identical local proxy geometry, so cross-object
  template reuse is forbidden by default;
- eight objects have non-manifold edges and seven are micro meshes below the
  64-face UVgami threshold.

The pinned UVgami checkout is commit
`8774be08537ab5e05e46381f2a1e3a3acc5c1324` (`GPL-3.0-or-later`). A real
check-only pass proposes 10,046 retained UVgami edges plus 3,014 mandatory
same-direction winding-conflict splits: 12,594 unique solver splits and 2,341
pre-unwrap charts. This is an 11.8× reduction from the current 27,720 charts
before Blender's effective seams and overlap repairs are counted.

## Pipeline contracts

### 1. Exact per-object proxy

`build_nonbody_proxies_v8.py` builds one independent proxy per object.

1. Source faces are selected by the dataset's exact object offsets.
2. The 84 `geometry_degenerate_face` rows are quarantined before solving.
3. Referenced positions are welded by exact float equality only; there is no
   tolerance, rounding, averaging, remesh or displacement.
4. Triangles are deduplicated by the sorted welded vertex triple. The first
   source occurrence supplies proxy winding.
5. Every renderable original face maps to one proxy face and every original
   corner maps through an explicit permutation.
6. Both forward arrays and reverse CSR maps are stored. Coverage of every
   renderable face and corner exactly once is a hard gate.

Each proxy NPZ contains, at minimum:

- `proxy_vertices`, `proxy_faces`;
- `original_to_proxy_face`, `original_corner_to_proxy_corner` (`-1` only for
  source-declared degenerate rows);
- reverse face and corner CSR arrays;
- source/global face IDs and the representative source row;
- connected-component labels;
- `proxy_seam_edges` and pre-unwrap chart labels after the seam stage.

Opposite-winding/coincident source copies receive the exact same UV triangle
through the stored corner permutation. They are not independently unwrapped.

The preferred final geometry is the unique proxy itself, not the doubled
source mesh with transferred UVs. The canonical unwrap/pack `.blend` is an
**intermediate UV carrier**: it contains 50 parent meshes, while its sidecars
carry the exact 63-piece partition. It is not the final selectable scene.
The downstream v8 assembler must split those carriers into 63 independent
non-BODY render pieces containing exactly 139,211 triangles in total. With
BODY remainder and its two separable mirror shells, the final scene contract
is 66 objects and 179,488 triangles. This removes
coincident duplicate surfaces (notably visible as z-fighting around the
wipers) while retaining per-object selection, material identity and future
replacement. The objects remain double-sided: Blender material bindings must
set backface culling off. Representative source corner normals are stored per
proxy loop in both local and world space; duplicate aliases are never averaged.

The source-transfer arrays remain mandatory even when proxy geometry is used
for display. They are the proof sidecar: every non-degenerate source face and
corner must map exactly onto its retained representative, including duplicate
and opposite-winding rows. A diagnostic source-geometry export may use that
sidecar, but it is not the recommended v8 scene deliverable.

### 2. Hard-surface seams and solver topology

UVgami `seam_edges(..., angle=66°)` is used when a proxy has at least 64 faces
and any 2-owner manifold adjacency. Smaller proxies use their natural
boundaries only; forcing feature cuts into a 19–46 triangle signature part
would make it less stable, not more detailed.

Object 15 is an explicit smooth-glass exception: it uses component/boundary
topology plus ABF and receives no 66° feature seams. Chrome 14 still uses the
hard-surface policy, but mirror-surface components 26/27 are protected from
feature cuts and remain one pre-unwrap chart each. Projector object 21 remains
two natural 160-face charts.

Before Blender unwrap, corner vertices are split across:

- every UVgami seam;
- every mesh boundary;
- every edge with owner count other than two;
- all 3,014 two-owner edges whose incident faces traverse the edge in the same
  direction.

This split is a parameterization-only topology. Face order and proxy-corner
order remain exact. The original mesh is never edited. In particular, the six
non-manifold edges in `chrome`, 39 each in the two main interior objects, four
in `plastic_gray`, and 56 in each wheel cannot reach Blender's ABF solver as
non-manifold adjacency.

### 3. Blender Angle Based unwrap and overlap patch

Each object is unwrapped independently with Blender `ANGLE_BASED`, zero unwrap
margin and `no_flip=True`. Island scale is normalized from its 3D surface area
before semantic weighting.

Positive-area triangle overlap is audited with a broad-phase grid and a strict
separating-axis test. Edge or point contact is not an overlap.

Repair is bounded:

1. preserve every clean Angle Based island;
2. Smart Project only faces participating in a proven overlap;
3. re-audit;
4. if any positive-area overlap remains, isolate only the residual faces with
   corrective seams and run Angle Based on those faces again;
5. fail if any positive-area overlap remains after the second audit.

There is no global Smart Project fallback.

### 4. Semantic, importance-weighted packing

V8 replaces the legacy three-page partition with six exact semantic pages:

- `exterior_opaque`: `[0, 14, 16, 44, 49, 50]`;
- `glass`: `[15]`;
- `lights_translucent`: `[6, 20, 21, 22, 38]`;
- `running_metal`: `[2–5, 10–13, 24–27, 29–32, 45–48]`;
- `tires`: `[40–43]`;
- `cockpit`: `[7–9, 17–19, 23, 28, 33–37, 39]`.

This split is required for both texture quality and later shader semantics:
glass, captured emissive/translucent lights, metallic running gear, rubber and
opaque cockpit appearance must not be forced into the same material page.

`OBJECT_SEMANTIC_SPECS` is the authoritative per-object/per-source-material
table. It assigns every one of the 50 source parents a new target PBR class,
transparency mode, stable composite `piece_id=(source_object_id, component_id)`, and optional
symmetry group/instance. Its required separation is:

- object 15 alone uses `v8_glass_transmission` on the dedicated `glass` page;
- object 14 uses opaque `v8_mirror_chrome`, never a glass/transmission shader;
- objects 6, 21 and 22 use translucent lamp-lens PBR, while 20 and 38 use a
  distinct translucent/emissive lamp PBR;
- running metal, tyres and opaque cockpit/exterior pieces use their own new
  semantic PBR templates;
- every source material name is retained as provenance only. The future
  integrator must construct the v8 Principled materials anew and must not copy
  a source PBR node tree or its scalar parameters.

The object-ID to source-material table is pinned by SHA-256
`5f55238f237e09407e2acda4f269cb2f79f62b7d6e99ee3a4b18c0c54a123d64`.
It is authoritative for both shaders and density weights. Object labels must
never be used to infer either: IDs 17/18 are the concrete counterexample,
because their `interior_dark`/`interior_light` labels are opposite their
`Interior_light`/`Interior_dark` source-material rows.

Two isolated component gates make the shader correction auditable. Object 15
has exactly 12 proxy components: component 2 is the 266-face windshield,
component 11 the 124-face rear window, and the remainder contains lamp/minor
glass. Object 14 has 66 proxy components; components 26 and 27 are the two
46-face rear-view mirror surfaces. The two 3,525-face mirror shells belong to
BODY object 1 and are outside v8. The current v7 appearance (opaque Emission
on transparent IDs 6, 15, 21, 22, 38 and non-metallic Emission on chrome 14)
is explicit negative evidence: v8 must construct real transmission/alpha lamp
and glass shaders plus a new opaque metallic mirror/chrome shader.

Symmetry metadata shares visibility priors and density between the four
brakes, centres, nuts, rims, wheels and tyres. It never authorizes UV stacking,
mirrored-coordinate reuse, cross-object chart merging or shared piece IDs.
Every page sidecar retains dense chart IDs plus both elements of the composite
piece ID. Chrome 14 becomes remainder + mirror surfaces 26/27; glass 15 becomes
its 12 connected components; the other 48 non-BODY parents remain whole. This
gives 63 separately maskable non-BODY render pieces. BODY remains on its exact
v6 UV/image but is exposed as remainder + shell components 17/18, giving 66
scene objects without changing the total face partition.

The hero-quality resolution policy is fixed before baking: `running_metal` and
`cockpit` are 8192², while `exterior_opaque`, `glass`, `lights_translucent` and
`tires` remain 4096². The two dense 8K pages are unconditional even when their
4K semantic floors would pass, because their 20/14 source-parent allocations
need materially more usable detail. Packing always enforces a 16 px boundary
margin at the chosen native resolution. Eight pixels at the top of each page
are reserved for private degenerate placeholders.

For each family within its new six-page target, the visibility prior is the maximum direct coverage among
its symmetric copies. Thus all four wheels, tyres, brakes, centres and nuts
retain matching density even when one side was less visible in the source
video. The linear density multiplier is:

```text
clip((0.35 + 0.65 * sqrt(family_visibility / atlas_max_visibility))
     * semantic_boost, 0.50, 3.00)
```

Signature parts receive explicit floors. A semantic boost of at least 2 gets
a 32,768 effective-texel minimum; other sub-128-face parts get 16,384; all
others get 4,096. Floors are applied before the global page scale.

Per-object unwrap UVs are first normalized to common geometric density, then
scaled about each island centre by the object's linear multiplier. Blender's
final pack may apply one common page scale, but must not average island scale
again because that would erase the weights.

Symmetric objects are never stacked. Their seam policy/density may match, but
each object receives separate atlas space because video-projected appearance
can differ between left/right and front/rear.

### 5. Geometry delivery, exact transfer and degenerate placeholders

The recommended geometry output is `unique_proxy_per_object`:

- exactly 63 non-BODY render pieces from 50 source parents, with source object
  IDs, topology component IDs and component roles preserved;
- exactly 139,211 renderable triangles in total, using the per-object proxy
  vertices/faces and the packed `proxy_corner_uv`;
- one material binding per source object, preserving its semantic target page;
- double-sided display (`use_backface_culling = False`);
- no merging across object boundaries, even if two objects share coordinates;
- no source-declared degenerate triangles in renderable geometry.

The 278,222-row source topology is retained only in the proof sidecar described
below. This distinction is intentional: transferring UVs back to every source
row proves completeness, but displaying coincident duplicate rows would
reintroduce the z-fighting that v8 is meant to remove.

Packed `proxy_corner_uv` transfers to original rows with:

```text
original_corner_uv[row, corner] =
    proxy_corner_uv[
        original_to_proxy_face[row],
        original_corner_to_proxy_corner[row, corner]
    ]
```

Renderable faces must reconstruct exactly through that mapping. The 84
source-declared degenerate faces bypass the formula and receive private 1 px
placeholder triangles in the reserved strip, following the source dataset's
existing non-rasterizing-face policy. Each placeholder is its own chart and
must not touch a renderable chart.

The final per-page sidecar must preserve `uv_vertex_to_vertex[uv_faces] ==
source_faces`, contain dense chart IDs, remain within `[0,1]`, have no
positive-area overlap, respect the 16 px pack margin, and cover exactly the
canonical object/face partition.

### 6. Required gates before any future rebake

- pinned source and UVgami hashes;
- BODY absent from every v8 sidecar;
- 278,222 original non-BODY faces covered exactly once;
- 139,211 renderable proxy faces and 84 private degenerates;
- the intermediate UV carrier may contain 50 parent meshes, but its sidecars
  must encode exactly 63 unique `piece_id` values;
- the final assembled non-BODY geometry contains exactly 63 separately
  selectable proxy pieces and 139,211 triangles, with no cross-parent weld;
  the complete BODY + non-BODY scene contains 66 objects and 179,488 triangles;
- source object/material identity is preserved and backface culling is off;
- proof sidecars retain the exact source-face and source-corner transfer maps;
- zero vertex displacement and exact corner permutation transfer;
- all solver components are manifold after seam/boundary splitting;
- finite in-bounds UVs and no renderable degenerate UV triangle;
- zero positive-area intra- or inter-island overlap;
- no chart crosses an object boundary;
- per-family densities match within a small numerical tolerance;
- semantic floors survive the common page scale;
- deterministic repeat hashes for proxy, UV and atlas sidecars.

Only after these gates pass should the inverse baker and six shader/material
bindings be adapted to the new v8 sidecars. Existing v7 textures cannot be
copied onto the new UVs.

### 7. Future transparency-aware rebake (documented, not run here)

The future v8 baker must retain the already validated two-depth-pass v3
contract:

1. a full-scene z-buffer, including transparent IDs, is used when texturing
   the glass/lens surfaces themselves;
2. a second z-buffer excludes exactly object IDs `[6, 15, 21, 22, 38]`, so
   interior surfaces behind glass receive their own exact observations;
3. both passes retain exact depth and object ID, and acceptance is decided per
   projected texel against the corresponding pass;
4. dense transparent warp is allowed only inside a proven strict proxy mask;
   it must never replace exact depth/object acceptance or spread into another
   piece/material.

This phase produces no z-buffer, dense warp or texture bake; the section is a
hard integration contract for the later baker.
