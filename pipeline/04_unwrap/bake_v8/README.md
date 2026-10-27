# Ferrari six-page video baker v8

This isolated package contains the implemented GPU/Fal v8 inverse-projection
runner and its CPU/runtime contract. The first full-104 1024 GPU smoke proved
the raw six-page pipeline but correctly failed on four layered-glass pieces and
one steering micro-piece. The corrected runner is **ready for another 1024 GPU
smoke with all 104 aligned views and production-resolution 1280x720
visibility**. It does not claim a production GO until that new proof passes.

Frozen implementation identity:

- GPU runner SHA-256: `43cc731be9c5293c29ad302ef57512c18bf4f820370ee9d53fc415c2f19673c9`
- Runtime contract SHA-256: `97fb974e917b1b55b43d6da0eebc9d9a4a9bef60f992d0803d58d12004796427`
- Canonical dataset manifest: `16486f64605c33bd993cb7f0562e5e21bfa17dd5f5f68ed89df941245c93cf49`
- Ferrari/BODY geometry: `ba944d9c58e0fe4e7ac4c02da59ea356cbc70909001c244c91b6414cf3a974f4`
- Seedance video: `2a0743327c0a0169c57d4bdf50c089b8a9a1e869cc41e941f4d5cac024b75d87`
- Existing SeedVR RGB-only video: `34b4be7a36c112cbd64b475b6967d72e30d92b64814a63b9e0b582ebd843b39f`
- Alignment archive/report: `a5344198dd7fb2edd49b732cd6c37ce9a81111cffff4a762c2874a5d35c2723b` / `591a18d25ff29ed7a622008f7196afd1484e31d5d32cd886846002991a3ca07f`
- Minimal unwrap archive: `cc1bb5c4f97cee196aee6572950112ec62ea760794b0222ee4471f2b148d26c2`

## What the runner does

The six non-BODY pages are processed strictly sequentially so the two 8K
pages do not coexist in VRAM:

| Page | Production size | Parents | Pieces |
| --- | ---: | ---: | ---: |
| `exterior_opaque` | 4096² | 6 | 8 |
| `glass` | 4096² | 1 | 12 |
| `lights_translucent` | 4096² | 5 | 5 |
| `running_metal` | 8192² | 20 | 20 |
| `tires` | 4096² | 4 | 4 |
| `cockpit` | 8192² | 14 | 14 |

Geometry, cameras, intrinsics, depth, owners and dense alignment always come
from the pinned native 1280x720 Seedance dataset. Every accepted RGB sample
comes instead from the existing pinned 3840x2160 SeedVR video; it is a
radiometry-only carrier and can never become geometric authority. The 104
canonical IDs are unchanged and lie in `0..330`, safely inside its 358 decoded
frames. Native dense coordinates are converted only at colour sampling with
the exact pixel-center rule `x_rgb=3*x+1`, `y_rgb=3*y+1`, then sampled with
`align_corners=False`. No resize, crop, retime, new flow or new alignment is
performed. The runner
ports the proven v3 corrected-camera inverse projection, dense-warp gates,
adaptive erosion, chart-coherent two-pass winner, deterministic holdouts and
hashed proof/archive handshake. It replaces v3 object-level ownership with an
exact 63-piece identity gate.

Four removable inner glass pieces (`15:6`, `15:7`, `15:9`, `15:10`) are hidden
by outer transmissive covers in an ordinary frontmost full-z raster. They are
therefore evaluated with an exact target-piece crop128 self-depth raster against
opaque-z: sibling transmissive pieces are excluded, opaque occluders remain,
and the cropped raster selects a front triangle whose plane is evaluated on the
exact target ray. The plane/target depth must match within 0.1 mm and its normal
must face the same direction as the target face. Their sampled video
colour is explicitly reported as transmitted/frontmost composite radiometry,
not as an unoccluded photograph of the removed layer. The correctness-first
implementation adds up to four small target-only rasters whenever the glass page
is evaluated; full/opaque occlusion remains production-resolution.

The tiny `39:-1 steering_trim` owner is only one to two pixels in the earlier
384x216 dense field. Its exact geometry is visible, but the coarse semantic
label lands on adjacent object 36. The sole allowed coarse-label surrogate is
`39:-1 -> 36`, only at dense confidence >=0.5 and erosion radius <2. Both parts
share the future `v8_leather` cohort, but their source materials deliberately
differ: object 36 is `Leather`, object 39 is `Leather_red`. Exact owner, depth,
interior and sampled RGB remain piece 39, while label 36 authorizes only the
common global warp and is rejected if object 36 supplied an accepted local
flow. Reports classify these samples as global-warp coarse-label surrogate
transfer, never as direct exact-label evidence. The measured transfer remains
a warm red-brown accent (train/holdout R-dominant samples 96.8%/100%), rather
than inheriting object 36's neutral source colour.

BODY is not baked as a seventh page. Its v6 8K texture remains an immutable
passthrough, while its 40,277 representative triangles are real visibility
geometry in both `full_z` and `opaque_z`. The complete visibility mesh is
179,488 triangles. A smoke must observe nonzero BODY raster pixels in every
selected view; reporting the input face count alone is insufficient.

The exact opaque-z exclusion set is `(6, 15, 21, 22, 38)`. Objects 20 and 38
share the future `v8_lamp_emissive` shader, but their visibility metadata is
deliberately split: object 20 participates in opaque-z, while object 38 is
excluded and validates against full-z. The cockpit must also demonstrate
nonzero reveal samples through mapped glass.

## Raw bake and later enhancement boundary

This runner performs the raw, RGB-only SeedVR transfer onto the new canonical UVs. It
does **not** call Seedream, Topaz or any source Ferrari PBR texture. For every
page it exports the raw RGB texture plus:

- occupied, directly observed and generated-fill masks;
- per-texel fallback mask;
- chart-owner and piece-owner masks;
- confidence, view-support and source-frame diagnostics.
- categorical final-winner provenance (`video_evidence_class_id.png`: exact
  label, piece39 global-warp surrogate, or permitted nonlocal evidence).
- pre-dense-warp training/holdout geometric-opportunity masks and exact
  screen piece-owner counts per selected view.

Fill is isolated per `(piece_id, chart_id)`. An unobserved chart receives only
its own deterministic semantic-material neutral prior—never a page mean or a
neighbouring chart colour. Gutters are propagated by exact piece+chart owner
without cross-domain averaging.

Seedream editing may happen only after this raw bake, with UV islands fixed.
The edited result must be recomposed through the occupancy/owner masks;
gutters are regenerated deterministically by this pipeline, and upscale comes
after recomposition. Seedream is never responsible for moving islands or
inventing gutter ownership.

## Fail-closed smoke and production gates

Smoke runs all six pages on all 104 canonical views at 512² or 1024². The Fal
entrypoint defaults to 1024² because `fal run file.py::function` does not pass
keyword arguments. Atlas resolution is reduced, but visibility remains
production-exact at 1280x720 and view coverage uses the same deterministic,
coverage-aware 83-train/21-holdout split as production. Relative to the legacy
positional split, frame 48 moves from training to holdout and frame 51 moves
from holdout to training. This exact no-alias swap exercises the two-triangle
minor glass piece `15:0` in both partitions while preserving every other
recoverable floor. Recoverable pieces need at
least four direct texels, 0.1% direct coverage, one observed chart-domain
floor, a winning source frame and four holdout samples. Page holdout mean L1
must be at most 0.35. Objects 0, 21 and 37 are the only explicit low-evidence
policy set; canonical evidence is respectively 2 px/2 frames, 0/0 and 6/4.
Those exceptions are keyed to exact piece IDs (`0:-1`, `21:-1`, `37:-1`), not
their parent IDs. Layered glass never receives an N/A exception. Every other
piece must expose at least four geometric UV
opportunity texels in both splits. Screen-owner visibility with zero pre-warp
UV opportunity is a hard failure.

The geometric audit is also emitted in production so the native output ZIP
retains the same owner/opportunity provenance. It adds two boolean atlas masks
and owner-count scans per page, but it does not alter RGB candidate acceptance,
winner selection, fill, gutters or the production 32/16 quality floors.

Production uses 83 train and 21 holdout views, native 4K/8K pages, stronger
32-direct/16-holdout floors, and requires an exact hashed smoke archive from
the same runner, runtime, dataset, alignment, unwrap and SeedVR identities.
The proof also binds all 104 decoded RGB-frame PNG hashes; an old native-RGB
smoke or any different decode is rejected before production. The proof
verifier reopens the six PNGs and required masks, recomputes the piece/page
thresholds, checks GPU/CUDA/VRAM evidence and rejects a forged CPU JSON/fake
PNG archive.

The CPU contract keeps `fal_remote_allowed=false`. Only the two explicitly
invoked Fal entrypoints in `bake_ferrari_atlases_v8.py` permit remote work;
module import defines `IsolatedFunction` objects but never submits a job.

## Files and local verification

- `bake_ferrari_atlases_v8.py`: frozen Fal H100 smoke/production runner.
- `runtime_contract.py`: pinned dataset plus canonical sidecar runtime join.
- `contract.py`: pages, pieces, hashes, materials and CPU no-remote policy.
- `package_unwrap_archive.py`: deterministic 52-file minimal unwrap packager.
- `preflight.py`: canonical topology/hash/source and frozen-runner audit.
- `cpu_smoke.py`: dual-z, ownership and 20/38 negative tests.
- `runner_synthetic_smoke.py`: no-submit import, forged-proof, BODY-hash,
  archive safety and cross-chart contamination negative tests.
- `real_fal_import_no_submit.py`: real Fal client import without invocation.

Run the post-freeze local evidence in this order:

```bash
cd <project-root>/video_fit/object_v2
python3 smart_uv_v8_full_unwrap/bake_v8/package_unwrap_archive.py
python3 smart_uv_v8_full_unwrap/bake_v8/runner_synthetic_smoke.py
uv run --with fal==1.75.9 --with numpy==1.26.4 python smart_uv_v8_full_unwrap/bake_v8/real_fal_import_no_submit.py
python3 smart_uv_v8_full_unwrap/bake_v8/preflight.py
python3 smart_uv_v8_full_unwrap/bake_v8/cpu_smoke.py
```

The pinned unwrap URL is already recorded in `unwrap_url_v8.txt`. Rerun the
smoke entrypoint first. Production remains blocked until the full-view GPU
smoke passes and its exact output archive is supplied as the production proof.
