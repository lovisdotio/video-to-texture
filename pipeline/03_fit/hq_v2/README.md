# Video-fit HQ v2

This is an independent, non-destructive successor to `video_fit/` v1.  None of the
commands below overwrite the 4K result, its datasets, its Blender integration, or its
GLBs.  New paid requests require an explicit `--submit` or `fal run` command; none was
launched while preparing this pipeline.

## What materially changes

- Restore the useful 358-frame/14.92 s fitting window to 3840x2160 with a temporally
  coherent video model.  The fitter uses frames `0,3,...,357`; later
  frames make the car progressively tiny and add less texture information.
- Store 120 square crops as lossless 1024 px PNGs instead of 512 px JPEGs.
- Store matching 1024 px soft PNG mattes generated from the restored source.
- Generate a new 8192x8192 xatlas unwrap with 16 px gutters and 1200 texels/unit.
- Stabilize at 4K texture / 512 px render, then build the 8K initialization by directly
  reprojecting the original 1024 px target crops.  The fine pass is 8K / 1024 px; it
  does not pretend that bilinear enlargement of the coarse atlas is new detail.
- Draw one frame from each temporal stratum in every batch, while retaining sharpness
  weighting, robust loss, the holdout split and residual rejection from v1.

The native v03 car is only about 374 px wide in the median sampled frame and 563 px at
its largest.  Restoration can clean compression, improve edge continuity and recover
stable linework, but it cannot reveal guaranteed ground-truth detail that never existed.
For that reason 1024 px fitting is the honest production default.  A 1536 crop is
supported for an A/B test, but should only be paired with `FIT_FINE_RENDER=1536` after
inspection proves that the restored video contains stable additional information.

## Exact production commands

Run from the project root.

### 1. Make a lossless-window copy (local, free)

```bash
mkdir -p video_fit/hq_v2/source
ffmpeg -hide_banner -loglevel error \
  -i after_effects/seedance_v03.mp4 \
  -map 0:v:0 -an -frames:v 358 -c:v copy \
  video_fit/hq_v2/source/seedance_v03_first15_native.mp4
```

Validate that the copy contains 358 frames at 24 fps:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of json video_fit/hq_v2/source/seedance_v03_first15_native.mp4
```

### 2. Restore to 2160p (paid; not run yet)

The conservative recommendation is SeedVR2 with target 2160p, fixed seed,
`noise_scale=0.05` (the setting approved in the six-second A/B), and maximum X264
quality.  The alternate `--backend topaz` uses Starlight Precise 2.5,
which is explicitly restoration-oriented for AI video but may invent more micro-detail.

```bash
# requires FAL_KEY in the environment
python \
  video_fit/hq_v2/restore_video_hq.py --backend seedvr --submit
```

SeedVR2 bills by output megapixel-frame.  At 3840x2160 and 358 frames, the documented
formula gives approximately $2.97.  Restoring the full 721-frame video would be about
$5.98 and would not improve this first-orbit fit.  Current endpoint/pricing references:

- <https://fal.ai/models/fal-ai/seedvr/upscale/video/api>
- <https://fal.ai/models/fal-ai/topaz/upscale/video/api>

### 3. Build the lossless 8K-layout dataset (local, free)

```bash
tools/venv/bin/python video_fit/hq_v2/prepare_dataset_hq.py
```

This command fails if the restored source is below 2160p, has a different aspect/crop,
does not retain the required frames, or if the 8K UV topology changes.

### 4. Generate and attach HQ mattes (paid mask request; not run yet)

```bash
# requires FAL_KEY in the environment
python \
  video_fit/hq_v2/run_video_mask_hq.py --submit
tools/venv/bin/python video_fit/hq_v2/add_masks_hq.py
```

BiRefNet runs at its 2048 operating resolution with foreground refinement.  The matte
video keeps the restored source dimensions; `add_masks_hq.py` scales calibrated crop
coordinates if the service returns another same-aspect resolution and retains soft
edges in lossless 1024 px PNGs.

### 5. Upload and run the H100 fit (paid compute; not run yet)

```bash
# requires FAL_KEY in the environment
python \
  video_fit/hq_v2/upload_dataset_hq.py
export DATASET_URL="$(sed -n '1p' video_fit/hq_v2/dataset_url_hq.txt)"
export FIT_COARSE_ITERS=500
export FIT_FINE_ITERS=1200
export FIT_COARSE_TEXTURE=4096
export FIT_FINE_TEXTURE=8192
export FIT_COARSE_RENDER=512
export FIT_FINE_RENDER=1024
export FIT_BATCH_SIZE=3
export FIT_SEED=63
export FIT_MIN_MEDIAN_IOU=0.55
fal run video_fit/hq_v2/fit_run_hq.py::fit_texture_hq
```

A small-file upload helper is not suitable here: a lossless 120-view HQ archive is normally around 160 MiB or more.
`upload_dataset_hq.py` uses fal's multipart repository and writes
`dataset_url_hq.txt`, which the fitter reads automatically.

`production_config.json` records the same values without secrets.

## Memory and time budget

For 120 views at 1024 render resolution, the persistent fixed render cache contains
about 5.36 GiB of raw float tensors.  Concatenation during cache construction can
temporarily approach 10.7 GiB.  An 8K RGB float atlas is 0.75 GiB; Adam state, logits,
gradients, fine confidence/support, direct-reprojection accumulators and raster
temporaries bring the estimated end-to-end peak to 18–28 GiB.  Budget 32 GiB.  The
requested H100 has 80 GiB, leaving a wide safety margin.

At 1536 fine render resolution, the cache scales by 2.25x and the estimated peak rises
to roughly 38–50 GiB.  It should fit H100 80GB, but the source-information ceiling does
not justify making that the default.  Expected HQ wall time is roughly 4–12 minutes,
versus 55.8 seconds for v1; the actual runner report records peak-independent elapsed
time and final metrics.

## Why this version does not split body/glass/wheels yet

`ferrari.glb` has 51 meshes and 17 materials, so a body-only texture plus original PBR
glass/wheels is architecturally attractive.  The fitted `ferrari_clean2.glb` / UniTEX
topology, however, is one material-less primitive after joining, welding and
decimation.  Its 179,498 faces break into 1,589 disconnected components; the largest
contains only 5,858 faces (3.26%).  There is no retained material or face-range label
that safely identifies every body panel versus glass, trim or wheels.

Guessing from position would create seams and missing panels.  The safe v2 production
path therefore keeps the proven complete proxy.  A later body-only branch should start
from the original named `body` mesh (80,566 of the original 358,788 triangles), align
that geometry to the calibrated proxy, unwrap only it, and leave every original PBR
part untouched.  That is a geometry/integration change, not a texture-resolution flag.

## Local validation already completed

- Python syntax compilation passed for every HQ pipeline script.
- `fit_run_hq.py` imported through the authenticated fal environment with defaults
  `4096 -> 8192`, `512 -> 1024`, batch 3; no remote function was invoked.
- The real mesh completed an 8K xatlas generation locally in 5.6 s: one 8192x8192
  atlas, 8,324 charts, 156,679 UV vertices, requested/actual density 1200.0 and xatlas
  utilization 70.49%.
- A complete 120-frame diagnostic dataset build passed PNG extraction, hashes, shapes,
  camera inverses, 1024 px crop calibration, UV topology and ZIP CRC.  The matching
  mask augmentation also passed with all 120 mattes and archive validation.  This dry
  run used the native file only to exercise the code path; it was deleted and is not a
  production HQ asset.
- Rasterized chart interiors occupy about 25.08% of the atlas.  At 8K this is equivalent
  to approximately a dense 4103x4103 texture, twice v1's effective dense linear detail.

## Hybrid Blender/GLB integration after the HQ fit

Extract the final fit ZIP to `video_fit/hq_v2/results_8k/`.  The directory must
contain `ferrari_video_fit_hq.obj`, `texture_8k_padded.png` and `report.json`.
Then run:

```bash
blender \
  -b video_fit/hybrid/car_city_video_fit_hybrid.blend \
  --python video_fit/hq_v2/integrate_hq_hybrid.py
```

This does not remap the 8K PNG onto the old 4K UVs.  It imports the HQ OBJ, transfers
the exact 40,277-face body selection from the proven hybrid by world-space triangle
signature, and reuses the existing 50 visible PBR mesh parts and their materials in
place.  It packs the 8K sRGB atlas into the `.blend`, exports a car-only GLB, and
renders four 3840x2160 validation views.  Outputs go to `video_fit/hq_v2/hybrid_8k/`.

Strict preflight without writes, renders or exports:

```bash
blender \
  -b video_fit/hybrid/car_city_video_fit_hybrid.blend \
  --python video_fit/hq_v2/integrate_hq_hybrid.py -- --dry-run
```

The preflight rejects an incomplete fit, an atlas other than 8192x8192, an OBJ/report
topology mismatch, a geometry face-count change, or a hybrid scene that no longer has
exactly 50 visible original PBR mesh parts.  A complete local proxy integration using
the current 4K OBJ/atlas also passed: all 40,277 body faces matched with zero unmatched
signatures, the portable `.blend` saved, and the GLB exported and re-imported.

## Compare v1 and HQ on the same views

The HQ fitter includes eight lossless full-resolution renders in
`results_8k/heldout_predictions/`.  After extracting the result ZIP, create the
quantitative and visual comparison locally (no paid endpoint is involved):

```bash
tools/venv/bin/python video_fit/hq_v2/evaluate_v1_vs_hq.py
```

This requires `video_fit/hq_v2/dataset_v03_hq_masked/` and compares dataset indices
`0,15,...,105` (source-video frames `0,45,...,315`) on both versions.  Outputs go to
`video_fit/hq_v2/evaluation_v1_vs_hq/`:

- `view_comparison.png`: v1 target/fitted and HQ target/fitted in four aligned columns;
- `texture_comparison.png`: normalized atlas overviews and matched UV-space crops;
- `crops/`: 512 px inspection crops for every common view;
- `metrics.json`: production metrics, standardized 256 px masked metrics, per-view
  errors, fitted/target sharpness ratios and atlas diagnostics;
- `README.md`: concise interpretation and a deliberately conservative rule-based verdict.

Preflight without writing:

```bash
tools/venv/bin/python video_fit/hq_v2/evaluate_v1_vs_hq.py --preflight
```

The evaluator has also passed an end-to-end self-comparison with v1 standing in for
both sides.  Reproduce that free proxy check with `--proxy-hq-v1`; it must report
`equivalent` and zero deltas.  Atlas sharpness is treated only as a diagnostic because
ringing, seams or noise can otherwise masquerade as improvement.
