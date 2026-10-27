# 03 · Differentiable texture fit (nvdiffrast on an H100)

The mesh and all camera poses stay fixed; only a static RGB atlas is optimized so that
rendering the car from each known pose reproduces the matching video frame.

| Script | Where | What |
|---|---|---|
| `prepare_dataset.py` | local | 120 frames (`0,3,…,357`, the first orbit), poses/intrinsics from Blender, crops, xatlas UVs → `dataset_v03.zip` |
| `run_video_mask.py` | fal (paid) | BiRefNet video matte of the fitting window (`fal-ai/birefnet/v2/video`) |
| `add_masks_to_dataset.py` | local | adds the 120 mattes → `dataset_v03_masked.zip` |
| `fit_run.py` | fal Serverless H100 | alignment gate (median silhouette IoU ≥ 0.55), robust reprojection init, 2K/256 px coarse pass, 4K/512 px fine pass |
| `integrate_video_fit.py` | Blender | non-destructive import of the fitted OBJ + atlas, renders, GLB export |
| `hq_v2/` | | 8K variant: SeedVR 2160p restoration, 1024 px lossless crops, 8K atlas (see its README) |

```bash
python prepare_dataset.py
python run_video_mask.py && python add_masks_to_dataset.py
export DATASET_URL=<url of the uploaded dataset_v03_masked.zip>
fal run fit_run.py::fit_texture
```

Overrides (read at import, since `fal run` passes no arguments): `FIT_COARSE_ITERS`
(400), `FIT_FINE_ITERS` (900), `FIT_BATCH_SIZE` (6), `FIT_SEED` (63),
`FIT_MIN_MEDIAN_IOU` (0.55), plus `FIT_{COARSE,FINE}_{TEXTURE,RENDER}`.

Recorded results:

| Run | Atlas | Median IoU | Held-out masked L1 | H100 time |
|---|---|---:|---:|---:|
| v1 (`fit_run.py`) | 4096² | 0.844 | 0.102 | 55.8 s |
| HQ (`hq_v2/fit_run_hq.py`) | 8192² | — | 0.108 | 219 s, 13 GiB peak VRAM |

The 8K atlas looked about the same as 4K: the limit is the source, a 720p video where
the car is roughly 374 px wide in the median frame.

**Infrastructure lessons (fal Serverless):**

- Never compile CUDA in the sandbox (`nvcc` / JIT extensions crash there). Use the
  prebuilt wheel `nvdiffrast==0.4.0+253ac4fpt2.4.1cu121` with torch 2.4.1+cu121 and
  `dr.RasterizeCudaContext` (headless, no EGL).
- Put every heavy install in the container image (Dockerfile build), never in
  `setup()`: run-time installs hit the setup timeout.
- Use ephemeral `fal run file.py::fn`; nothing here needs a deployed app.

> nvdiffrast fitting can be deployed on fal Serverless — contact the fal team.
