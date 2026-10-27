# Targeted Seedream V10

This additive stage creates exactly nine 2048×2048 Seedream donors from the
validated V10 input package. It does not change UV support, canonical textures,
Blender files, or native-resolution atlases.

## Fixed candidates

1. `body_hero_8k`
2. `exterior_opaque`
3. `glass`
4. `lights_translucent`
5. `running_metal`
6. `tires`
7. `cockpit_leather`
8. `cockpit_plastics`
9. `cockpit_carbon`

Every request begins with the same three kinds of authoritative input:

1. preguttered working RGB atlas;
2. sparse occupied-only RGB layout;
3. exact binary candidate mask.

Rendered views are pinned by SHA256 and are explicitly described as identity
references, never as lighting or color references. Prompts require neutral
albedo with no baked highlights, reflections, shadows, AO, glow, scenery or
black void. Cockpit prompts separately require matte cognac leather,
anthracite plastic/Alcantara, and deep dark satin carbon; true metals remain
excluded.

## Offline preflight

The default command cannot perform network access:

```sh
cd pipeline/05_seedream
python3 run_seedream_v10.py \
  --emit-preflight preflight_seedream_v10.json
```

It rehashes the manifest, every declared raster and every pinned reference,
checks decoded pixel hashes, verifies mask/owner support, and proves the three
cockpit target masks are disjoint and partition occupancy with the exclusion.

The fully local journal/resume smoke test is:

```sh
python3 test_seedream_v10_offline.py
```

## Approved network execution

Network execution has two explicit locks (`--execute` and the exact go token)
and also requires `FAL_KEY`:

```sh
python3 run_seedream_v10.py \
  --execute \
  --go-token SEEDREAM_V10_NETWORK_GO \
  --run-id seedream_v10_all_v1 \
  --max-workers 3
```

Resume the identical frozen contract without resubmitting recorded request IDs:

```sh
python3 run_seedream_v10.py \
  --execute \
  --go-token SEEDREAM_V10_NETWORK_GO \
  --run-id seedream_v10_all_v1 \
  --max-workers 3 \
  --resume
```

The runner uploads each unique input hash once, atomically writes each request
plan before submission, atomically records `request_id`, response/status/cancel
URLs immediately after submission, and writes each decoded RGB output plus file
and pixel hashes before producing:

`runs/<run-id>/seedream_v10_results_manifest.json`

Its schema is `car-city.ferrari-targeted-seedream-results/v10`; status is
`passed` only for all nine candidates. Downstream stages must use the pinned
input material masks and must never infer UV support from generated pixels.

Fal exposes no idempotency key in the installed client. A process death in the
small interval between successful remote submission and the atomic local
`submitted.json` write can therefore leave one orphan provider request. This
limitation is recorded in every frozen run contract; all normally recorded
requests resume by `(model, request_id)`.

