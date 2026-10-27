# Explainer film

The 42 s explainer (`media/explainer.mp4`) is rendered from code, in three layers.

## 1. Blender shots (`blender/shots.py`)

Read-only renders from the final step-06 scene (the `.blend` is never saved):

```bash
blender -b final.blend --python blender/shots.py -- --shot turntable --variant clay     --frames 150 --out renders/tt_clay
blender -b final.blend --python blender/shots.py -- --shot turntable --variant videofit --frames 150 --out renders/tt_videofit
blender -b final.blend --python blender/shots.py -- --shot turntable --variant v11      --frames 150 --out renders/tt_v11
blender -b final.blend --python blender/shots.py -- --shot unfold    --variant v11      --frames 150 --out renders/unfold_v11
blender -b final.blend --python blender/shots.py -- --shot hero      --variant v11      --frames 180 --out renders/hero
```

- `turntable`: clay vs. step-03 fit vs. final, same camera.
- `unfold`: the **mesh → UV unfold**. The body is rebuilt as one mesh where every
  (vertex, UV) pair is unique, so UV islands can separate. A `Flat` shape key places
  each vertex at its UV coordinate on a vertical plane; animating the key from 0 to 1
  unfolds the textured car into its atlas while the other parts shrink away and the
  camera moves to face the plane. `--tex` swaps the body texture (e.g. before / after
  step 05).
- `--texdir DIR` swaps every atlas page for `DIR/<page>_v12.png` (step 07 output);
  `--pbr` turns the unlit atlases into a lacquered car paint (Principled BSDF + coat).
- `hero`: slow, low closing orbit.

## 2. Motion layer (`motion/`)

`index.html` + `film.js` is a single 1920×1080 page. The whole film is one paused GSAP
timeline on a 120 BPM grid (scene windows in `SCENES`), and image sequences are picked
by time, so every frame is a pure function of `t`. `window.seek(frame)` sets that state.

`render.mjs` opens the page in headless Chrome (puppeteer-core) and, for each frame at
30 fps, calls `seek(frame)` and takes a screenshot: deterministic, no real-time playback.

```bash
cd motion
npm install                       # gsap, puppeteer-core
node render.mjs --out frames      # optional: --from N --to N --step N
```

`render.mjs` expects Chrome at the default macOS path; edit `executablePath` for other
systems. Image sequences go in `assets/seq/<name>/0001.jpg…` and stills in
`assets/img/` (not included: they are renders and video frames). Fonts are not
included either; the CSS falls back to Inter / Helvetica.

## 3. Encode

```bash
ffmpeg -framerate 30 -i motion/frames/%05d.jpg -i music.mp3 \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest media/explainer.mp4
```

The music (120 BPM, generated separately) is not included.
