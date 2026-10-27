# 01 · Clay orbit with known camera poses

The whole method rests on one fact: **the AI video is generated from a camera move we
rendered ourselves**, so every frame comes with a known camera pose.

- `rig_v2.py` keys the `OrbitCam` / `CamTarget` rig of the `CarCity` scene: 720 frames
  at 24 fps (30 s), 24 mm lens, two revolutions spiralling up and out
  (r 6.5 → 26 m, h 1.8 → 15 m) while the look-at target rises from the car to the
  skyline. It prints a sanity check that the car stays inside the vertical FOV.
- `export_for_ab.py` exports those poses (and a point cloud / OBJ) for the two
  photogrammetry-style tools we tried first: nerfstudio `transforms.json` for Brush
  (OpenGL c2w) and Agisoft XML / NVM for Metashape (OpenCV axes). Kept as a reference
  for the camera-convention conversions.

```bash
blender -b your_scene.blend -P rig_v2.py        # keys the camera, saves the .blend
blender -b your_scene.blend -P export_for_ab.py # PROJECT_ROOT=... to choose the output folder
```

Then render the orbit in clay (any untextured material) to a 1280×720, 24 fps mp4.
Step 03 reads the poses straight from the `.blend`.
