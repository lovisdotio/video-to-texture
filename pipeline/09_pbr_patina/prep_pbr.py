"""Turn PATINA outputs (0 basecolor, 1 normal, 2 roughness, 3 metalness, 4 height) into PBR maps per page.
Normals are flattened near UV island borders so PATINA's per-island bevel does not draw lines along seams."""
import glob, sys, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
pages = ["body_hero_8k", "exterior_opaque", "running_metal", "lights_translucent", "cockpit", "glass", "tires"]
for p in pages:
    get = lambda i: glob.glob(f"v12/{p}_{i}_*.png")[0]
    n = np.asarray(Image.open(get(1)).convert("RGB")).astype(np.float32); N = n.shape[0]
    im = Image.new("L", (N, N), 0); dr = ImageDraw.Draw(im)
    for t in np.load(f"../fill/pages/{p}_uv_tris.npy"): dr.polygon([(u * N, (1 - v) * N) for u, v in t], fill=255)
    isl = np.asarray(im) > 127
    d = ndimage.distance_transform_edt(isl)                 # px from the island border
    w = np.clip((d - 3) / 10, 0, 1)[..., None]              # flat at the border, full detail 13 px inside
    flat = np.array([128, 128, 255], np.float32)
    Image.fromarray((n * w + flat * (1 - w)).astype(np.uint8)).save(f"pbr/{p}_normal.png")
    Image.open(get(2)).convert("L").save(f"pbr/{p}_rough.png")
    Image.open(get(3)).convert("L").save(f"pbr/{p}_metal.png")
    print("ok", p)
