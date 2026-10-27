"""Remove the dark rim baked at every UV island border, then refill empty texels from the island interiors.

usage: python seamfix.py atlas.png occupancy_mask.png out.png --erode PX
"""
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
p = argparse.ArgumentParser()
p.add_argument("atlas"); p.add_argument("mask"); p.add_argument("out")
p.add_argument("--erode", type=int, default=12)
p.add_argument("--pad", type=int, default=32)
a = p.parse_args()

img = np.asarray(Image.open(a.atlas).convert("RGB"))
h, w = img.shape[:2]
if a.mask.endswith(".npy"):
    # exact island mask: rasterize the mesh's UV triangles at full atlas resolution
    from PIL import ImageDraw
    tris = np.load(a.mask)
    mi = Image.new("L", (w, h), 0); dr = ImageDraw.Draw(mi)
    for t in tris:
        dr.polygon([(u * w, (1 - v) * h) for u, v in t], fill=255)
    m = np.asarray(mi) > 127
else:
    m = np.asarray(Image.open(a.mask).convert("L").resize((w, h), Image.NEAREST)) > 127
core = ndimage.binary_erosion(m, iterations=a.erode)
# every non-core texel takes the colour of the nearest core texel (sharp, no blur)
_, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
out = img[iy, ix]
# keep only a padding band around the islands; the rest of the atlas stays black (never sampled)
band = ndimage.binary_dilation(m, iterations=a.pad)
out[~band] = 0
Image.fromarray(out).save(a.out)
print("ok", a.out, f"core {core.mean()*100:.1f}% of atlas, rim removed {(m & ~core).mean()*100:.2f}%")
