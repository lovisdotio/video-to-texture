"""Remove thin dark grey/brown lines baked inside UV islands (panel gaps and shading copied from the video),
keep saturated livery strokes (indigo, gold). usage: python delines.py in.png uv_tris.npy out.png [--k 15 --t 30]"""
import argparse, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
p = argparse.ArgumentParser(); p.add_argument("src"); p.add_argument("tris"); p.add_argument("out")
p.add_argument("--k", type=int, default=15); p.add_argument("--t", type=float, default=30)
a = p.parse_args()
img = np.asarray(Image.open(a.src).convert("RGB")).astype(np.float32); N = img.shape[0]
im = Image.new("L", (N, N), 0); dr = ImageDraw.Draw(im)
for t in np.load(a.tris): dr.polygon([(u * N, (1 - v) * N) for u, v in t], fill=255)
isl = np.asarray(im) > 127
lum = img.mean(-1)
blackhat = ndimage.grey_closing(lum, size=(a.k, a.k)) - lum          # thin features darker than surroundings
blueish = (img[..., 2] - img[..., 0]) > 25                          # indigo strokes are part of the livery
mask = (blackhat > a.t) & ~blueish & isl
mask = ndimage.binary_dilation(mask, iterations=2) & isl
Image.fromarray((mask * 255).astype(np.uint8)).resize((2048, 2048)).save(a.out.replace(".png", "_linemask_2k.png"))

def pushpull(c, w):
    if min(w.shape) <= 4 or w.min() >= 1: return c
    h2, w2 = c.shape[0] // 2, c.shape[1] // 2
    cw = (c[:h2*2, :w2*2] * w[:h2*2, :w2*2, None]).reshape(h2, 2, w2, 2, 3).sum((1, 3))
    ws = w[:h2*2, :w2*2].reshape(h2, 2, w2, 2).sum((1, 3))
    cd = pushpull(cw / np.maximum(ws, 1e-6)[..., None], np.minimum(ws, 1.0))
    up = np.stack([np.asarray(Image.fromarray(cd[..., k]).resize((c.shape[1], c.shape[0]), Image.BILINEAR)) for k in range(3)], -1)
    return c * w[..., None] + up * (1 - w[..., None])

out = pushpull(img, (~mask).astype(np.float32))
Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(a.out)
print("ok", a.out, f"lines removed {mask[isl].mean()*100:.2f}% of island texels")
