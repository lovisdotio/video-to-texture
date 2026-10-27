"""Fill every empty texel of a UV atlas with a smooth continuation of the nearest islands (push-pull / mip dilation).

usage: python pushpull.py atlas.png mask.png out.png   (mask: white = real texels)
"""
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def down(c, w):
    h, W = c.shape[0] // 2 * 2, c.shape[1] // 2 * 2
    c, w = c[:h, :W], w[:h, :W]
    cw = (c * w[..., None]).reshape(h // 2, 2, W // 2, 2, -1).sum((1, 3))
    ws = w.reshape(h // 2, 2, W // 2, 2).sum((1, 3))
    return cw / np.maximum(ws, 1e-6)[..., None], np.minimum(ws, 1.0)

def pushpull(c, w):
    if min(w.shape) <= 2 or w.min() >= 1: return c
    cd, wd = down(c, w)
    cd = pushpull(cd, wd)
    # bilinear upsample keeps the fill smooth instead of blocky
    up = np.stack([np.asarray(Image.fromarray(cd[..., k]).resize((c.shape[1], c.shape[0]), Image.BILINEAR)) for k in range(cd.shape[2])], -1)
    return c * w[..., None] + up * (1 - w[..., None])

src, msk, out = sys.argv[1:4]
img = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255
m = np.asarray(Image.open(msk).convert("L").resize(img.shape[1::-1], Image.NEAREST), np.float32) / 255
m = (m > 0.5).astype(np.float32)
res = pushpull(img, m)
Image.fromarray((np.clip(res, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
print("filled", out, f"{(1 - m.mean()) * 100:.1f}% empty texels")
