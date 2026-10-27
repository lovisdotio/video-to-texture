"""Clean UV atlas edges (step 07, v12 textures):
- islands keep their interior texels byte-exact (core = island eroded by --erode px, minus dark texels)
- near-black texels, dark texels in the --zone edge band, and the --erode band inside each island border
  are rebuilt smoothly from the core (push-pull / mip-pyramid diffusion)
- a --guard px band outside the islands gets the same diffusion
- everything further out is the Seedream padding fill (seedream_fill.py), so the atlas is one continuous surface
- also writes <out>_rebuilt_mask_2k.png (white = rebuilt texels) for inspection
usage: python recomposite.py <atlas.png> <uv_tris.npy> <seedream_fill_2k.png> <out.png> [--erode 24 --guard 40]"""
import argparse, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
p = argparse.ArgumentParser(); p.add_argument("src"); p.add_argument("tris"); p.add_argument("fill"); p.add_argument("out")
p.add_argument("--erode", type=int, default=3); p.add_argument("--black", type=int, default=45); p.add_argument("--zone", type=int, default=24); p.add_argument("--global-black", dest="global_black", type=int, default=30); p.add_argument("--guard", type=int, default=40)
a = p.parse_args()
img = np.asarray(Image.open(a.src).convert("RGB")).astype(np.float32); N = img.shape[0]
im = Image.new("L", (N, N), 0); dr = ImageDraw.Draw(im)
for t in np.load(a.tris): dr.polygon([(u * N, (1 - v) * N) for u, v in t], fill=255)
isl = np.asarray(im) > 127
# rebuild only what is actually black: near-black texels (and a 2 px halo) plus a thin edge sliver
edge_zone = isl & ~ndimage.binary_erosion(isl, iterations=a.zone)
dark = (img.max(-1) < a.global_black) | (edge_zone & (img.max(-1) < a.black))
core = ndimage.binary_erosion(isl, iterations=a.erode) & ~ndimage.binary_dilation(dark, iterations=2)
Image.fromarray(((isl & ~core) * 255).astype(np.uint8)).resize((2048, 2048)).save(a.out.replace(".png", "_rebuilt_mask_2k.png"))

def pushpull(c, w):
    if min(w.shape) <= 4 or w.min() >= 1: return c
    h2, w2 = c.shape[0] // 2, c.shape[1] // 2
    cw = (c[:h2*2, :w2*2] * w[:h2*2, :w2*2, None]).reshape(h2, 2, w2, 2, 3).sum((1, 3))
    ws = w[:h2*2, :w2*2].reshape(h2, 2, w2, 2).sum((1, 3))
    cd = pushpull(cw / np.maximum(ws, 1e-6)[..., None], np.minimum(ws, 1.0))
    up = np.stack([np.asarray(Image.fromarray(cd[..., k]).resize((c.shape[1], c.shape[0]), Image.BILINEAR)) for k in range(3)], -1)
    return c * w[..., None] + up * (1 - w[..., None])

smooth = pushpull(img, core.astype(np.float32))
fill = np.asarray(Image.open(a.fill).convert("RGB").resize((N, N), Image.LANCZOS), np.float32)
guard = ndimage.binary_dilation(isl, iterations=a.guard)
wf = ndimage.gaussian_filter(guard.astype(np.float32), 1.0); wf[isl] = 1.0
out = smooth * wf[..., None] + fill * (1 - wf[..., None])
Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(a.out)
print("ok", a.out, f"rim rebuilt {(isl & ~core).mean()*100:.2f}% of atlas")
