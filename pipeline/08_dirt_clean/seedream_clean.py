"""Seedream 5 Pro clean pass on one atlas page (2K), then put it back ONLY on dirty texels (dull grey/brown smudges).
usage: FAL_KEY=... python seedream_clean.py <atlas.png> <uv_tris.npy> <out.png> "<page guidance>" """
import json, os, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
src, tris_p, out_p, guidance = sys.argv[1:5]
KEY = os.environ["FAL_KEY"]; H = {"Authorization": f"Key {KEY}"}
stem = os.path.splitext(out_p)[0]
full = Image.open(src).convert("RGB"); N = full.size[0]
full.resize((2048, 2048), Image.LANCZOS).save(stem + "_in_2k.png")

def upload(path):
    req = urllib.request.Request("https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3",
        json.dumps({"content_type": "image/png", "file_name": os.path.basename(path)}).encode(), {**H, "Content-Type": "application/json"})
    up = json.load(urllib.request.urlopen(req))
    urllib.request.urlopen(urllib.request.Request(up["upload_url"], open(path, "rb").read(), {"Content-Type": "image/png"}, method="PUT"))
    return up["file_url"]

prompt = ("This is a UV texture atlas of a car livery. Clean it: remove every dirt smudge, grey or brown stain, baked shadow, "
          "soot and blurry dark patch, and repaint those areas as clean flat paint that continues the surrounding livery. "
          + guidance + " Keep every island, its outline, position and the livery pattern exactly where it is; do not move, "
          "resize or redesign anything. Flat albedo, no lighting, no shadows, no reflections, no text. Same 2048x2048 framing.")
body = {"prompt": prompt, "image_urls": [upload(stem + "_in_2k.png")], "image_size": {"width": 2048, "height": 2048},
        "num_images": 1, "output_format": "png"}
sub = json.load(urllib.request.urlopen(urllib.request.Request("https://queue.fal.run/bytedance/seedream/v5/pro/edit",
    json.dumps(body).encode(), {**H, "Content-Type": "application/json"})))
print("submitted", sub["request_id"], flush=True)
while json.load(urllib.request.urlopen(urllib.request.Request(sub["status_url"], headers=H)))["status"] != "COMPLETED":
    time.sleep(4)
res = json.load(urllib.request.urlopen(urllib.request.Request(sub["response_url"], headers=H)))
urllib.request.urlretrieve(res["images"][0]["url"], stem + "_seedream_2k.png")

# dirty = dull (low saturation), mid/dark, inside islands — not cream paint, not indigo
a = np.asarray(full).astype(np.float32)
im = Image.new("L", (N, N), 0); dr = ImageDraw.Draw(im)
for t in np.load(tris_p): dr.polygon([(u * N, (1 - v) * N) for u, v in t], fill=255)
isl = np.asarray(im) > 127
mx, mn = a.max(-1), a.min(-1); sat = (mx - mn) / np.maximum(mx, 1)
dirty = isl & (sat < 0.28) & (mx < 170) & ~((a[..., 2] - a[..., 0]) > 30)
dirty = ndimage.binary_opening(dirty, iterations=2)
dirty = ndimage.binary_dilation(dirty, iterations=6) & isl
w = ndimage.gaussian_filter(dirty.astype(np.float32), 4)
clean = np.asarray(Image.open(stem + "_seedream_2k.png").convert("RGB").resize((N, N), Image.LANCZOS), np.float32)
out = a * (1 - w[..., None]) + clean * w[..., None]
Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(out_p)
Image.fromarray((dirty * 255).astype(np.uint8)).resize((2048, 2048)).save(stem + "_dirtmask_2k.png")
print("ok", out_p, f"dirty {dirty[isl].mean()*100:.1f}% of island texels")
