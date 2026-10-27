"""Fill the empty UV padding of an atlas with Seedream 5 Pro edit (same recipe as the v6 gutter fill),
then composite at full resolution: islands + adjacent-face margin stay byte-exact, Seedream only fills beyond.

usage: FAL_KEY=... python seedream_fill.py <baked_8k.png> <uv_tris.npy> <guide_2k.png> <out_dir> "<page guidance>"
"""
import json, os, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
baked_p, tris_p, guide_p, out_dir, guidance = sys.argv[1:6]
os.makedirs(out_dir, exist_ok=True)
KEY = os.environ["FAL_KEY"]; H = {"Authorization": f"Key {KEY}"}

def raster(tris, n):
    im = Image.new("L", (n, n), 0); dr = ImageDraw.Draw(im)
    for t in tris: dr.polygon([(u * n, (1 - v) * n) for u, v in t], fill=255)
    return np.asarray(im) > 127

tris = np.load(tris_p)
baked = Image.open(baked_p).convert("RGB"); N = baked.size[0]
fig1 = baked.resize((2048, 2048), Image.LANCZOS); fig1.save(f"{out_dir}/fig1_atlas_2k.png")
Image.open(guide_p).convert("RGB").resize((2048, 2048), Image.LANCZOS).save(f"{out_dir}/fig2_guide_2k.png")
Image.fromarray((raster(tris, 2048) * 255).astype(np.uint8)).save(f"{out_dir}/fig3_support_2k.png")

def upload(path):
    req = urllib.request.Request("https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3",
        json.dumps({"content_type": "image/png", "file_name": os.path.basename(path)}).encode(), {**H, "Content-Type": "application/json"})
    up = json.load(urllib.request.urlopen(req))
    urllib.request.urlopen(urllib.request.Request(up["upload_url"], open(path, "rb").read(), {"Content-Type": "image/png"}, method="PUT"))
    return up["file_url"]

prompt = (
    "This is a technical UV texture-atlas gutter reconstruction, not a presentation image. "
    "Figure 1 is a 2048x2048 UV atlas. Its pure-black background, black gaps and black holes are unfilled atlas padding, "
    "not transparency, and must not remain. Figure 2 is the same atlas with a crude stretched mechanical padding; use it only "
    "as a continuity and coverage guide, replacing the stretched streaks with coherent material extrapolation. Figure 3 is "
    "the authoritative UV support map: white shapes are the true islands, black is padding to reconstruct. Preserve every "
    "true island in exact count, position, scale, orientation, silhouette and internal detail. Do not move, rotate, resize, "
    "merge, split or redesign any island. Fill the entire square canvas outside the islands by smoothly extending the nearest "
    "island's paint and pattern, so the whole square reads as one continuous painted surface. No pure-black background, no "
    "holes, no checkerboard. " + guidance + " Keep lighting neutral and texture-map-like: flat albedo, no shadows, no "
    "highlights, no reflections, no scene, no text, no logos. Return one 2048x2048 PNG with UV registration unchanged.")
body = {"prompt": prompt, "image_urls": [upload(f"{out_dir}/fig{i}") for i in ("1_atlas_2k.png", "2_guide_2k.png", "3_support_2k.png")],
        "image_size": {"width": 2048, "height": 2048}, "num_images": 1, "output_format": "png"}
sub = json.load(urllib.request.urlopen(urllib.request.Request("https://queue.fal.run/bytedance/seedream/v5/pro/edit",
    json.dumps(body).encode(), {**H, "Content-Type": "application/json"})))
print("submitted", sub["request_id"], flush=True)
while json.load(urllib.request.urlopen(urllib.request.Request(sub["status_url"], headers=H)))["status"] != "COMPLETED":
    time.sleep(4)
res = json.load(urllib.request.urlopen(urllib.request.Request(sub["response_url"], headers=H)))
urllib.request.urlretrieve(res["images"][0]["url"], f"{out_dir}/seedream_fill_2k.png")
json.dump({"request_id": sub["request_id"], "prompt": prompt}, open(f"{out_dir}/request.json", "w"), indent=2)

# composite at full res: exact islands + their baked margin, Seedream beyond, soft 12 px blend at the band edge
fill = np.asarray(Image.open(f"{out_dir}/seedream_fill_2k.png").convert("RGB").resize((N, N), Image.LANCZOS), np.float32)
b = np.asarray(baked, np.float32)
isl = raster(tris, N)
band = ndimage.binary_dilation(isl, iterations=40)
w = ndimage.gaussian_filter(band.astype(np.float32), 6)
w[isl] = 1.0
out = b * w[..., None] + fill * (1 - w[..., None])
Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(f"{out_dir}/atlas_filled_8k.png")
print("done", f"{out_dir}/atlas_filled_8k.png")
