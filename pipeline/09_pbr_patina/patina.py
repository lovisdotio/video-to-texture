"""Run fal-ai/patina on one atlas and download its PBR maps. usage: FAL_KEY=... python patina.py in.png out_prefix"""
import json, os, sys, time, urllib.request
src, prefix = sys.argv[1], sys.argv[2]
KEY = os.environ["FAL_KEY"]
H = {"Authorization": f"Key {KEY}"}
# upload via fal storage
init = urllib.request.Request("https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3",
    json.dumps({"content_type": "image/png", "file_name": os.path.basename(src)}).encode(), {**H, "Content-Type": "application/json"})
up = json.load(urllib.request.urlopen(init))
urllib.request.urlopen(urllib.request.Request(up["upload_url"], open(src, "rb").read(), {"Content-Type": "image/png"}, method="PUT"))
body = {"image_url": up["file_url"], "maps": ["basecolor", "normal", "roughness", "metalness", "height"], "output_format": "png"}
sub = json.load(urllib.request.urlopen(urllib.request.Request("https://queue.fal.run/fal-ai/patina", json.dumps(body).encode(), {**H, "Content-Type": "application/json"})))
print("submitted", sub["request_id"], flush=True)
while json.load(urllib.request.urlopen(urllib.request.Request(sub["status_url"], headers=H)))["status"] != "COMPLETED":
    time.sleep(3)
res = json.load(urllib.request.urlopen(urllib.request.Request(sub["response_url"], headers=H)))
json.dump(res, open(prefix + "_result.json", "w"), indent=2)
for i, im in enumerate(res["images"]):
    name = im.get("file_name") or f"map{i}.png"
    urllib.request.urlretrieve(im["url"], f"{prefix}_{i}_{name}")
    print("got", name, im.get("width"), im.get("height"))
