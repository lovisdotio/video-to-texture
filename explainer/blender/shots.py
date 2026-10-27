"""Render explainer shots from the V11 final blend (read-only: never saves the .blend).

Usage: Blender -b V11.blend --python shots.py -- --shot turntable --variant v11 --out DIR [--frames 120] [--preview]
"""
import argparse, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--shot", required=True)            # turntable | unfold
p.add_argument("--variant", default="v11")         # clay | videofit | v11
p.add_argument("--out", required=True)
p.add_argument("--frames", type=int, default=120)
p.add_argument("--only", type=str, default="")     # comma list of frames to render
p.add_argument("--preview", action="store_true")
p.add_argument("--transparent", action="store_true")
p.add_argument("--tex", default="")
a = p.parse_args(argv)

scene = bpy.context.scene
PIECES = bpy.data.collections["Ferrari_V8_Selectable_Pieces"]

def all_objs(col):
    out = list(col.objects)
    for c in col.children: out += all_objs(c)
    return out

pieces = [o for o in all_objs(PIECES) if o.type == "MESH"]
if a.variant == "videofit":
    shown = [bpy.data.objects["Ferrari_VideoFit_4K"]]
else:
    shown = pieces
for o in bpy.data.objects:
    o.hide_render = o not in shown
    if o in shown:
        o.hide_viewport = False
# make sure every collection holding a shown object is renderable
def unexclude(lc):
    lc.exclude = False
    lc.collection.hide_render = False
    for c in lc.children: unexclude(c)
unexclude(bpy.context.view_layer.layer_collection)
for o in bpy.data.objects:
    o.hide_render = o not in shown

if a.variant == "clay":
    clay = bpy.data.materials.new("EXPLAINER_CLAY")
    clay.use_nodes = True
    bsdf = next(n for n in clay.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.42, 0.42, 0.43, 1)
    bsdf.inputs["Roughness"].default_value = 0.62
    bsdf.inputs["Base Color"].default_value = (0.20, 0.20, 0.21, 1)
    for o in pieces:
        for slot in o.material_slots:
            slot.link = "OBJECT"; slot.material = clay

r = scene.render
try:
    r.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    r.engine = "BLENDER_EEVEE"
r.resolution_x, r.resolution_y = (960, 540) if a.preview else (1920, 1080)
r.resolution_percentage = 100
r.fps = 30
r.film_transparent = a.transparent
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGBA" if a.transparent else "RGB"
scene.use_nodes = False
scene.view_settings.view_transform = "AgX"
try: scene.view_settings.look = "AgX - Medium High Contrast"
except TypeError: pass
try: scene.eevee.taa_render_samples = 16 if a.preview else 32
except Exception: pass

world = bpy.data.worlds.new("EXPLAINER_WORLD")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.008, 0.012, 0.022, 1)
bg.inputs["Strength"].default_value = 0.30
scene.world = world

def bbox(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi

lo, hi = bbox(pieces)
center = (lo + hi) / 2
if a.variant == "videofit":
    vf = shown[0]
    vlo, vhi = bbox([vf])
    vf.location += Vector((center.x - (vlo.x + vhi.x) / 2, center.y - (vlo.y + vhi.y) / 2, lo.z - vlo.z))
    print("VF_BBOX", tuple(vlo), tuple(vhi))
print("BBOX", tuple(lo), tuple(hi))

def look(obj, loc, tgt):
    obj.location = loc
    obj.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()

bpy.ops.mesh.primitive_plane_add(size=40.0, location=(center.x, center.y, lo.z - 0.018))
floor = bpy.context.object
fm = bpy.data.materials.new("EXPLAINER_FLOOR"); fm.use_nodes = True
fb = next(n for n in fm.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
fb.inputs["Base Color"].default_value = (0.009, 0.013, 0.021, 1)
fb.inputs["Roughness"].default_value = 0.48
floor.data.materials.append(fm)
floor.hide_render = a.transparent

for i, (loc, e, s, col) in enumerate((((-3.5, -4.0, 5.5), 1250, 4.0, (1, .88, .75)),
                                      ((4.5, -1.0, 3.0), 950, 3.0, (.66, .8, 1)),
                                      ((0, 5.0, 4.0), 1350, 3.5, (1, .45, .3)))):
    d = bpy.data.lights.new(f"EXPLAINER_L{i}", "AREA"); d.energy = e; d.shape = "DISK"; d.size = s; d.color = col
    L = bpy.data.objects.new(d.name, d); scene.collection.objects.link(L)
    look(L, Vector(loc) + Vector((center.x, center.y, 0)), (center.x, center.y, 0.65))

cd = bpy.data.cameras.new("EXPLAINER_CAM"); cd.lens = 50; cd.clip_start = 0.05; cd.clip_end = 200
cam = bpy.data.objects.new("EXPLAINER_CAM", cd); scene.collection.objects.link(cam); scene.camera = cam

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
only = {int(x) for x in a.only.split(",") if x}

def render(i):
    if only and i not in only: return
    r.filepath = str(out / f"f{i:04d}.png")
    bpy.ops.render.render(write_still=True)

if a.shot == "turntable":
    # one slow, slightly rising orbit; identical for every variant so stages can be wiped
    for i in range(a.frames):
        t = i / a.frames
        ang = math.radians(35 + 120 * t)
        rad = 7.6
        h = 2.2 + 0.6 * t
        look(cam, (center.x + rad * math.cos(ang), center.y + rad * math.sin(ang), h), (center.x, center.y, 0.55))
        render(i)

def smooth(x):
    x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)

if a.shot == "unfold":
    import bmesh
    body = [o for o in pieces if o.name.startswith("FerrariV8_01_body")]
    others = [o for o in pieces if o not in body]
    mat = body[0].material_slots[0].material
    if a.tex:
        img = bpy.data.images.load(a.tex)
        next(n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE").image = img
    # rebuild one mesh where every (vertex, uv) pair is unique so islands can separate
    SIZE = 3.0
    verts, flat, faces, uvs = [], [], [], []
    for o in body:
        me = o.data; uvl = me.uv_layers["Appearance_UV"].data; mw = o.matrix_world
        key = {}
        for poly in me.polygons:
            f = []
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index; u, v = uvl[li].uv
                k = (vi, round(u, 6), round(v, 6))
                if k not in key:
                    key[k] = len(verts)
                    verts.append(mw @ me.vertices[vi].co)
                    flat.append(Vector((center.x + (u - 0.5) * SIZE, center.y, 0.62 + (v - 0.5) * SIZE)))
                f.append(key[k]); uvs.append((u, v))
            faces.append(f)
    me = bpy.data.meshes.new("UNFOLD"); me.from_pydata([tuple(v) for v in verts], [], faces)
    uvlay = me.uv_layers.new(name="Appearance_UV")
    for i, uv in enumerate(uvs): uvlay.data[i].uv = uv
    me.materials.append(mat)
    ob = bpy.data.objects.new("UNFOLD", me); scene.collection.objects.link(ob)
    ob.shape_key_add(name="Basis"); sk = ob.shape_key_add(name="Flat")
    for i, co in enumerate(flat): sk.data[i].co = co
    for o in body: o.hide_render = True
    N = a.frames
    cam_a = (Vector((center.x + 7.6 * math.cos(math.radians(35)), center.y + 7.6 * math.sin(math.radians(35)), 2.2)), Vector((center.x, center.y, 0.55)))
    cam_b = (Vector((center.x, center.y - 9.2, 0.62)), Vector((center.x, center.y, 0.62)))
    for i in range(N):
        t = i / (N - 1)
        k_others = smooth((t - 0.08) / 0.20)
        for o in others: o.hide_render = k_others > 0.98; o.scale = (1 - k_others,) * 3
        floor.hide_render = t > 0.12
        sk.value = smooth((t - 0.18) / 0.55)
        c = smooth((t - 0.10) / 0.70)
        loc = cam_a[0].lerp(cam_b[0], c); tgt = cam_a[1].lerp(cam_b[1], c)
        look(cam, loc, tgt)
        render(i)

if a.shot == "hero":
    # low, close, slow orbit for the closing beat
    cd.lens = 35
    for i in range(a.frames):
        t = i / a.frames
        ang = math.radians(-60 + 70 * t)
        look(cam, (center.x + 5.0 * math.cos(ang), center.y + 5.0 * math.sin(ang), 0.55 + 0.25 * t), (center.x, center.y + 0.3, 0.5))
        render(i)
