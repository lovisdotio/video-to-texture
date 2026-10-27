"""Dump UV triangles per texture page (objects grouped by the atlas image their material samples).

Gives an exact island mask that recomposite.py rasterizes at full atlas resolution.

usage: blender -b final.blend -P dump_uv_pages.py -- <out_dir> [collection] [image-name-prefix]
  writes <out_dir>/<page>_uv_tris.npy  (float32, [n_tris, 3, 2] UV coordinates)
"""
import sys, bpy, numpy as np
argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
coll_name = argv[1] if len(argv) > 1 else "Ferrari_V8_Selectable_Pieces"
prefix = argv[2] if len(argv) > 2 else "Ferrari_TargetedV11_"
col = bpy.data.collections[coll_name]
def objs(c):
    o = list(c.objects)
    for ch in c.children: o += objs(ch)
    return o
pages = {}
for o in objs(col):
    if o.type != "MESH": continue
    me = o.data; me.calc_loop_triangles()
    for t in me.loop_triangles:
        m = o.material_slots[t.material_index].material if o.material_slots else None
        if not m or not m.use_nodes: continue
        tex = next((n for n in m.node_tree.nodes if n.type == "TEX_IMAGE"), None)
        if not tex or not tex.image: continue
        uvn = next((n for n in m.node_tree.nodes if n.type == "UVMAP"), None)
        layer = me.uv_layers[uvn.uv_map] if uvn and uvn.uv_map else next(l for l in me.uv_layers if l.active_render)
        key = tex.image.name.split(prefix)[-1].rsplit("_", 2)[0]
        pages.setdefault(key, []).append([tuple(layer.data[l].uv) for l in t.loops])
for k, v in pages.items():
    np.save(f"{out}/{k}_uv_tris.npy", np.array(v, np.float32)); print("PAGE", k, len(v))
