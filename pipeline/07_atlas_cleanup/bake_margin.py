"""Re-bake an atlas onto itself with Blender's 'Adjacent Faces' margin so UV padding continues the
texture from the neighbouring island across each seam (run inside Blender, Cycles).

args: <object-name-prefix> <uv-map> <in.png> <out.png> <margin-px>
"""
import sys, bpy
argv = sys.argv[sys.argv.index("--") + 1:]
prefix, uvname, src, out, margin = argv[0], argv[1], argv[2], argv[3], int(argv[4])
scene = bpy.context.scene
scene.render.engine = "CYCLES"
try:
    scene.cycles.device = "GPU"
except Exception:
    pass
scene.cycles.samples = 1
objs = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith(prefix)]
src_img = bpy.data.images.load(src)
src_img.colorspace_settings.name = "sRGB"
w, h = src_img.size
dst = bpy.data.images.new("BAKE_DST", w, h, alpha=False)
mat = objs[0].material_slots[0].material
nt = mat.node_tree
tex = next(n for n in nt.nodes if n.type == "TEX_IMAGE")
tex.image = src_img
tex.interpolation = "Closest"  # texel-exact copy inside the islands
target = nt.nodes.new("ShaderNodeTexImage"); target.image = dst
nt.nodes.active = target
for o in objs:
    o.data.uv_layers.active = o.data.uv_layers[uvname]
bpy.ops.object.select_all(action="DESELECT")
def unexclude(lc):
    lc.exclude = False
    for c in lc.children: unexclude(c)
unexclude(bpy.context.view_layer.layer_collection)
for o in objs:
    o.hide_set(False); o.hide_viewport = False
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
bake = scene.render.bake
bake.margin = margin
bake.margin_type = "ADJACENT_FACES"
bpy.ops.object.bake(type="EMIT", margin=margin, margin_type="ADJACENT_FACES", use_clear=True)
dst.filepath_raw = out
dst.file_format = "PNG"
dst.save()
print("BAKED", out, w, h)
