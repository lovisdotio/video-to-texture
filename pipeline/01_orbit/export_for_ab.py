"""Export everything needed for option A (Metashape) and B (Brush) from car_city.blend.
Run headless:
blender -b car_city.blend -P export_for_ab.py

Outputs:
  brush_dataset/transforms.json   (240 poses, nerfstudio convention = matrix_world direct)
  brush_dataset/init.ply          (~300k colored points sampled on the meshes, binary PLY)
  metashape/cameras.xml           (120 poses, Agisoft XML, CV axis convention)
  metashape/cameras.nvm           (fallback, VisualSFM N-View Match)
  metashape/carcity.obj           (whole scene, Blender world axes preserved)
"""
import bpy, os, json, math, struct
import numpy as np
from mathutils import Matrix, Vector

ROOT = os.environ.get("PROJECT_ROOT") or bpy.path.abspath("//")  # folder of the .blend
scn = bpy.data.scenes["CarCity"]
bpy.context.window.scene = scn
cam = bpy.data.objects["OrbitCam"]

W, H = 1280, 720
FL = (W / 2) / math.tan(cam.data.angle_x / 2)   # focal in pixels

BRUSH_FRAMES = list(range(1, 721, 3))       # 240
MS_FRAMES = list(range(1, 721, 6))          # 120

os.makedirs(f"{ROOT}/brush_dataset", exist_ok=True)
os.makedirs(f"{ROOT}/metashape", exist_ok=True)

poses = {}
for f in sorted(set(BRUSH_FRAMES + MS_FRAMES)):
    scn.frame_set(f)
    bpy.context.view_layer.update()
    poses[f] = cam.matrix_world.copy()

# ---------- Brush: transforms.json (OpenGL c2w = Blender matrix_world as-is) ----------
frames_out = []
for f in BRUSH_FRAMES:
    M = poses[f]
    frames_out.append({
        "file_path": f"images/{f:04d}.png",
        "transform_matrix": [list(row) for row in M.row],
    })
tj = {
    "fl_x": FL, "fl_y": FL, "cx": W / 2, "cy": H / 2, "w": W, "h": H,
    "camera_model": "OPENCV",
    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
    "ply_file_path": "init.ply",
    "frames": frames_out,
}
with open(f"{ROOT}/brush_dataset/transforms.json", "w") as fp:
    json.dump(tj, fp, indent=1)
print("[ok] transforms.json", len(BRUSH_FRAMES), "frames, fl =", round(FL, 2))

# ---------- Metashape: Agisoft XML (CV convention: flip Y and Z columns) ----------
FLIP = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
cams_xml = []
for i, f in enumerate(MS_FRAMES):
    Mcv = poses[f] @ FLIP
    vals = " ".join(f"{v:.10g}" for row in Mcv.row for v in row)
    cams_xml.append(
        f'      <camera id="{i}" sensor_id="0" label="{f:04d}">\n'
        f'        <transform>{vals}</transform>\n'
        f'      </camera>'
    )
xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<document version="1.5.0">
  <chunk>
    <sensors>
      <sensor id="0" label="blender_24mm" type="frame">
        <resolution width="{W}" height="{H}"/>
        <calibration type="frame" class="adjusted">
          <resolution width="{W}" height="{H}"/>
          <f>{FL:.6f}</f>
          <cx>0</cx>
          <cy>0</cy>
        </calibration>
      </sensor>
    </sensors>
    <cameras>
{chr(10).join(cams_xml)}
    </cameras>
  </chunk>
</document>
'''
with open(f"{ROOT}/metashape/cameras.xml", "w") as fp:
    fp.write(xml)
print("[ok] cameras.xml", len(MS_FRAMES), "cameras")

# ---------- Metashape fallback: NVM (w2c quaternion + camera center, CV axes) ----------
lines = ["NVM_V3", "", str(len(MS_FRAMES))]
for f in MS_FRAMES:
    Mcv = poses[f] @ FLIP
    R_c2w = Mcv.to_3x3()
    q = R_c2w.transposed().to_quaternion()      # world->cam
    c = poses[f].translation
    lines.append(f"{f:04d}.png {FL:.6f} {q.w:.10g} {q.x:.10g} {q.y:.10g} {q.z:.10g} "
                 f"{c.x:.10g} {c.y:.10g} {c.z:.10g} 0 0")
lines += ["", "0", "", "0"]
with open(f"{ROOT}/metashape/cameras.nvm", "w") as fp:
    fp.write("\n".join(lines))
print("[ok] cameras.nvm")

# ---------- OBJ export for Metashape (keep Blender world axes!) ----------
bpy.ops.object.select_all(action='DESELECT')
n_sel = 0
for o in scn.objects:
    if o.type == 'MESH' and o.name != "PROJ_PROXY" and o.visible_get():
        o.select_set(True)
        n_sel += 1
bpy.ops.wm.obj_export(
    filepath=f"{ROOT}/metashape/carcity.obj",
    export_selected_objects=True,
    apply_modifiers=True,
    export_materials=False,
    export_uv=False,
    export_normals=True,
    forward_axis='Y', up_axis='Z',           # no axis conversion — same frame as cameras
)
print("[ok] carcity.obj", n_sel, "meshes")

# ---------- init.ply for Brush: area-weighted surface samples with rough colors ----------
def group_color(o):
    n = o.name
    def under_car(x):
        p = x
        while p:
            if p.name == "CAR":
                return True
            p = p.parent
        return False
    if under_car(o):
        return (30, 35, 80)          # indigo car
    if n == "Ground" or n.startswith("Road"):
        return (35, 33, 34)          # dark asphalt
    if n.startswith("Sidewalk"):
        return (150, 140, 125)
    if n == "Markings":
        return (200, 200, 195)
    if n.startswith("Tree"):
        return (60, 90, 45)
    if n.startswith("Bldg"):
        return (140, 125, 105)       # warm concrete/glass
    return (90, 90, 90)

TOTAL = 400_000
deps = bpy.context.evaluated_depsgraph_get()
tris = []
for o in scn.objects:
    if o.type != 'MESH' or o.name == "PROJ_PROXY" or not o.visible_get():
        continue
    ev = o.evaluated_get(deps)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    Mw = ev.matrix_world
    col = group_color(o)
    verts = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    Mnp = np.array(Mw)
    vw = verts @ Mnp[:3, :3].T + Mnp[:3, 3]
    idx = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", idx)
    idx = idx.reshape(-1, 3)
    a, b, c = vw[idx[:, 0]], vw[idx[:, 1]], vw[idx[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tris.append((a, b, c, areas, col))
    ev.to_mesh_clear()

total_area = sum(t[3].sum() for t in tris)
pts_list, col_list = [], []
rng = np.random.default_rng(7)
for a, b, c, areas, col in tris:
    n = int(round(TOTAL * areas.sum() / total_area))
    if n == 0:
        continue
    ti = rng.choice(len(areas), size=n, p=areas / areas.sum())
    r1 = np.sqrt(rng.random(n))[:, None]
    r2 = rng.random(n)[:, None]
    p = (1 - r1) * a[ti] + r1 * (1 - r2) * b[ti] + r1 * r2 * c[ti]
    pts_list.append(p)
    cc = np.tile(np.array(col, dtype=np.uint8), (n, 1))
    jitter = rng.integers(-12, 13, size=cc.shape)
    col_list.append(np.clip(cc.astype(int) + jitter, 0, 255).astype(np.uint8))

pts = np.concatenate(pts_list).astype(np.float32)
cols = np.concatenate(col_list)
with open(f"{ROOT}/brush_dataset/init.ply", "wb") as fp:
    fp.write(b"ply\nformat binary_little_endian 1.0\n")
    fp.write(f"element vertex {len(pts)}\n".encode())
    fp.write(b"property float x\nproperty float y\nproperty float z\n")
    fp.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
    fp.write(b"end_header\n")
    buf = bytearray()
    for p, c in zip(pts, cols):
        buf += struct.pack("<fffBBB", p[0], p[1], p[2], c[0], c[1], c[2])
    fp.write(bytes(buf))
print("[ok] init.ply", len(pts), "points")
print("EXPORT_AB_DONE")
