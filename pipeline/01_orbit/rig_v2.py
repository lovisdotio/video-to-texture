"""Camera rig v2 — 2 revolutions, 24mm wide, progressive look-up. Run headless:
blender -b car_city.blend -P rig_v2.py
"""
import bpy, math

scn = bpy.data.scenes["CarCity"]
cam_obj = bpy.data.objects["OrbitCam"]
target = bpy.data.objects["CamTarget"]

cam_obj.data.lens = 24.0

cam_obj.animation_data_clear()
target.animation_data_clear()

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)

F0, F1 = 1, 720
for f in range(F0, F1 + 1):
    t = (f - F0) / (F1 - F0)
    theta = t * 2 * 2 * math.pi - math.pi / 2   # 2 revolutions
    r = 6.5 + 19.5 * (t ** 1.8)                  # 6.5 -> 26 m, stays close early
    h = 1.8 + 13.2 * (t ** 1.6)                  # 1.8 -> 15 m
    tz = 0.6 + 6.9 * smoothstep((t - 0.3) / 0.7) # car -> mid buildings -> skyline
    cam_obj.location = (r * math.cos(theta), r * math.sin(theta), h)
    cam_obj.keyframe_insert("location", frame=f)
    target.location = (0.0, 0.0, tz)
    target.keyframe_insert("location", frame=f)

# sanity: car must stay inside vertical FOV at key points
sensor_h = cam_obj.data.sensor_width * 9 / 16
half_vfov = math.degrees(math.atan(sensor_h / 2 / cam_obj.data.lens))
for f in (1, 240, 480, 720):
    scn.frame_set(f)
    c = cam_obj.matrix_world.translation
    d = math.hypot(c.x, c.y)
    look_down = math.degrees(math.atan2(c.z - target.location.z, d))
    car_down = math.degrees(math.atan2(c.z - 0.6, d))
    off = car_down - look_down
    print(f"f{f}: r={d:.1f} h={c.z:.1f} tz={target.location.z:.2f} "
          f"car offset {off:.1f}deg vs half-FOV {half_vfov:.1f}deg "
          f"{'IN' if off < half_vfov else 'OUT'}")

bpy.ops.wm.save_mainfile()
print("RIG_V2_SAVED lens=24mm")
