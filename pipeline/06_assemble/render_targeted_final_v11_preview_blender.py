#!/usr/bin/env python3
"""Render five read-only views from audited V11 full-bleed using the exact V10 rig."""

from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path
from typing import Any
import bpy
from mathutils import Vector

SCRIPT_DIR=Path(__file__).resolve().parent; V8_ROOT=SCRIPT_DIR.parent; V10_ROOT=V8_ROOT/"targeted_final_v10"; FINAL_ASSEMBLER=V8_ROOT/"integration_v8/final_assembler_v1"
for value in (str(SCRIPT_DIR),str(V10_ROOT),str(FINAL_ASSEMBLER)):
    if value not in sys.path: sys.path.insert(0,value)
from blender_final_v8 import collection_objects_recursive  # noqa: E402
from contract_final_v8 import FINAL_WRAPPER_COLLECTION,V8_PIECES_COLLECTION  # noqa: E402
from targeted_final_v11_blender import reopened_images,validate_v11_bindings  # noqa: E402
from targeted_final_v11_common import BUILD_SCHEMA,EXPECTED_MESHES,EXPECTED_TRIANGLES,RENDER_ROOT,RENDER_SCHEMA,load_json,load_texture_set,require,sha256  # noqa: E402

VIEWS={
 "front":((0.0,7.8,2.65),(0.0,0.15,0.65)),
 "front_right":((6.2,4.0,2.75),(0.0,0.05,0.65)),
 "rear_right":((5.2,-5.5,2.85),(0.0,-0.10,0.65)),
 "rear_left":((-5.2,-5.5,2.85),(0.0,-0.10,0.65)),
 "front_left":((-6.2,4.0,2.75),(0.0,0.05,0.65)),
}

def point_at(obj,location,target): obj.location=location; obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()

def main()->None:
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--build-report",type=Path,required=True); parser.add_argument("--width",type=int,default=1920); parser.add_argument("--height",type=int,default=1080); parser.add_argument("--view-set",choices=("around5",),default="around5"); args=parser.parse_args(argv)
    blend=Path(bpy.data.filepath).resolve(); require(blend.is_file(),"Blender did not open saved V11 blend"); before_sha=sha256(blend); before_mtime=blend.stat().st_mtime_ns
    build_path=args.build_report.expanduser().resolve(); require(build_path==blend.with_name(blend.stem+".build_report.json"),"Build report basename mismatch"); build=load_json(build_path)
    require(build.get("schema")==BUILD_SCHEMA and build.get("status")=="passed" and all(build.get("hard_gates",{}).values()),"V11 build report failed")
    require(Path(build.get("final_blend","")).resolve()==blend and build.get("final_blend_sha256")==before_sha,"Opened V11 blend/build mismatch")
    for label,pkey,hkey in (("assembly","assembly_report","assembly_report_sha256"),("reopen","reopen_audit","reopen_audit_sha256"),("generator","generator_report","generator_report_sha256"),("independent audit","independent_audit","independent_audit_sha256")):
        path=Path(build.get(pkey,"")).expanduser().resolve(); require(path.is_file() and sha256(path)==build.get(hkey),f"V11 {label} proof drift")
    output=args.output_dir.expanduser().resolve()
    try: output.relative_to(RENDER_ROOT.resolve())
    except ValueError as exc: raise RuntimeError(f"V11 render output escapes root: {output}") from exc
    require(output.parent==RENDER_ROOT.resolve() and not output.exists(),"Unsafe/existing V11 render directory"); staging=output.with_name(output.name+".staging"); require(not staging.exists(),"Stale V11 render staging")
    require(args.width==1920 and args.height==1080,"V11 comparison must be exact 1920x1080")
    wrapper=bpy.data.collections.get(FINAL_WRAPPER_COLLECTION); pieces_collection=bpy.data.collections.get(V8_PIECES_COLLECTION); require(wrapper is not None and pieces_collection is not None,"V11 collections missing")
    pieces=sorted(collection_objects_recursive(pieces_collection),key=lambda o:str(o.get("piece_id"))); triangles=sum(len(o.data.polygons) for o in pieces); require(len(pieces)==EXPECTED_MESHES and triangles==EXPECTED_TRIANGLES,"V11 render geometry inventory drift")
    textures=load_texture_set(); images=reopened_images(textures); bindings=validate_v11_bindings(textures,pieces,images)
    piece_set=set(pieces)
    for obj in bpy.data.objects: obj.hide_render=obj not in piece_set
    for obj in pieces: obj.hide_render=False
    RENDER_ROOT.mkdir(parents=True,exist_ok=True); staging.mkdir(parents=True)
    try:
        scene=bpy.context.scene; scene.render.engine="BLENDER_EEVEE"; scene.render.resolution_x=args.width; scene.render.resolution_y=args.height; scene.render.resolution_percentage=100
        scene.render.image_settings.file_format="PNG"; scene.render.image_settings.color_mode="RGB"; scene.render.image_settings.color_depth="8"; scene.render.image_settings.compression=18
        scene.render.film_transparent=False; scene.render.use_border=False; scene.render.dither_intensity=0.0; scene.use_nodes=False; scene.view_settings.view_transform="AgX"
        try: scene.view_settings.look="AgX - Medium High Contrast"
        except TypeError: pass
        world=bpy.data.worlds.new("TARGETED_V11_PREVIEW_WORLD"); world.use_nodes=True; background=world.node_tree.nodes.get("Background"); require(background is not None,"V11 preview background missing"); background.inputs["Color"].default_value=(0.008,0.012,0.022,1.0); background.inputs["Strength"].default_value=0.30; scene.world=world
        minimum_z=min((obj.matrix_world@Vector(corner)).z for obj in pieces for corner in obj.bound_box); bpy.ops.mesh.primitive_plane_add(size=30.0,location=(0.0,0.0,minimum_z-0.018)); floor=bpy.context.object; floor.name="TARGETED_V11_PREVIEW_FLOOR"; floor.hide_render=False
        floor_mat=bpy.data.materials.new("TARGETED_V11_PREVIEW_FLOOR_MATERIAL"); floor_mat.use_nodes=True; bsdf=floor_mat.node_tree.nodes.get("Principled BSDF"); require(bsdf is not None,"V11 floor shader missing"); bsdf.inputs["Base Color"].default_value=(0.009,0.013,0.021,1.0); bsdf.inputs["Roughness"].default_value=0.48; floor.data.materials.append(floor_mat)
        lights=(((-3.5,-4.0,5.5),1250.0,4.0,(1.0,0.88,0.75)),((4.5,-1.0,3.0),950.0,3.0,(0.66,0.80,1.0)),((0.0,5.0,4.0),1350.0,3.5,(1.0,0.45,0.30)))
        for i,(loc,energy,size,color) in enumerate(lights):
            data=bpy.data.lights.new(f"TARGETED_V11_PREVIEW_LIGHT_{i}",type="AREA"); data.energy=energy; data.shape="DISK"; data.size=size; data.color=color; light=bpy.data.objects.new(data.name,data); scene.collection.objects.link(light); light.hide_render=False; point_at(light,loc,(0.0,0.0,0.65))
        camera_data=bpy.data.cameras.new("TARGETED_V11_PREVIEW_CAMERA"); camera_data.lens=58.0; camera_data.clip_start=0.05; camera_data.clip_end=100.0; camera=bpy.data.objects.new(camera_data.name,camera_data); scene.collection.objects.link(camera); camera.hide_render=False; scene.camera=camera
        renders={}
        for name,(location,target) in VIEWS.items():
            point_at(camera,location,target); staged_path=staging/f"{name}.png"; final_path=output/staged_path.name; scene.render.filepath=str(staged_path); bpy.ops.render.render(write_still=True); require(staged_path.is_file() and staged_path.stat().st_size>=32768,f"Missing V11 render: {name}"); renders[name]={"path":str(final_path),"sha256":sha256(staged_path),"bytes":staged_path.stat().st_size,"camera_location":list(location),"camera_target":list(target)}
        after_sha=sha256(blend); after_mtime=blend.stat().st_mtime_ns; require(after_sha==before_sha and after_mtime==before_mtime,"V11 blend changed during render")
        report={"schema":RENDER_SCHEMA,"status":"passed","blend":str(blend),"blend_sha256_before":before_sha,"blend_sha256_after":after_sha,"blend_mtime_ns_before":before_mtime,"blend_mtime_ns_after":after_mtime,"build_report":str(build_path),"build_report_sha256":sha256(build_path),"generator_report":str(textures.generator_report_path),"generator_report_sha256":textures.generator_report_sha256,"independent_audit":str(textures.independent_audit_path),"independent_audit_sha256":textures.independent_audit_sha256,"render_engine":scene.render.engine,"resolution":[args.width,args.height],"view_set":args.view_set,"renders":renders,"validated_image_bindings":len(bindings["nodes"]),"blend_saved":False,"remote_generation_called":False,"hard_gates":{"build_and_double_proofs_pinned":True,"opened_final_blend_exact":before_sha==build["final_blend_sha256"],"exact_66_meshes":len(pieces)==EXPECTED_MESHES,"exact_179488_triangles":triangles==EXPECTED_TRIANGLES,"seven_v11_images_packed":len(images)==7,"exact_32_bindings":len(bindings["nodes"])==32,"five_views_complete":len(renders)==5,"blend_bytes_unchanged":after_sha==before_sha,"blend_mtime_unchanged":after_mtime==before_mtime,"render_scene_not_saved":True,"no_remote_generation":True,"atomic_output_directory":True}}
        require(all(report["hard_gates"].values()),"V11 render report gate failed"); (staging/"render_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(staging,output)
    except Exception: shutil.rmtree(staging,ignore_errors=True); raise
    print("TARGETED_V11_PREVIEW="+json.dumps({"status":"passed","output":str(output),"report":str(output/"render_report.json")},sort_keys=True))

if __name__=="__main__": main()
