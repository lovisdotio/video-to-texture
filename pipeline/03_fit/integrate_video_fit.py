#!/usr/bin/env python3
"""Integrate the 4K video-fitted Ferrari into a non-destructive scene copy.

Run with Blender (the source scene is never overwritten):

  blender \
    -b ../car_city.blend --python integrate_video_fit.py -- \
    --render --compare --export-glb

The fitted OBJ was exported in Blender-native coordinates (forward -Y, up Z),
so it is imported without an axis conversion or a placement transform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


HERE = Path(__file__).resolve().parent
DEFAULT_OBJ = HERE / "results_4k" / "ferrari_video_fit.obj"
DEFAULT_TEXTURE = HERE / "results_4k" / "texture_4k_padded.png"
DEFAULT_BLEND = HERE / "car_city_video_fit.blend"
DEFAULT_GLB = HERE / "car_city_ferrari_video_fit.glb"
DEFAULT_CAR_GLB = HERE / "ferrari_video_fit_4k.glb"
DEFAULT_RENDERS = HERE / "integration_renders"


def script_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ)
    parser.add_argument("--texture", type=Path, default=DEFAULT_TEXTURE)
    parser.add_argument("--output-blend", type=Path, default=DEFAULT_BLEND)
    parser.add_argument("--output-glb", type=Path, default=DEFAULT_GLB)
    parser.add_argument("--output-car-glb", type=Path, default=DEFAULT_CAR_GLB)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDERS)
    parser.add_argument("--report", type=Path, default=HERE / "integration_report.json")
    parser.add_argument("--frames", default="1,91,181,271")
    parser.add_argument(
        "--mode",
        choices=("unlit", "principled"),
        default="unlit",
        help="Canonical material. Unlit is preferred because the fitted atlas already contains video lighting.",
    )
    parser.add_argument("--render", action="store_true", help="Render the validation frames.")
    parser.add_argument("--compare", action="store_true", help="Render unlit and Principled at frame 91.")
    parser.add_argument("--export-glb", action="store_true", help="Export the complete visible scene as a GLB.")
    parser.add_argument("--export-car-glb", action="store_true", help="Export a standalone Ferrari-only GLB.")
    parser.add_argument("--no-save", action="store_true", help="Do not save an integrated .blend copy.")
    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def hide_original_car() -> list[bpy.types.Object]:
    root = bpy.data.objects.get("CAR")
    if root is None:
        raise RuntimeError("The expected original car root object 'CAR' was not found")

    original = [root, *list(root.children_recursive)]
    for obj in original:
        obj.hide_render = True
        obj.hide_viewport = True
        obj.hide_set(True)
        obj["video_fit_original_hidden"] = True
    return original


def import_fitted_mesh(obj_path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        forward_axis="NEGATIVE_Y",
        up_axis="Z",
        global_scale=1.0,
        use_split_objects=False,
        use_split_groups=False,
        validate_meshes=True,
    )
    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if len(imported) != 1:
        raise RuntimeError(f"Expected one imported mesh, got {[obj.name for obj in imported]}")

    car = imported[0]
    car.name = "Ferrari_VideoFit_4K"
    car.data.name = "Ferrari_VideoFit_4K_Mesh"
    car.location = (0.0, 0.0, 0.0)
    car.rotation_euler = (0.0, 0.0, 0.0)
    car.scale = (1.0, 1.0, 1.0)
    car["video_fit_source_obj"] = str(obj_path)
    car["video_fit_coordinates"] = "Blender native: forward -Y, up Z; identity transform"

    # Keep the replacement clearly separated from the preserved original hierarchy.
    collection = bpy.data.collections.get("VIDEO_FIT")
    if collection is None:
        collection = bpy.data.collections.new("VIDEO_FIT")
        bpy.context.scene.collection.children.link(collection)
    for old_collection in list(car.users_collection):
        old_collection.objects.unlink(car)
    collection.objects.link(car)

    for polygon in car.data.polygons:
        polygon.use_smooth = True
    return car


def load_srgb_texture(texture_path: Path) -> bpy.types.Image:
    image = bpy.data.images.load(str(texture_path), check_existing=True)
    image.name = "Ferrari_VideoFit_4K_BaseColor"
    image.colorspace_settings.name = "sRGB"
    # The .blend must remain usable even if results_4k is moved later.
    image.pack()
    return image


def add_image_node(nodes: bpy.types.Nodes, image: bpy.types.Image, x: float, y: float):
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = "VideoFit 4K sRGB"
    texture.label = "Video-fit atlas (sRGB)"
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "EXTEND"
    texture.location = (x, y)
    return texture


def make_unlit_material(image: bpy.types.Image) -> bpy.types.Material:
    """Create a camera-visible emission graph recognized as KHR_materials_unlit.

    This mirrors Blender's own glTF unlit import graph. The Light Path branch
    prevents the baked highlights in the atlas from illuminating the scene.
    """
    material = bpy.data.materials.new("Ferrari_VideoFit_Unlit")
    material.use_nodes = True
    material.diffuse_color = (0.15, 0.20, 0.22, 1.0)
    material["video_fit_role"] = "canonical"
    material["video_fit_reason"] = "Atlas already contains Seedance lighting; avoid double illumination"
    material["base_color_space"] = "sRGB"
    material["metallic_reference"] = 0.0
    material["roughness_reference"] = 0.68

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    image_node = add_image_node(nodes, image, -620, 40)
    emission = nodes.new("ShaderNodeEmission")
    emission.name = "Unlit Video Color"
    emission.inputs["Strength"].default_value = 1.0
    emission.location = (-320, 40)
    links.new(image_node.outputs["Color"], emission.inputs["Color"])

    light_path = nodes.new("ShaderNodeLightPath")
    light_path.location = (-320, 300)
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-90, -90)
    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (150, 80)
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 80)

    links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    return material


def make_principled_material(image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new("Ferrari_VideoFit_Principled_Reference")
    material.use_nodes = True
    material.diffuse_color = (0.15, 0.20, 0.22, 1.0)
    material.use_fake_user = True
    material["video_fit_role"] = "lighting comparison only"
    material["base_color_space"] = "sRGB"
    material["metallic"] = 0.0
    material["roughness"] = 0.68

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    image_node = add_image_node(nodes, image, -420, 40)
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (-100, 40)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.68
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.30
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (200, 40)
    links.new(image_node.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs[0], output.inputs["Surface"])
    return material


def assign_material(car: bpy.types.Object, material: bpy.types.Material) -> None:
    car.data.materials.clear()
    car.data.materials.append(material)


def render_frames(
    car: bpy.types.Object,
    material: bpy.types.Material,
    frames: list[int],
    directory: Path,
    prefix: str,
) -> None:
    scene = bpy.context.scene
    directory.mkdir(parents=True, exist_ok=True)
    assign_material(car, material)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.resolution_percentage = 100
    for frame in frames:
        scene.frame_set(frame)
        scene.render.filepath = str(directory / f"{prefix}_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {scene.render.filepath}")


def export_scene_glb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_visible=True,
        use_renderable=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_cameras=True,
        export_lights=True,
        export_animations=True,
        export_force_sampling=True,
        export_extras=True,
        export_yup=True,
        export_apply=False,
    )
    print(f"Exported autonomous GLB: {path}")


def export_car_glb(car: bpy.types.Object, path: Path) -> None:
    """Export only the replacement Ferrari and its packed unlit texture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = car.users_collection[0]
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=False,
        collection=collection.name,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_extras=True,
        export_yup=True,
        export_apply=False,
    )
    print(f"Exported Ferrari-only autonomous GLB: {path}")


def mesh_summary(car: bpy.types.Object, original: list[bpy.types.Object]) -> dict:
    world_vertices = [car.matrix_world @ vertex.co for vertex in car.data.vertices]
    bounds_min = [min(vertex[i] for vertex in world_vertices) for i in range(3)]
    bounds_max = [max(vertex[i] for vertex in world_vertices) for i in range(3)]
    return {
        "object": car.name,
        "vertices": len(car.data.vertices),
        "triangles": len(car.data.polygons),
        "uv_layers": len(car.data.uv_layers),
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "dimensions": list(car.dimensions),
        "identity_transform": list(car.location) == [0.0, 0.0, 0.0]
        and list(car.rotation_euler) == [0.0, 0.0, 0.0]
        and list(car.scale) == [1.0, 1.0, 1.0],
        "original_root": "CAR",
        "original_objects_preserved_and_hidden": len(original),
        "camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
        "camera_lens_mm": bpy.context.scene.camera.data.lens if bpy.context.scene.camera else None,
    }


def main() -> None:
    args = script_args()
    obj_path = require_file(args.obj, "fitted OBJ")
    texture_path = require_file(args.texture, "4K texture")
    if bpy.context.scene.camera is None:
        raise RuntimeError("The loaded scene has no active camera")

    original_frame = bpy.context.scene.frame_current
    original = hide_original_car()
    car = import_fitted_mesh(obj_path)
    image = load_srgb_texture(texture_path)
    unlit = make_unlit_material(image)
    principled = make_principled_material(image)
    canonical = unlit if args.mode == "unlit" else principled
    assign_material(car, canonical)
    car["video_fit_material_mode"] = args.mode

    frames = [int(item.strip()) for item in args.frames.split(",") if item.strip()]
    if args.render:
        render_frames(car, canonical, frames, args.render_dir.resolve(), f"video_fit_{args.mode}")
    if args.compare:
        comparison_dir = args.render_dir.resolve() / "material_comparison"
        compare_frame = 91
        render_frames(car, unlit, [compare_frame], comparison_dir, "unlit")
        render_frames(car, principled, [compare_frame], comparison_dir, "principled")

    assign_material(car, canonical)
    bpy.context.scene.frame_set(original_frame)
    summary = mesh_summary(car, original)
    summary.update(
        {
            "canonical_material": canonical.name,
            "texture": str(texture_path),
            "texture_color_space": image.colorspace_settings.name,
            "texture_packed_in_blend": bool(image.packed_file),
            "validation_frames": frames,
            "deliverables": {
                "integrated_blend": str(args.output_blend.resolve()),
                "full_scene_glb": str(args.output_glb.resolve()),
                "full_scene_glb_role": "complete visible city scene + fitted Ferrari",
                "ferrari_only_glb": str(args.output_car_glb.resolve()),
                "ferrari_only_glb_role": "standalone fitted Ferrari with embedded 4K unlit sRGB texture",
            },
        }
    )
    args.render_dir.resolve().mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(summary, indent=2) + "\n"
    (args.render_dir.resolve() / "integration_report.json").write_text(report_text, encoding="utf-8")
    args.report.resolve().write_text(report_text, encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not args.no_save:
        output_blend = args.output_blend.resolve()
        output_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), compress=True)
        print(f"Saved non-destructive scene copy: {output_blend}")
    if args.export_glb:
        export_scene_glb(args.output_glb.resolve())
    if args.export_car_glb:
        export_car_glb(car, args.output_car_glb.resolve())


if __name__ == "__main__":
    main()
