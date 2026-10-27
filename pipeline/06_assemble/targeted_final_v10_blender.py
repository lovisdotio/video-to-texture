#!/usr/bin/env python3
"""Blender-only implementation and invariants for the texture-only V10 build."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
V9_ROOT = V8_ROOT / "targeted_final_v9"
FINAL_ASSEMBLER = V8_ROOT / "integration_v8/final_assembler_v1"
for value in (str(SCRIPT_DIR), str(V9_ROOT), str(FINAL_ASSEMBLER)):
    if value not in sys.path:
        sys.path.insert(0, value)

from blender_final_v8 import (  # noqa: E402
    custom_properties,
    environment_signature_sha256,
    material_signature_sha256,
    mesh_corner_normals,
    node_signature,
    packed_sha256,
    scalar,
    used_materials,
    validate_specialized_shaders,
)
from contract_final_v8 import BODY_MATERIAL_NAME  # noqa: E402
from targeted_final_blender import capture_car_state, car_objects  # noqa: E402
from targeted_final_v10_common import (  # noqa: E402
    ASSEMBLY_SCHEMA,
    EXPECTED_IMAGE_BINDINGS,
    EXPECTED_MESHES,
    EXPECTED_PAGE_MATERIAL_COUNTS,
    EXPECTED_TRIANGLES,
    IMAGE_SCHEMA,
    PAGE_IDS,
    REOPEN_SCHEMA,
    SOURCE_ASSEMBLY_REPORT,
    SOURCE_BLEND,
    SOURCE_PINS,
    TextureSet,
    canonical_json_sha256,
    load_json,
    require,
    sha256,
)
from targeted_final_common import hash_array  # noqa: E402


def _car_state(objects: list[Any]) -> dict[str, Any]:
    state = capture_car_state(objects)
    state["decoded_corner_normal_sha256_by_piece"] = dict(
        sorted(
            (
                str(obj["piece_id"]),
                hash_array(mesh_corner_normals(obj.data)),
            )
            for obj in objects
        )
    )
    return state


def _node_tree_signature(node_tree) -> dict[str, Any] | None:
    if node_tree is None:
        return None
    return {
        "nodes": [
            node_signature(node, include_image=True)
            for node in sorted(node_tree.nodes, key=lambda item: item.name)
        ],
        # JSON round-trips tuples as lists. Emit lists at capture time so the
        # in-memory post-save state and the reopened report compare exactly;
        # all four endpoint strings remain part of the invariant.
        "links": sorted(
            [
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            ]
            for link in node_tree.links
        ),
    }


def _camera_data_row(camera) -> dict[str, Any]:
    dof = camera.dof
    return {
        "name": camera.name,
        "type": camera.type,
        "lens": float(camera.lens),
        "lens_unit": camera.lens_unit,
        "sensor_fit": camera.sensor_fit,
        "sensor_width": float(camera.sensor_width),
        "sensor_height": float(camera.sensor_height),
        "clip_start": float(camera.clip_start),
        "clip_end": float(camera.clip_end),
        "shift_x": float(camera.shift_x),
        "shift_y": float(camera.shift_y),
        "ortho_scale": float(camera.ortho_scale),
        "panorama_type": getattr(camera, "panorama_type", None),
        "display_size": float(camera.display_size),
        "dof": {
            "use_dof": bool(dof.use_dof),
            "focus_object": dof.focus_object.name if dof.focus_object else None,
            "focus_distance": float(dof.focus_distance),
            "aperture_fstop": float(dof.aperture_fstop),
            "aperture_blades": int(dof.aperture_blades),
            "aperture_rotation": float(dof.aperture_rotation),
            "aperture_ratio": float(dof.aperture_ratio),
        },
        "custom_properties": custom_properties(camera),
    }


def _world_row(world) -> dict[str, Any]:
    return {
        "name": world.name,
        "use_nodes": bool(world.use_nodes),
        "color": scalar(world.color),
        "node_tree": _node_tree_signature(world.node_tree if world.use_nodes else None),
        "custom_properties": custom_properties(world),
    }


def camera_world_state(objects: list[Any]) -> dict[str, Any]:
    scene = bpy.context.scene
    view = scene.view_settings
    result = {
        "environment_sha256": environment_signature_sha256(exclude_objects=objects),
        "active_scene": scene.name,
        "active_camera": scene.camera.name if scene.camera else None,
        "active_world": scene.world.name if scene.world else None,
        "render_engine": scene.render.engine,
        "resolution": [
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
            int(scene.render.resolution_percentage),
        ],
        "frame_current": int(scene.frame_current),
        "view_settings": {
            "view_transform": view.view_transform,
            "look": view.look,
            "exposure": float(view.exposure),
            "gamma": float(view.gamma),
        },
        "cameras": [_camera_data_row(camera) for camera in sorted(bpy.data.cameras, key=lambda item: item.name)],
        "worlds": [_world_row(world) for world in sorted(bpy.data.worlds, key=lambda item: item.name)],
        "scene_custom_properties": custom_properties(scene),
    }
    result["sha256"] = canonical_json_sha256(result)
    return result


def _image_row(image) -> dict[str, Any]:
    return {
        "name": image.name,
        "source": image.source,
        "type": image.type,
        "size": [int(image.size[0]), int(image.size[1])],
        "colorspace": image.colorspace_settings.name,
        "alpha_mode": image.alpha_mode,
        "filepath": image.filepath,
        "packed_sha256": packed_sha256(image) if image.packed_file is not None else None,
        "users": int(image.users),
        "use_fake_user": bool(image.use_fake_user),
        "custom_properties": custom_properties(image),
    }


def image_inventory(*, exclude_names: set[str] | None = None) -> dict[str, Any]:
    excluded = exclude_names or set()
    rows = [_image_row(image) for image in sorted(bpy.data.images, key=lambda item: item.name) if image.name not in excluded]
    return {"count": len(rows), "rows": rows, "sha256": canonical_json_sha256(rows)}


def material_state(objects: list[Any]) -> dict[str, Any]:
    materials = used_materials(objects)
    rows = {
        material.name: {
            "shader_without_image_sha256": material_signature_sha256(material, include_image=False),
            "custom_properties": custom_properties(material),
        }
        for material in materials
    }
    return {"count": len(materials), "materials": dict(sorted(rows.items()))}


def _page_for_material(material) -> str:
    if material.name == BODY_MATERIAL_NAME:
        return "body_hero_8k"
    page_id = str(material.get("atlas_page_id", ""))
    require(page_id in PAGE_IDS, f"Used material has no supported atlas page: {material.name}")
    return page_id


def source_bindings(objects: list[Any]) -> dict[str, Any]:
    source_assembly = load_json(SOURCE_ASSEMBLY_REPORT)
    production = source_assembly.get("production_images", {})
    require(set(production) == set(PAGE_IDS), "Pinned V9 production-image report drift")
    images: dict[str, Any] = {}
    for page_id in PAGE_IDS:
        record = production[page_id]
        image = bpy.data.images.get(record.get("name"))
        require(image is not None, f"Pinned V9 image datablock is missing: {page_id}")
        require(tuple(int(value) for value in image.size) == tuple(record.get("size", [])), f"Pinned V9 image size drift: {page_id}")
        require(packed_sha256(image) == record.get("packed_sha256"), f"Pinned V9 packed image drift: {page_id}")
        require(image.get("atlas_page_id") == page_id, f"Pinned V9 image metadata drift: {page_id}")
        images[page_id] = image

    counts = {page_id: 0 for page_id in PAGE_IDS}
    rows: list[dict[str, Any]] = []
    for material in used_materials(objects):
        page_id = _page_for_material(material)
        nodes = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage"]
        require(len(nodes) == 1, f"Pinned V9 image-node inventory drift: {material.name}")
        node = nodes[0]
        if page_id != "body_hero_8k":
            require(node.name == "V8_ATLAS_IMAGE", f"Pinned V9 image node name drift: {material.name}")
        require(node.image is images[page_id], f"Pinned V9 material/image binding drift: {material.name}")
        rows.append(
            {
                "material": material.name,
                "node": node.name,
                "page_id": page_id,
                "image": node.image.name,
                "sampling": [node.interpolation, node.projection, node.extension],
            }
        )
        counts[page_id] += 1
    require(len(rows) == EXPECTED_IMAGE_BINDINGS, "Pinned V9 must expose exactly 32 used image nodes")
    require(counts == EXPECTED_PAGE_MATERIAL_COUNTS, "Pinned V9 material/page counts drift")
    return {"images": images, "page_material_counts": counts, "nodes": sorted(rows, key=lambda row: row["material"])}


def load_v10_images(texture_set: TextureSet) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for page_id in PAGE_IDS:
        record = texture_set.textures[page_id]
        name = f"Ferrari_TargetedV10_{page_id}_{record.size[0]}_{record.sha256[:8]}"
        require(bpy.data.images.get(name) is None, f"Target V10 image name already exists: {name}")
        image = bpy.data.images.load(str(record.path), check_existing=False)
        require(tuple(int(value) for value in image.size) == record.size, f"Blender-decoded V10 texture size drift: {page_id}")
        image.name = name
        require(image.name == name, f"Blender renamed V10 texture: {page_id}")
        image.colorspace_settings.name = "sRGB"
        image.alpha_mode = "STRAIGHT"
        image["schema"] = IMAGE_SCHEMA
        image["atlas_page_id"] = page_id
        image["atlas_set_id"] = texture_set.asset_id
        image["texture_sha256"] = record.sha256
        image["texture_role"] = "seedream_topaz_targeted_v10_final"
        image["texture_report_sha256"] = texture_set.report_sha256
        image["decoded_pixel_sha256"] = record.decoded_pixel_sha256
        image["production_ready"] = True
        image["test_only"] = False
        image.pack()
        require(packed_sha256(image) == record.sha256, f"Packed V10 PNG differs from source file: {page_id}")
        image.filepath = f"//textures/{record.path.name}"
        images[page_id] = image
    require(len(images) == len(PAGE_IDS), "Did not load all seven V10 images")
    return images


def replace_image_bindings(objects: list[Any], source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    materials = used_materials(objects)
    graphs_before = {material.name: material_signature_sha256(material, include_image=False) for material in materials}
    counts = {page_id: 0 for page_id in PAGE_IDS}
    rows: list[dict[str, Any]] = []
    for material in materials:
        page_id = _page_for_material(material)
        nodes = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage"]
        require(len(nodes) == 1, f"Image-node inventory drift before V10 replacement: {material.name}")
        node = nodes[0]
        require(node.image is source["images"][page_id], f"V9 source binding drift before V10 replacement: {material.name}")
        sampling = [node.interpolation, node.projection, node.extension]
        old_name = node.image.name
        node.image = target[page_id]
        require([node.interpolation, node.projection, node.extension] == sampling, f"Texture sampling changed: {material.name}")
        rows.append(
            {
                "material": material.name,
                "node": node.name,
                "page_id": page_id,
                "old_image": old_name,
                "new_image": target[page_id].name,
                "sampling": sampling,
            }
        )
        counts[page_id] += 1
    require(len(rows) == EXPECTED_IMAGE_BINDINGS, "Did not replace exactly 32 image bindings")
    require(counts == EXPECTED_PAGE_MATERIAL_COUNTS, "V10 material/page counts drift")
    graphs_after = {material.name: material_signature_sha256(material, include_image=False) for material in materials}
    require(graphs_after == graphs_before, "Shader structure/defaults/sampling changed during image replacement")
    return {
        "graph_signatures_before": graphs_before,
        "graph_signatures_after": graphs_after,
        "page_material_counts": counts,
        "replaced_nodes": sorted(rows, key=lambda row: row["material"]),
    }


def remove_source_images(source: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for page_id in PAGE_IDS:
        image = source["images"][page_id]
        require(image.users == 0, f"Pinned V9 image still has users after replacement: {page_id}/{image.name}")
        names.append(image.name)
        bpy.data.images.remove(image)
    require(len(names) == 7 and all(bpy.data.images.get(name) is None for name in names), "A replaced V9 image datablock remains")
    return names


def validate_v10_bindings(texture_set: TextureSet, objects: list[Any], images: dict[str, Any]) -> dict[str, Any]:
    counts = {page_id: 0 for page_id in PAGE_IDS}
    used_images: set[Any] = set()
    rows: list[dict[str, Any]] = []
    for material in used_materials(objects):
        page_id = _page_for_material(material)
        nodes = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage"]
        require(len(nodes) == 1, f"Final V10 image-node inventory drift: {material.name}")
        node = nodes[0]
        require(node.image is images[page_id], f"Wrong final V10 image binding: {material.name}")
        used_images.add(node.image)
        counts[page_id] += 1
        rows.append(
            {
                "material": material.name,
                "node": node.name,
                "page_id": page_id,
                "image": node.image.name,
                "sampling": [node.interpolation, node.projection, node.extension],
            }
        )
    require(len(rows) == EXPECTED_IMAGE_BINDINGS, "Final V10 image-binding count drift")
    require(len(used_images) == len(PAGE_IDS), "Final V10 used-image inventory drift")
    require(counts == EXPECTED_PAGE_MATERIAL_COUNTS, "Final V10 page/material counts drift")
    for page_id, image in images.items():
        record = texture_set.textures[page_id]
        require(tuple(int(value) for value in image.size) == record.size, f"Final V10 image size drift: {page_id}")
        require(packed_sha256(image) == record.sha256, f"Final V10 packed image SHA drift: {page_id}")
        require(image.get("schema") == IMAGE_SCHEMA, f"Final V10 image schema drift: {page_id}")
        require(image.get("atlas_page_id") == page_id, f"Final V10 page metadata drift: {page_id}")
        require(image.get("atlas_set_id") == texture_set.asset_id, f"Final V10 asset metadata drift: {page_id}")
        require(image.get("texture_report_sha256") == texture_set.report_sha256, f"Final V10 report metadata drift: {page_id}")
    return {
        "used_images": sorted(image.name for image in used_images),
        "page_material_counts": counts,
        "nodes": sorted(rows, key=lambda row: row["material"]),
    }


def integrate_texture_only_v10(texture_set: TextureSet) -> dict[str, Any]:
    require(Path(bpy.data.filepath).resolve() == SOURCE_BLEND, "Blender must start from the pinned V9 blend")
    require(sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND], "Pinned V9 blend changed before integration")
    _wrapper, _pieces, objects = car_objects()
    state_before = _car_state(objects)
    require(state_before["meshes"] == EXPECTED_MESHES, "Source mesh count drift")
    require(state_before["triangles"] == EXPECTED_TRIANGLES, "Source triangle count drift")
    materials_before = material_state(objects)
    specialized_before = validate_specialized_shaders(objects)
    camera_world_before = camera_world_state(objects)
    source = source_bindings(objects)
    old_names = {image.name for image in source["images"].values()}
    unaffected_images_before = image_inventory(exclude_names=old_names)
    total_images_before = len(bpy.data.images)

    images = load_v10_images(texture_set)
    replacement = replace_image_bindings(objects, source, images)
    removed_names = remove_source_images(source)

    state_after = _car_state(objects)
    materials_after = material_state(objects)
    specialized_after = validate_specialized_shaders(objects)
    camera_world_after = camera_world_state(objects)
    bindings = validate_v10_bindings(texture_set, objects, images)
    new_names = {image.name for image in images.values()}
    unaffected_images_after = image_inventory(exclude_names=new_names)
    require(state_after == state_before, "Geometry/topology/UV/normals/transforms/material slots changed")
    require(materials_after == materials_before, "Material shader structure or custom properties changed")
    require(specialized_after == specialized_before, "Specialized shader state changed")
    require(camera_world_after == camera_world_before, "Camera/world/environment state changed")
    require(unaffected_images_after == unaffected_images_before, "An image datablock outside the exact seven replacements changed")
    require(len(bpy.data.images) == total_images_before, "Total image datablock count changed")
    require(len(removed_names) == len(images) == 7, "Texture-only replacement did not exchange exactly seven image datablocks")
    require(sha256(texture_set.report_path) == texture_set.report_sha256, "V10 texture report changed during integration")
    require(sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND], "Pinned V9 blend changed during integration")

    return {
        "schema": ASSEMBLY_SCHEMA,
        "status": "passed",
        "mode": "texture_only",
        "source_v9_blend": str(SOURCE_BLEND),
        "source_v9_blend_sha256": SOURCE_PINS[SOURCE_BLEND],
        "texture_report": str(texture_set.report_path),
        "texture_report_sha256": texture_set.report_sha256,
        "texture_report_canonical_sha256": texture_set.report_canonical_sha256,
        "asset_id": texture_set.asset_id,
        "state_before": state_before,
        "state_after": state_after,
        "materials_before": materials_before,
        "materials_after": materials_after,
        "specialized_shaders_before": specialized_before,
        "specialized_shaders_after": specialized_after,
        "camera_world_before": camera_world_before,
        "camera_world_after": camera_world_after,
        "source_bindings": {key: value for key, value in source.items() if key != "images"},
        "image_replacement": replacement,
        "removed_v9_images": removed_names,
        "unaffected_images_before": unaffected_images_before,
        "unaffected_images_after": unaffected_images_after,
        "total_images_before": total_images_before,
        "total_images_after": len(bpy.data.images),
        "v10_bindings": bindings,
        "production_images": {
            page_id: {
                "name": image.name,
                "size": [int(image.size[0]), int(image.size[1])],
                "packed_sha256": packed_sha256(image),
                "source_path": str(texture_set.textures[page_id].path),
                "source_file_sha256": texture_set.textures[page_id].sha256,
            }
            for page_id, image in images.items()
        },
        "hard_gates": {
            "pinned_v9_source_exact": True,
            "exact_66_meshes": state_after["meshes"] == EXPECTED_MESHES,
            "exact_179488_triangles": state_after["triangles"] == EXPECTED_TRIANGLES,
            "geometry_topology_uv_normals_transforms_material_slots_preserved": state_after == state_before,
            "shader_structure_sampling_and_material_properties_preserved": materials_after == materials_before,
            "specialized_shaders_preserved": specialized_after == specialized_before,
            "camera_world_environment_preserved": camera_world_after == camera_world_before,
            "exact_seven_image_datablocks_replaced": len(removed_names) == len(images) == 7,
            "exact_32_image_bindings_replaced": len(replacement["replaced_nodes"]) == EXPECTED_IMAGE_BINDINGS,
            "seven_v10_images_packed": all(
                packed_sha256(images[page_id]) == texture_set.textures[page_id].sha256 for page_id in PAGE_IDS
            ),
            "unaffected_image_datablocks_preserved": unaffected_images_after == unaffected_images_before,
            "image_datablock_count_preserved": len(bpy.data.images) == total_images_before,
            "source_v9_not_overwritten": sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND],
            "texture_report_unchanged": sha256(texture_set.report_path) == texture_set.report_sha256,
            "no_remote_generation_by_builder": True,
        },
    }


def _reopened_v10_images(texture_set: TextureSet) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for page_id in PAGE_IDS:
        record = texture_set.textures[page_id]
        candidates = [
            image
            for image in bpy.data.images
            if image.get("schema") == IMAGE_SCHEMA
            and image.get("atlas_page_id") == page_id
            and image.get("atlas_set_id") == texture_set.asset_id
            and image.get("texture_sha256") == record.sha256
        ]
        require(len(candidates) == 1, f"Reopened V10 image inventory drift: {page_id}")
        images[page_id] = candidates[0]
    return images


def audit_reopened_texture_only_v10(
    texture_set: TextureSet,
    assembly: dict[str, Any],
    opened_blend: Path,
) -> dict[str, Any]:
    require(assembly.get("schema") == ASSEMBLY_SCHEMA and assembly.get("status") == "passed", "Assembly schema/status drift")
    require(all(assembly.get("hard_gates", {}).values()), "Assembly report contains a failed gate")
    require(assembly.get("texture_report_sha256") == texture_set.report_sha256, "Assembly used another texture report")
    _wrapper, _pieces, objects = car_objects()
    state = _car_state(objects)
    materials = material_state(objects)
    specialized = validate_specialized_shaders(objects)
    camera_world = camera_world_state(objects)
    require(state == assembly.get("state_after"), "Car structure/UV/normals/transforms/material slots drifted after reopen")
    require(materials == assembly.get("materials_after"), "Material state drifted after reopen")
    require(specialized == assembly.get("specialized_shaders_after"), "Specialized shader state drifted after reopen")
    require(camera_world == assembly.get("camera_world_after"), "Camera/world/environment drifted after reopen")
    images = _reopened_v10_images(texture_set)
    bindings = validate_v10_bindings(texture_set, objects, images)
    new_names = {image.name for image in images.values()}
    unaffected_images = image_inventory(exclude_names=new_names)
    require(unaffected_images == assembly.get("unaffected_images_after"), "Unaffected image datablocks drifted after reopen")
    require(len(bpy.data.images) == assembly.get("total_images_after"), "Image datablock count drifted after reopen")
    require(all(bpy.data.images.get(name) is None for name in assembly.get("removed_v9_images", [])), "A replaced V9 image returned after reopen")
    require(sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND], "Pinned V9 blend changed after reopen")
    require(sha256(texture_set.report_path) == texture_set.report_sha256, "V10 texture report changed after reopen")
    return {
        "schema": REOPEN_SCHEMA,
        "status": "passed",
        "mode": "texture_only",
        "opened_blend": str(opened_blend),
        "opened_blend_sha256": sha256(opened_blend),
        "source_v9_blend": str(SOURCE_BLEND),
        "source_v9_blend_sha256": SOURCE_PINS[SOURCE_BLEND],
        "texture_report": str(texture_set.report_path),
        "texture_report_sha256": texture_set.report_sha256,
        "asset_id": texture_set.asset_id,
        "state": state,
        "materials": materials,
        "specialized_shaders": specialized,
        "camera_world": camera_world,
        "image_bindings": bindings,
        "unaffected_images": unaffected_images,
        "total_images": len(bpy.data.images),
        "hard_gates": {
            "assembly_report_passed": True,
            "geometry_topology_uv_normals_transforms_material_slots_exact_after_reopen": state == assembly.get("state_after"),
            "shader_structure_sampling_and_material_properties_exact_after_reopen": materials == assembly.get("materials_after"),
            "specialized_shaders_exact_after_reopen": specialized == assembly.get("specialized_shaders_after"),
            "camera_world_environment_exact_after_reopen": camera_world == assembly.get("camera_world_after"),
            "seven_v10_images_packed_after_reopen": len(images) == 7,
            "exact_32_v10_image_bindings_after_reopen": len(bindings["nodes"]) == EXPECTED_IMAGE_BINDINGS,
            "old_v9_images_absent_after_reopen": all(
                bpy.data.images.get(name) is None for name in assembly.get("removed_v9_images", [])
            ),
            "unaffected_image_datablocks_exact_after_reopen": unaffected_images == assembly.get("unaffected_images_after"),
            "image_datablock_count_exact_after_reopen": len(bpy.data.images) == assembly.get("total_images_after"),
            "source_v9_not_overwritten_after_reopen": sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND],
            "texture_report_unchanged_after_reopen": sha256(texture_set.report_path) == texture_set.report_sha256,
        },
    }
