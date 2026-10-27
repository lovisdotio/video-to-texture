#!/usr/bin/env python3
"""Blender-only implementation for strict texture-only V11 full-bleed integration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
V10_ROOT = V8_ROOT / "targeted_final_v10"
FINAL_ASSEMBLER = V8_ROOT / "integration_v8/final_assembler_v1"
for value in (str(SCRIPT_DIR), str(V10_ROOT), str(FINAL_ASSEMBLER)):
    if value not in sys.path:
        sys.path.insert(0, value)

from blender_final_v8 import material_signature_sha256, packed_sha256, used_materials, validate_specialized_shaders  # noqa: E402
from contract_final_v8 import BODY_MATERIAL_NAME  # noqa: E402
from targeted_final_v10_blender import (  # noqa: E402
    _car_state,
    _page_for_material,
    camera_world_state,
    car_objects,
    image_inventory,
    material_state,
)
from targeted_final_v11_common import (  # noqa: E402
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
    load_json,
    require,
    sha256,
)


def source_bindings(objects: list[Any]) -> dict[str, Any]:
    source_assembly = load_json(SOURCE_ASSEMBLY_REPORT)
    production = source_assembly.get("production_images", {})
    require(set(production) == set(PAGE_IDS), "Pinned V10 production-image report drift")
    images: dict[str, Any] = {}
    for page in PAGE_IDS:
        record = production[page]
        image = bpy.data.images.get(record.get("name"))
        require(image is not None, f"Pinned V10 image datablock missing: {page}")
        require(tuple(int(v) for v in image.size) == tuple(record.get("size", [])), f"Pinned V10 image size drift: {page}")
        require(packed_sha256(image) == record.get("packed_sha256"), f"Pinned V10 packed image drift: {page}")
        require(image.get("atlas_page_id") == page, f"Pinned V10 image metadata drift: {page}")
        images[page] = image
    counts = {page: 0 for page in PAGE_IDS}
    rows = []
    for material in used_materials(objects):
        page = _page_for_material(material)
        nodes = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage"]
        require(len(nodes) == 1, f"Pinned V10 image-node inventory drift: {material.name}")
        node = nodes[0]
        if material.name != BODY_MATERIAL_NAME:
            require(node.name == "V8_ATLAS_IMAGE", f"Pinned V10 image node name drift: {material.name}")
        require(node.image is images[page], f"Pinned V10 image binding drift: {material.name}")
        rows.append({"material": material.name, "node": node.name, "page_id": page, "image": node.image.name, "sampling": [node.interpolation, node.projection, node.extension]})
        counts[page] += 1
    require(len(rows) == EXPECTED_IMAGE_BINDINGS and counts == EXPECTED_PAGE_MATERIAL_COUNTS, "Pinned V10 binding partition drift")
    return {"images": images, "page_material_counts": counts, "nodes": sorted(rows, key=lambda row: row["material"])}


def load_v11_images(texture_set: TextureSet) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for page in PAGE_IDS:
        record = texture_set.textures[page]
        name = f"Ferrari_TargetedV11_{page}_{record.size[0]}_{record.sha256[:8]}"
        require(bpy.data.images.get(name) is None, f"Target V11 image name already exists: {name}")
        image = bpy.data.images.load(str(record.path), check_existing=False)
        require(tuple(int(v) for v in image.size) == record.size, f"Blender-decoded V11 size drift: {page}")
        image.name = name
        require(image.name == name, f"Blender renamed V11 texture: {page}")
        image.colorspace_settings.name = "sRGB"
        image.alpha_mode = "STRAIGHT"
        image["schema"] = IMAGE_SCHEMA
        image["atlas_page_id"] = page
        image["atlas_set_id"] = texture_set.asset_id
        image["texture_sha256"] = record.sha256
        image["texture_role"] = "v11_full_bleed_v10_occupied_byte_exact"
        image["generator_report_sha256"] = texture_set.generator_report_sha256
        image["independent_audit_sha256"] = texture_set.independent_audit_sha256
        image["decoded_pixel_sha256"] = record.decoded_pixel_sha256
        image["production_ready"] = True
        image["test_only"] = False
        image.pack()
        require(packed_sha256(image) == record.sha256, f"Packed V11 PNG differs from source: {page}")
        image.filepath = f"//textures/{record.path.name}"
        images[page] = image
    require(len(images) == 7, "Did not load all seven V11 images")
    return images


def replace_bindings(objects: list[Any], source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    materials = used_materials(objects)
    graphs_before = {m.name: material_signature_sha256(m, include_image=False) for m in materials}
    counts = {page: 0 for page in PAGE_IDS}
    rows = []
    for material in materials:
        page = _page_for_material(material)
        nodes = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage"]
        require(len(nodes) == 1, f"V10 image-node inventory drift before V11: {material.name}")
        node = nodes[0]
        require(node.image is source["images"][page], f"V10 source binding drift before V11: {material.name}")
        sampling = [node.interpolation, node.projection, node.extension]
        old_name = node.image.name
        node.image = target[page]
        require([node.interpolation, node.projection, node.extension] == sampling, f"Sampling changed: {material.name}")
        rows.append({"material": material.name, "node": node.name, "page_id": page, "old_image": old_name, "new_image": target[page].name, "sampling": sampling})
        counts[page] += 1
    graphs_after = {m.name: material_signature_sha256(m, include_image=False) for m in materials}
    require(graphs_after == graphs_before, "Shader structure/defaults/sampling changed during V11 replacement")
    require(len(rows) == EXPECTED_IMAGE_BINDINGS and counts == EXPECTED_PAGE_MATERIAL_COUNTS, "V11 replacement binding partition drift")
    return {"graph_signatures_before": graphs_before, "graph_signatures_after": graphs_after, "page_material_counts": counts, "replaced_nodes": sorted(rows, key=lambda row: row["material"])}


def remove_source_images(source: dict[str, Any]) -> list[str]:
    names = []
    for page in PAGE_IDS:
        image = source["images"][page]
        require(image.users == 0, f"V10 source image still used: {page}/{image.name}")
        names.append(image.name)
        bpy.data.images.remove(image)
    require(len(names) == 7 and all(bpy.data.images.get(name) is None for name in names), "A replaced V10 image remains")
    return names


def validate_v11_bindings(texture_set: TextureSet, objects: list[Any], images: dict[str, Any]) -> dict[str, Any]:
    counts = {page: 0 for page in PAGE_IDS}
    used = set()
    rows = []
    for material in used_materials(objects):
        page = _page_for_material(material)
        nodes = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage"]
        require(len(nodes) == 1 and nodes[0].image is images[page], f"Wrong final V11 binding: {material.name}")
        node = nodes[0]
        used.add(node.image)
        counts[page] += 1
        rows.append({"material": material.name, "node": node.name, "page_id": page, "image": node.image.name, "sampling": [node.interpolation, node.projection, node.extension]})
    require(len(rows) == EXPECTED_IMAGE_BINDINGS and len(used) == 7 and counts == EXPECTED_PAGE_MATERIAL_COUNTS, "Final V11 binding inventory drift")
    for page, image in images.items():
        record = texture_set.textures[page]
        require(tuple(int(v) for v in image.size) == record.size and packed_sha256(image) == record.sha256, f"Final V11 packed image drift: {page}")
        require(image.get("schema") == IMAGE_SCHEMA and image.get("atlas_page_id") == page and image.get("atlas_set_id") == texture_set.asset_id, f"Final V11 image metadata drift: {page}")
        require(image.get("generator_report_sha256") == texture_set.generator_report_sha256, f"Final V11 generator pin drift: {page}")
        require(image.get("independent_audit_sha256") == texture_set.independent_audit_sha256, f"Final V11 audit pin drift: {page}")
    return {"used_images": sorted(image.name for image in used), "page_material_counts": counts, "nodes": sorted(rows, key=lambda row: row["material"])}


def integrate(texture_set: TextureSet) -> dict[str, Any]:
    require(Path(bpy.data.filepath).resolve() == SOURCE_BLEND, "Blender must start from pinned V10 blend")
    require(sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND], "Pinned V10 blend changed before V11 integration")
    _wrapper, _pieces, objects = car_objects()
    state_before = _car_state(objects)
    require(state_before["meshes"] == EXPECTED_MESHES and state_before["triangles"] == EXPECTED_TRIANGLES, "Source V10 inventory drift")
    materials_before = material_state(objects)
    specialized_before = validate_specialized_shaders(objects)
    camera_world_before = camera_world_state(objects)
    source = source_bindings(objects)
    old_names = {image.name for image in source["images"].values()}
    unaffected_before = image_inventory(exclude_names=old_names)
    total_before = len(bpy.data.images)

    images = load_v11_images(texture_set)
    replacement = replace_bindings(objects, source, images)
    removed = remove_source_images(source)

    state_after = _car_state(objects)
    materials_after = material_state(objects)
    specialized_after = validate_specialized_shaders(objects)
    camera_world_after = camera_world_state(objects)
    bindings = validate_v11_bindings(texture_set, objects, images)
    unaffected_after = image_inventory(exclude_names={image.name for image in images.values()})
    require(state_after == state_before, "Geometry/topology/UV/normals/transforms/material slots changed")
    require(materials_after == materials_before, "Material structure/properties changed")
    require(specialized_after == specialized_before, "Specialized shaders changed")
    require(camera_world_after == camera_world_before, "Camera/world/environment changed")
    require(unaffected_after == unaffected_before and len(bpy.data.images) == total_before, "Image inventory outside exact seven replacements changed")
    require(sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND], "Pinned V10 changed during integration")
    require(sha256(texture_set.generator_report_path) == texture_set.generator_report_sha256, "V11 generator report changed")
    require(sha256(texture_set.independent_audit_path) == texture_set.independent_audit_sha256, "V11 independent audit changed")
    return {
        "schema": ASSEMBLY_SCHEMA, "status": "passed", "mode": "texture_only_full_bleed",
        "source_v10_blend": str(SOURCE_BLEND), "source_v10_blend_sha256": SOURCE_PINS[SOURCE_BLEND],
        "generator_report": str(texture_set.generator_report_path), "generator_report_sha256": texture_set.generator_report_sha256,
        "independent_audit": str(texture_set.independent_audit_path), "independent_audit_sha256": texture_set.independent_audit_sha256,
        "asset_id": texture_set.asset_id,
        "state_before": state_before, "state_after": state_after,
        "materials_before": materials_before, "materials_after": materials_after,
        "specialized_shaders_before": specialized_before, "specialized_shaders_after": specialized_after,
        "camera_world_before": camera_world_before, "camera_world_after": camera_world_after,
        "source_bindings": {k: v for k, v in source.items() if k != "images"},
        "image_replacement": replacement, "removed_v10_images": removed,
        "unaffected_images_before": unaffected_before, "unaffected_images_after": unaffected_after,
        "total_images_before": total_before, "total_images_after": len(bpy.data.images),
        "v11_bindings": bindings,
        "production_images": {page: {"name": image.name, "size": [int(image.size[0]), int(image.size[1])], "packed_sha256": packed_sha256(image), "source_path": str(texture_set.textures[page].path)} for page, image in images.items()},
        "hard_gates": {
            "pinned_v10_source_exact": True,
            "generator_and_independent_audit_passed_and_pinned": True,
            "independent_zero_occupied_rgb_mismatch": texture_set.audit_raw["summary"]["total_occupied_rgb_mismatch_texels"] == 0,
            "independent_zero_background_donor_mismatch": texture_set.audit_raw["summary"]["total_background_nearest_donor_mismatch_texels"] == 0,
            "exact_66_meshes": state_after["meshes"] == EXPECTED_MESHES,
            "exact_179488_triangles": state_after["triangles"] == EXPECTED_TRIANGLES,
            "geometry_topology_uv_normals_transforms_material_slots_preserved": state_after == state_before,
            "shader_structure_sampling_and_material_properties_preserved": materials_after == materials_before,
            "specialized_shaders_preserved": specialized_after == specialized_before,
            "camera_world_environment_preserved": camera_world_after == camera_world_before,
            "exact_seven_image_datablocks_replaced": len(removed) == len(images) == 7,
            "exact_32_image_bindings_replaced": len(replacement["replaced_nodes"]) == EXPECTED_IMAGE_BINDINGS,
            "seven_v11_images_packed": all(packed_sha256(images[p]) == texture_set.textures[p].sha256 for p in PAGE_IDS),
            "unaffected_image_datablocks_preserved": unaffected_after == unaffected_before,
            "image_datablock_count_preserved": len(bpy.data.images) == total_before,
            "source_v10_not_overwritten": sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND],
            "no_remote_generation_by_builder": True,
        },
    }


def reopened_images(texture_set: TextureSet) -> dict[str, Any]:
    images = {}
    for page in PAGE_IDS:
        record = texture_set.textures[page]
        candidates = [image for image in bpy.data.images if image.get("schema") == IMAGE_SCHEMA and image.get("atlas_page_id") == page and image.get("atlas_set_id") == texture_set.asset_id and image.get("texture_sha256") == record.sha256]
        require(len(candidates) == 1, f"Reopened V11 image inventory drift: {page}")
        images[page] = candidates[0]
    return images


def audit_reopened(texture_set: TextureSet, assembly: dict[str, Any], opened_blend: Path) -> dict[str, Any]:
    require(assembly.get("schema") == ASSEMBLY_SCHEMA and assembly.get("status") == "passed" and all(assembly.get("hard_gates", {}).values()), "V11 assembly report failed")
    _wrapper, _pieces, objects = car_objects()
    state = _car_state(objects); materials = material_state(objects); specialized = validate_specialized_shaders(objects); camera_world = camera_world_state(objects)
    require(state == assembly.get("state_after"), "Car state drift after V11 reopen")
    require(materials == assembly.get("materials_after"), "Material state drift after V11 reopen")
    require(specialized == assembly.get("specialized_shaders_after"), "Specialized shader drift after V11 reopen")
    require(camera_world == assembly.get("camera_world_after"), "Camera/world drift after V11 reopen")
    images = reopened_images(texture_set)
    bindings = validate_v11_bindings(texture_set, objects, images)
    unaffected = image_inventory(exclude_names={image.name for image in images.values()})
    require(unaffected == assembly.get("unaffected_images_after") and len(bpy.data.images) == assembly.get("total_images_after"), "Image inventory drift after V11 reopen")
    require(all(bpy.data.images.get(name) is None for name in assembly.get("removed_v10_images", [])), "A replaced V10 image returned")
    require(sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND], "Pinned V10 source changed after reopen")
    require(sha256(texture_set.generator_report_path) == texture_set.generator_report_sha256 and sha256(texture_set.independent_audit_path) == texture_set.independent_audit_sha256, "V11 proof drift after reopen")
    return {
        "schema": REOPEN_SCHEMA, "status": "passed", "mode": "texture_only_full_bleed",
        "opened_blend": str(opened_blend), "opened_blend_sha256": sha256(opened_blend),
        "source_v10_blend": str(SOURCE_BLEND), "source_v10_blend_sha256": SOURCE_PINS[SOURCE_BLEND],
        "generator_report": str(texture_set.generator_report_path), "generator_report_sha256": texture_set.generator_report_sha256,
        "independent_audit": str(texture_set.independent_audit_path), "independent_audit_sha256": texture_set.independent_audit_sha256,
        "asset_id": texture_set.asset_id, "state": state, "materials": materials, "specialized_shaders": specialized,
        "camera_world": camera_world, "image_bindings": bindings, "unaffected_images": unaffected, "total_images": len(bpy.data.images),
        "hard_gates": {
            "assembly_report_passed": True,
            "geometry_topology_uv_normals_transforms_material_slots_exact_after_reopen": state == assembly.get("state_after"),
            "shader_structure_sampling_and_material_properties_exact_after_reopen": materials == assembly.get("materials_after"),
            "specialized_shaders_exact_after_reopen": specialized == assembly.get("specialized_shaders_after"),
            "camera_world_environment_exact_after_reopen": camera_world == assembly.get("camera_world_after"),
            "seven_v11_images_packed_after_reopen": len(images) == 7,
            "exact_32_v11_bindings_after_reopen": len(bindings["nodes"]) == EXPECTED_IMAGE_BINDINGS,
            "old_v10_images_absent_after_reopen": all(bpy.data.images.get(name) is None for name in assembly.get("removed_v10_images", [])),
            "unaffected_image_datablocks_exact_after_reopen": unaffected == assembly.get("unaffected_images_after"),
            "source_v10_not_overwritten_after_reopen": sha256(SOURCE_BLEND) == SOURCE_PINS[SOURCE_BLEND],
            "generator_and_independent_audit_pins_exact_after_reopen": True,
        },
    }
