#!/usr/bin/env python3
"""Build, reopen-audit, and promote the additive texture-only V11 full-bleed blend."""

from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR)) if str(SCRIPT_DIR) not in sys.path else None
from targeted_final_v11_common import ASSEMBLY_SCHEMA, BUILD_SCHEMA, EXPECTED_IMAGE_BINDINGS, OUTPUT_ROOT, PAGE_IDS, REOPEN_SCHEMA, SOURCE_BLEND, SOURCE_PINS, WORK_ROOT, load_json, load_texture_set, preflight_summary, safe_output_path, sha256  # noqa: E402

BLENDER=Path("blender")
ASSEMBLER=SCRIPT_DIR/"assemble_targeted_final_v11_blender.py"; AUDITOR=SCRIPT_DIR/"audit_targeted_final_v11_blender.py"
DEFAULT_NAME="ferrari_targeted_reunwrap_seedream_v11_full_bleed_final.blend"

def run(command:list[str])->None: subprocess.run(command,check=True)

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-name",default=DEFAULT_NAME); parser.add_argument("--blender",type=Path,default=BLENDER); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args()
    try: textures=load_texture_set()
    except Exception as exc: print(f"TARGETED_FINAL_V11_BUILD_REFUSED: {exc}",file=sys.stderr); return 2
    if args.preflight_only: print("TARGETED_FINAL_V11_BUILD_PREFLIGHT_PASSED="+json.dumps(preflight_summary(textures),sort_keys=True)); return 0
    try: output=safe_output_path(args.output_name)
    except Exception as exc: print(f"TARGETED_FINAL_V11_BUILD_REFUSED: {exc}",file=sys.stderr); return 2
    blender=args.blender.expanduser().resolve()
    if not blender.is_file(): print(f"TARGETED_FINAL_V11_BUILD_REFUSED: Blender missing: {blender}",file=sys.stderr); return 2
    stem=output.stem; assembly_output=output.with_name(stem+".assembly_report.json"); audit_output=output.with_name(stem+".reopen_audit.json"); build_output=output.with_name(stem+".build_report.json")
    existing=[str(p) for p in (output,assembly_output,audit_output,build_output) if p.exists()]
    if existing: print("TARGETED_FINAL_V11_BUILD_REFUSED: refusing overwrite: "+", ".join(existing),file=sys.stderr); return 2
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); WORK_ROOT.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="targeted_final_v11_",dir=WORK_ROOT) as temporary:
            work=Path(temporary).resolve(); staged=work/"staged_v11.blend"; staged_a=work/"assembly.json"; staged_r=work/"reopen.json"; staged_b=work/"build.json"
            run([str(blender),"-b",str(SOURCE_BLEND),"--python",str(ASSEMBLER),"--","--staged-blend",str(staged),"--report",str(staged_a)])
            run([str(blender),"-b",str(staged),"--python",str(AUDITOR),"--","--staged-blend",str(staged),"--assembly-report",str(staged_a),"--report",str(staged_r)])
            assembly=load_json(staged_a); audit=load_json(staged_r)
            if assembly.get("schema")!=ASSEMBLY_SCHEMA or assembly.get("status")!="passed" or not all(assembly.get("hard_gates",{}).values()): raise RuntimeError("Staged V11 assembly failed")
            if audit.get("schema")!=REOPEN_SCHEMA or audit.get("status")!="passed" or not all(audit.get("hard_gates",{}).values()): raise RuntimeError("Staged V11 reopen failed")
            blend_sha=sha256(staged)
            if blend_sha!=assembly.get("staged_blend_sha256") or blend_sha!=audit.get("opened_blend_sha256"): raise RuntimeError("V11 staged hash disagreement")
            if sha256(SOURCE_BLEND)!=SOURCE_PINS[SOURCE_BLEND]: raise RuntimeError("Pinned V10 changed")
            if sha256(textures.generator_report_path)!=textures.generator_report_sha256 or sha256(textures.independent_audit_path)!=textures.independent_audit_sha256: raise RuntimeError("V11 proof changed")
            report={
                "schema":BUILD_SCHEMA,"status":"passed","mode":"texture_only_full_bleed","final_blend":str(output),"final_blend_sha256":blend_sha,
                "source_v10_blend":str(SOURCE_BLEND),"source_v10_blend_sha256":SOURCE_PINS[SOURCE_BLEND],"generator_report":str(textures.generator_report_path),"generator_report_sha256":textures.generator_report_sha256,
                "independent_audit":str(textures.independent_audit_path),"independent_audit_sha256":textures.independent_audit_sha256,"asset_id":textures.asset_id,
                "textures":{p:{"path":str(r.path),"sha256":r.sha256,"size":list(r.size),"bytes":r.bytes} for p,r in textures.textures.items()},
                "assembly_report":str(assembly_output),"assembly_report_sha256":sha256(staged_a),"reopen_audit":str(audit_output),"reopen_audit_sha256":sha256(staged_r),
                "source_blend_overwritten":False,"geometry_topology_uv_normals_transforms_material_slots_changed":False,"shader_structure_or_sampling_changed":False,"camera_or_world_changed":False,"rendered":False,"remote_generation_called":False,
                "hard_gates":{
                    "generator_report_passed_and_pinned":True,"independent_audit_passed_and_pinned":True,"independent_zero_occupied_rgb_mismatch":textures.audit_raw["summary"]["total_occupied_rgb_mismatch_texels"]==0,
                    "exact_seven_v11_textures":len(textures.textures)==7,"assembly_passed":True,"save_reopen_audit_passed":True,"staged_hashes_agree":True,
                    "exact_seven_image_datablocks_replaced":assembly["hard_gates"]["exact_seven_image_datablocks_replaced"],"exact_32_image_bindings_replaced":len(assembly["image_replacement"]["replaced_nodes"])==EXPECTED_IMAGE_BINDINGS,
                    "seven_images_packed":len(assembly["production_images"])==len(PAGE_IDS),"geometry_topology_uv_normals_transforms_material_slots_preserved":assembly["hard_gates"]["geometry_topology_uv_normals_transforms_material_slots_preserved"],
                    "shader_structure_sampling_and_properties_preserved":assembly["hard_gates"]["shader_structure_sampling_and_material_properties_preserved"],"camera_world_environment_preserved":assembly["hard_gates"]["camera_world_environment_preserved"],
                    "source_v10_not_overwritten":sha256(SOURCE_BLEND)==SOURCE_PINS[SOURCE_BLEND],"no_remote_generation_by_builder":True,"no_overwrite":True}}
            if not all(report["hard_gates"].values()): raise RuntimeError("V11 build report gate failed")
            staged_b.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            os.replace(staged_a,assembly_output); os.replace(staged_r,audit_output); os.replace(staged_b,build_output); os.replace(staged,output)
        if sha256(output)!=blend_sha: raise RuntimeError("Promoted V11 blend hash differs")
    except (OSError,RuntimeError,subprocess.CalledProcessError) as exc: print(f"TARGETED_FINAL_V11_BUILD_FAILED_NO_APPROVED_OUTPUT: {exc}",file=sys.stderr); return 1
    print("TARGETED_FINAL_V11_BUILD_PASSED="+json.dumps({"blend":str(output),"sha256":blend_sha,"assembly_report":str(assembly_output),"reopen_audit":str(audit_output),"build_report":str(build_output)},sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
