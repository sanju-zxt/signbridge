import bpy
import glob
import os

fbx_files = glob.glob(r"C:\Users\drx20\Downloads\3d model\*.fbx")

for f in fbx_files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=f)
    name = os.path.basename(f)
    arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
    act = arm.animation_data.action if arm.animation_data else None
    print(f"=== {name} ===")
    if act:
        print(f"  Action name: {act.name}")
        print(f"  Frame range: {act.frame_range[0]} to {act.frame_range[1]}")
    else:
        print("  NO ACTION FOUND")
